from flask import Flask, request, jsonify, send_file
import numpy as np
import nibabel as nib
import tempfile
import os
import cv2
from keras.models import load_model
from werkzeug.utils import secure_filename
from io import BytesIO
import zipfile
from flask import Flask, request, send_file, jsonify, render_template_string
import io
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

# Load the model once at startup
MODEL_PATH = "unet_model.h5"
model = load_model(MODEL_PATH, compile=False)

TARGET_SIZE = (128, 128)


def preprocess_slice(slice_2d):
    resized = cv2.resize(slice_2d, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / (np.max(resized) + 1e-6)
    return np.expand_dims(normalized, axis=(0, -1))  # (1, 128, 128, 1)


def predict_volume(volume_3d):
    pred_slices = []

    for i in range(volume_3d.shape[2]):
        slice_2d = volume_3d[:, :, i]
        input_tensor = preprocess_slice(slice_2d)
        prediction = model.predict(input_tensor)[0]
        pred_class = np.argmax(prediction, axis=-1).astype(np.uint8)
        pred_resized = cv2.resize(pred_class, (volume_3d.shape[0], volume_3d.shape[1]), interpolation=cv2.INTER_NEAREST)
        pred_slices.append(pred_resized)

    return np.stack(pred_slices, axis=-1)  # (H, W, D)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename.lower())

    if filename.endswith('.nii') or filename.endswith('.nii.gz'):
        # Load 3D volume
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            volume = nib.load(tmp.name).get_fdata()
            os.unlink(tmp.name)

        pred_mask = predict_volume(volume)

        # Save slices as PNG using OpenCV
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for i in range(pred_mask.shape[2]):
                mask_2d = (pred_mask[:, :, i] * 127).astype(np.uint8)
                success, buffer = cv2.imencode('.png', mask_2d)
                if success:
                    zf.writestr(f"slice_{i:03d}.png", buffer.tobytes())

        memory_file.seek(0)
        return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name='predictions.zip')

    elif filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
        # Load 2D slice
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        input_tensor = preprocess_slice(image)
        prediction = model.predict(input_tensor)[0]
        pred_class = np.argmax(prediction, axis=-1).astype(np.uint8)
        pred_resized = cv2.resize(pred_class, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        output_image = (pred_resized * 127).astype(np.uint8)

        _, buffer = cv2.imencode('.png', output_image)
        return send_file(BytesIO(buffer.tobytes()), mimetype='image/png', as_attachment=True, download_name='prediction.png')

    else:
        return jsonify({'error': 'Unsupported file format'}), 400


@app.route("/", methods=["GET"])
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Liver Segmentation API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; background: #f9f9f9; }
            h1 { color: #3c4f76; }
            code { background: #eee; padding: 2px 4px; border-radius: 4px; }
            pre { background: #f0f0f0; padding: 10px; border-radius: 5px; overflow-x: auto; }
            .endpoint { color: #333; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>🧠 Liver Segmentation API</h1>
        <p>Bu API, karaciğer MR görüntüsünden bir <strong>slice (PNG dosyası)</strong> alır ve segmentasyon sonucunu döner.</p>
        
        <h2>🔗 API Adresi</h2>
        <p class="endpoint">POST <code>https://mr-api-mqfu.onrender.com/predict</code></p>

        <h2>📤 Girdi</h2>
        <ul>
            <li><code>multipart/form-data</code> formatında bir PNG dosyası</li>
            <li>Form alan adı: <code>file</code></li>
        </ul>

        <h2>🧪 Örnek curl</h2>
        <pre><code>curl -X POST https://mr-api-mqfu.onrender.com/predict -F "file=@slice.png"</code></pre>

        <h2>🧪 Frontend JavaScript örneği</h2>
        <pre><code>
const formData = new FormData();
formData.append("file", selectedFile);
fetch("https://mr-api-mqfu.onrender.com/predict", {
    method: "POST",
    body: formData
})
.then(response => response.blob())
.then(blob => {
    const imgUrl = URL.createObjectURL(blob);
    document.getElementById("result").src = imgUrl;
});
        </code></pre>

        <h2>📥 Yanıt</h2>
        <p>Segmentasyon sonucu bir <code>image/png</code> dosyası olarak döner.</p>

        <h2>⚠️ Hatalar</h2>
        <ul>
            <li><code>Unsupported file format</code> – PNG dışı dosya yüklendi</li>
            <li><code>File missing</code> – <code>file</code> alanı eksik</li>
            <li><code>Prediction failed</code> – Model hata verdi</li>
        </ul>

        <h2>👨‍💻 Geliştirici</h2>
        <p><strong>Ahmet Ağan</strong> – Yapay zeka & entegrasyon</p>
    </body>
    </html>
    """)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
