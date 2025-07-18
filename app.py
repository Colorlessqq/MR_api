from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import nibabel as nib
import tempfile
import os
import cv2
from keras.models import load_model
from werkzeug.utils import secure_filename
from io import BytesIO
import zipfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

MODEL_PATH = "unet_model.h5"
model = load_model(MODEL_PATH, compile=False)

TARGET_SIZE = (128, 128)


def preprocess_slice(slice_2d):
    resized = cv2.resize(slice_2d, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / (np.max(resized) + 1e-6)
    return np.expand_dims(normalized, axis=(0, -1))  # shape (1, 128, 128, 1)


def predict_volume(volume_3d):
    pred_slices = []

    for i in range(volume_3d.shape[2]):
        slice_2d = volume_3d[:, :, i]
        input_tensor = preprocess_slice(slice_2d)
        prediction = model.predict(input_tensor)[0]
        pred_class = np.argmax(prediction, axis=-1).astype(np.uint8)
        pred_resized = cv2.resize(pred_class, (volume_3d.shape[0], volume_3d.shape[1]), interpolation=cv2.INTER_NEAREST)
        pred_slices.append(pred_resized)

    return np.stack(pred_slices, axis=-1)  # shape (H, W, D)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename.lower())

    if filename.endswith('.nii') or filename.endswith('.nii.gz'):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            volume = nib.load(tmp.name).get_fdata()
            os.unlink(tmp.name)

        pred_mask = predict_volume(volume)

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


@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>Upload NIfTI File</title>
    <h1>Upload .nii or .nii.gz file for segmentation</h1>
    <form action="/predict" method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
