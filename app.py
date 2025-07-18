from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model("unet_model.h5", compile=False)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    image = Image.open(file.stream).convert("L").resize((128, 128))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    
    pred = model.predict(img_array)
    result = np.argmax(pred[0], axis=-1).tolist()
    
    return jsonify({"mask": result})

@app.route("/")
def home():
    return "UNet Flask API is running!"
