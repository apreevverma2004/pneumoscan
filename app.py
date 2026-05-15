import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Load the trained EfficientNet model once at startup
MODEL_PATH = 'pneumonia_detector_efficientnet.keras'
model = None

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print(f"[PneumoScan] Model loaded from {MODEL_PATH}")
else:
    print(f"[PneumoScan] WARNING: Model file '{MODEL_PATH}' not found. Predictions will be unavailable.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path):
    """Load, resize, normalize image for EfficientNetB0."""
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Validate file presence
        if 'file' not in request.files:
            return render_template('upload.html', error='No file selected.')

        img_file = request.files['file']

        if img_file.filename == '':
            return render_template('upload.html', error='No file selected.')

        if not allowed_file(img_file.filename):
            return render_template('upload.html', error='Invalid file type. Please upload JPG, PNG, or WebP.')

        if model is None:
            return render_template('upload.html', error='Model not loaded. Please place the .keras model file in the project root.')

        # Save uploaded file securely
        filename = secure_filename(img_file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img_file.save(img_path)

        # Preprocess and predict
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])

        if confidence > 0.5:
            result = 'Pneumonia'
            confidence_pct = round(confidence * 100, 1)
        else:
            result = 'Normal'
            confidence_pct = round((1 - confidence) * 100, 1)

        # Clean up uploaded file
        os.remove(img_path)

        return render_template(
            'upload.html',
            prediction=result,
            confidence=confidence_pct
        )

    return render_template('upload.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5001)
