import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MODEL_PATH = "pneumonia_detector_efficientnet.keras"

model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print(f"[PneumoScan] Model loaded from {MODEL_PATH}")
else:
    print(f"[PneumoScan] WARNING: Model file not found.")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        try:
            if "file" not in request.files:
                return render_template("upload.html", error="No file selected.")
            img_file = request.files["file"]
            if img_file.filename == "":
                return render_template("upload.html", error="No file selected.")
            if not allowed_file(img_file.filename):
                return render_template("upload.html", error="Invalid file type. Please upload JPG, PNG, or WebP.")
            if model is None:
                return render_template("upload.html", error="Model not loaded. Please check server configuration.")

            filename = secure_filename(img_file.filename)
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            img_file.save(img_path)

            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            confidence = float(prediction[0][0])

            if confidence > 0.5:
                result = "Pneumonia"
                confidence_pct = round(confidence * 100, 1)
            else:
                result = "Normal"
                confidence_pct = round((1 - confidence) * 100, 1)

            if os.path.exists(img_path):
                os.remove(img_path)

            return render_template("upload.html", prediction=result, confidence=confidence_pct)

        except Exception as e:
            return render_template("upload.html", error=f"Server error: {str(e)}")

    return render_template("upload.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host="0.0.0.0", port=port)
