# PneumoScan AI — Chest X-Ray Pneumonia Detector

AI-assisted medical image analysis system using EfficientNet + Flask + TensorFlow.

---

## Project Structure

```
pneumoscan/
├── app.py                              ← Flask backend
├── requirements.txt                    ← Python dependencies
├── pneumonia_detector_efficientnet.keras  ← Your trained model (place here)
├── uploads/                            ← Temp folder (auto-created)
└── templates/
    └── upload.html                     ← Frontend UI
```

---

## Setup & Run (Local Development)

### Step 1 — Create a virtual environment

```bash
python3 -m venv venv
```

### Step 2 — Activate it

macOS / Linux:
```bash
source venv/bin/activate
```

Windows:
```bash
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Place your model

Copy your trained model file into the project root:
```
pneumoscan/pneumonia_detector_efficientnet.keras
```

### Step 5 — Run the app

```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## Production Deployment (Vultr / Any Linux VPS)

### Step 1 — SSH into your server

```bash
ssh root@<your-server-ip>
```

### Step 2 — Clone and set up

```bash
git clone <your-repo-url>
cd pneumoscan
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3 — Place the model on the server

```bash
# From your local machine (in a new terminal):
scp pneumonia_detector_efficientnet.keras root@<your-server-ip>:/root/pneumoscan/
```

### Step 4 — Run with Gunicorn (production server)

```bash
gunicorn --workers 2 --bind 0.0.0.0:80 app:app
```

App will be live at: **http://<your-server-ip>**

### Step 5 — Run as background service (optional)

```bash
nohup gunicorn --workers 2 --bind 0.0.0.0:80 app:app &
```

To stop it:
```bash
pkill gunicorn
```

---

## How It Works

1. User uploads a chest X-ray (JPG/PNG/WebP).
2. Flask saves it temporarily to `./uploads/`.
3. Image is resized to 224×224 and normalized (÷255).
4. EfficientNetB0 model predicts: `> 0.5` → Pneumonia, `≤ 0.5` → Normal.
5. Confidence percentage is shown alongside the result.
6. Uploaded file is deleted after prediction.

---

## Technologies

| Layer     | Stack                         |
|-----------|-------------------------------|
| Backend   | Flask 3.x, Python 3.x        |
| Model     | TensorFlow 2.x, EfficientNetB0|
| Frontend  | HTML, CSS, Vanilla JS         |
| Production| Gunicorn, Vultr VPS          |

---

> **Disclaimer**: For educational use only. Not a substitute for clinical diagnosis by a licensed medical professional.
