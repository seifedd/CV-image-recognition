from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import cv2
import os

app = FastAPI(title="CV Image Classifier MVP", description="MVP wrapper for k-NN Image Classifier")

# Setup CORS to allow Vite frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], # Restrict origins for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
MODEL_PATH = "model/knn_model.pkl"
ENCODER_PATH = "model/label_encoder.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
else:
    model = None
    label_encoder = None
    print("[WARNING] Model not found! Please run train_and_save_model.py first.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the FullStack Image Classifier API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model or not label_encoder:
        return {"error": "Model not loaded. Please train first."}
    
    # Read the uploaded image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Failed to decode the expected image."}

    # Preprocess the image (re-using logic from SimplePreprocessor)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

    # Flatten the image into a 3072 feature vector
    data = img.flatten().reshape(1, -1)

    # Perform prediction
    prediction = model.predict(data)
    
    # Decode label
    label = label_encoder.inverse_transform(prediction)[0]
    
    return {"prediction": label, "filename": file.filename}
