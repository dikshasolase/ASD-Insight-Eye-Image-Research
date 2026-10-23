from pathlib import Path
import io
import json

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="ASD Insight Research API",
    description="Experimental eye-image classifier. Not for diagnosis.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "ASD Insight research API", "status": "running", "diagnostic_use": False}

@app.get("/api/health")
def health():
    return {"status": "healthy", "model_ready": (MODEL_DIR / "asd_resnet18.pth").exists()}

def load_classifier():
    weights_path = MODEL_DIR / "asd_resnet18.pth"
    classes_path = MODEL_DIR / "classes.json"
    if not weights_path.exists() or not classes_path.exists():
        return None, None
    with classes_path.open(encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(index): name for name, index in class_to_idx.items()}
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(idx_to_class))
    state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, idx_to_class

@app.post("/api/analyze")
async def analyze_eye_image(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WebP image.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be smaller than 8 MB.")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            image = image.convert("RGB")
            tensor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])(image).unsqueeze(0).to(DEVICE)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    model, idx_to_class = load_classifier()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not trained. Add your dataset and run: python train_model.py",
        )
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    scores = {
        idx_to_class[i]: round(float(probabilities[i]) * 100, 2)
        for i in range(len(idx_to_class))
    }
    return {
        "research_output": max(scores, key=scores.get),
        "class_scores_percent": scores,
        "is_diagnostic": False,
        "notice": (
            "Experimental research output only. This result cannot diagnose or rule out ASD. "
            "Scores are model outputs, not calibrated clinical probabilities."
        ),
    }
