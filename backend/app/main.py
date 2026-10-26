
from pathlib import Path
import io
import json

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------
# Paths and device
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
WEIGHTS_PATH = MODEL_DIR / "asd_resnet18.pth"
CLASSES_PATH = MODEL_DIR / "classes.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="ASD Insight Research API",
    description="Experimental eye-image classifier. Not for diagnosis.",
    version="1.0.0",
)

# Add your actual Vercel production domain here.
# Add any other Vercel deployment domains you use as needed.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://asd-insight-eye-image-research.vercel.app",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://asd-insight-eye-image-research.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Load model once when the API starts
# --------------------------------------------------

model = None
idx_to_class = None
model_error = None


def load_classifier():
    """Load trained ResNet18 weights and class labels."""
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {WEIGHTS_PATH}"
        )

    if not CLASSES_PATH.is_file():
        raise FileNotFoundError(
            f"Class labels not found: {CLASSES_PATH}"
        )

    with CLASSES_PATH.open("r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError("classes.json must contain a non-empty JSON object.")

    idx_to_class = {
        int(index): name
        for name, index in class_to_idx.items()
    }

    network = models.resnet18(weights=None)
    network.fc = nn.Linear(
        network.fc.in_features,
        len(idx_to_class),
    )

    state = torch.load(
        WEIGHTS_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    network.load_state_dict(state)
    network.to(DEVICE)
    network.eval()

    return network, idx_to_class


@app.on_event("startup")
def initialize_model():
    """Try loading the model at startup and retain any error."""
    global model, idx_to_class, model_error

    try:
        model, idx_to_class = load_classifier()
        model_error = None
        print("ASD Insight model loaded successfully.")
    except Exception as exc:
        model = None
        idx_to_class = None
        model_error = f"{type(exc).__name__}: {exc}"
        print(f"ASD Insight model failed to load: {model_error}")


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "ASD Insight research API",
        "status": "running",
        "diagnostic_use": False,
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_ready": model is not None,
    }


@app.get("/api/model-status")
def model_status():
    """Report model availability without exposing server paths."""
    return {
        "model_ready": model is not None,
        "message": (
            "Model loaded and ready."
            if model is not None
            else "Model unavailable. Check the Render application logs."
        ),
    }


# --------------------------------------------------
# Image analysis
# --------------------------------------------------

@app.post("/api/analyze")
async def analyze_eye_image(file: UploadFile = File(...)):
    if model is None or idx_to_class is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The research model is unavailable. "
                "Check that backend/models/asd_resnet18.pth "
                "and backend/models/classes.json exist and contain "
                "valid trained model data."
            ),
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, or WebP image.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > 8 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 8 MB.",
        )

    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()

        with Image.open(io.BytesIO(content)) as image:
            image = image.convert("RGB")

            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    try:
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)[0]

        scores = {
            idx_to_class[i]: round(
                float(probabilities[i].item()) * 100,
                2,
            )
            for i in range(len(idx_to_class))
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="The model could not process this image.",
        )

    return {
        "research_output": max(scores, key=scores.get),
        "class_scores_percent": scores,
        "is_diagnostic": False,
        "notice": (
            "Experimental research output only. This result cannot "
            "diagnose or rule out ASD. Scores are model outputs, "
            "not calibrated clinical probabilities."
        ),
    }
