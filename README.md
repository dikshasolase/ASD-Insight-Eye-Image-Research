# ASD Insight — Eye-Image Research Prototype

Experimental eye-region image classification with React + Vite + Tailwind CSS and Python FastAPI.

**Important:** This is a research prototype, not a medical device. Eye-region images alone are not a validated standalone ASD diagnostic test. Never use model outputs for diagnosis or care decisions.

## Structure
- `Dataset/ASD/` — put labeled images here (not included)
- `Dataset/non-ASD/` — put labeled images here (not included)
- `backend/app/main.py` — FastAPI endpoints
- `backend/train_model.py` — ResNet18 training baseline
- `frontend/` — React upload and results UI

## Setup (Windows PowerShell)

### 1. Add your dataset
At project root create:
- `Dataset/ASD/`
- `Dataset/non-ASD/`

Put the images into the correct folders. The dataset is not bundled in this ZIP.

### 2. Backend
Open terminal in `backend` and run:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
python -m uvicorn app.main:app --reload
```
The first training run downloads pretrained ResNet18 weights. After training, API docs are at http://127.0.0.1:8000/docs.

Next time, from `backend`:
```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload
```

### 3. Frontend (second terminal)
```powershell
cd frontend
npm install
npm run dev
```
Open the URL Vite prints, usually http://localhost:5173.

## Notes
- The model is a baseline and needs careful evaluation.
- A random image split can leak participant-specific patterns. For research, split by participant and report sensitivity, specificity, precision, recall, F1 and confusion matrix.
- Upload checks verify file type/integrity/size, but do not confirm the image actually contains an eye.
- Class scores are model outputs, not calibrated clinical probabilities.
