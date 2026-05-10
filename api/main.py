"""
api/main.py
===========
FastAPI backend for Diamond-Lite.
Integrates Vision-Transformer, Fusion Classifier, XAI, and LLM Reporting.

Endpoints:
    POST /predict  - Accepts MRI image and patient data
"""

import os
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.explain import explain
from src.report import generate_clinical_report

app = FastAPI(title="Diamond-Lite Alzheimer API")

# Allow CORS for Flask frontend (port 5000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for demo purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs for static access to heatmaps
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ── Model Config ─────────────────────────────────────────────────────────────

# Automatically find the best checkpoint in outputs/
def get_best_checkpoint():
    ckpts = list(OUTPUTS_DIR.glob("best_*.pt"))
    if not ckpts:
        return None
    # Sort by creation time (newest first)
    return str(sorted(ckpts, key=lambda x: x.stat().st_mtime, reverse=True)[0])

CHECKPOINT = get_best_checkpoint()

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": CHECKPOINT is not None}

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    age: Optional[float] = Form(None),
    mmse: Optional[float] = Form(None),
    cdr: Optional[float] = Form(None),
    education_years: Optional[float] = Form(None),
    apoe4: Optional[int] = Form(None),
):
    use_metadata = all(v is not None for v in [age, mmse, cdr, education_years, apoe4])
    if not CHECKPOINT:
        return {"error": "No model checkpoint found. Please train the model first."}

    # 1. Save uploaded image temporarily
    temp_dir = OUTPUTS_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    image_path = temp_dir / image.filename
    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())

    # 2. Run Inference & XAI
    if use_metadata:
        # Normalize ALL 5 features using the StandardScaler values from the training distribution
        # Means: [70.215, 22.034, 0.808, 13.325, 0.415]
        # Stds:  [8.319, 6.658, 0.699, 3.080, 0.493]
        tabular_data = [
            (age - 70.215) / 8.319,
            (mmse - 22.034) / 6.658,
            (cdr - 0.808) / 0.699,
            (education_years - 13.325) / 3.080,
            (apoe4 - 0.415) / 0.493,
        ]
    else:
        # No metadata — pass zeros (training mean), model relies solely on MRI
        tabular_data = [0.0, 0.0, 0.0, 0.0, 0.0]

    result = explain(
        image_path=str(image_path),
        tabular_data=tabular_data,
        checkpoint_path=CHECKPOINT
    )

    # 3. Generate Clinical Report
    patient_data = {
        "age": age if use_metadata else None,
        "mmse": mmse if use_metadata else None,
        "cdr": cdr if use_metadata else None,
        "education_years": education_years if use_metadata else None,
        "apoe4": apoe4 if use_metadata else None,
        "metadata_used": use_metadata,
    }
    
    report = generate_clinical_report(
        prediction=result["prediction"],
        confidence=result["confidence"],
        patient_data=patient_data
    )

    # 4. Cleanup/Final Response
    return {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "heatmap_url": f"/outputs/xai/{Path(result['heatmap_path']).name}",
        "report": report,
        "patient_summary": patient_data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
