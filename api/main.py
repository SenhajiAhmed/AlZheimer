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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.explain import explain
from src.report import generate_clinical_report

app = FastAPI(title="Diamond-Lite Alzheimer API")

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
    age: float = Form(...),
    mmse: float = Form(...),
    cdr: float = Form(...),
    education_years: float = Form(...),
    apoe4: int = Form(...),
):
    if not CHECKPOINT:
        return {"error": "No model checkpoint found. Please train the model first."}

    # 1. Save uploaded image temporarily
    temp_dir = OUTPUTS_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    image_path = temp_dir / image.filename
    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())

    # 2. Run Inference & XAI
    # We use raw values here; the explain() function handles normalization
    result = explain(
        image_path=str(image_path),
        tabular_data=[age, mmse, cdr, education_years, apoe4],
        checkpoint_path=CHECKPOINT
    )

    # 3. Generate Clinical Report
    patient_data = {
        "age": age,
        "mmse": mmse,
        "cdr": cdr,
        "education_years": education_years,
        "apoe4": apoe4
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
