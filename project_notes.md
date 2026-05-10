# 📒 Diamond-Lite — Project Technical Notes

> Living document. Update as decisions are made. Never delete entries — append with dates.

---

## Architecture Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Vision backbone | `google/vit-base-patch16-224` (Hugging Face) OR `ResNet50` (torchvision) | ViT for best accuracy; ResNet for speed. Choose based on available GPU RAM. |
| Tabular model | 3-layer MLP (input → 128 → 64 → embedding_dim) | Lightweight, fast to train |
| Fusion strategy | **Late fusion** — concatenate `h_img` and `h_tab`, pass through Dense head | Simpler than cross-attention; sufficient for hackathon |
| Classification head | Linear(fused_dim, 4) + Softmax | 4 classes: NonDemented, VeryMild, Mild, Moderate |
| XAI | `pytorch-grad-cam` if ResNet; Attention Rollout if ViT | Plug-and-play for ResNet |
| LLM backend | Groq API (fast, free tier) or OpenAI | Groq preferred for demo speed |
| Backend | FastAPI (async, JSON-native) | Best for ML serving |
| Frontend | Streamlit | Fastest path to professional medical UI |

---

## Dataset

- **Source:** Kaggle — *"Alzheimer's Dataset (4 class of Images)"*
  - `NonDemented` / `VeryMildDemented` / `MildDemented` / `ModerateDemented`
  - Pre-processed 2D MRI slices (no raw DICOM needed)
- **Split strategy:** 80/10/10 (train/val/test), stratified by class
- **Synthetic tabular columns:**
  - `age` — sampled from class-appropriate Gaussian (e.g., NonDemented: μ=62, ModerateDemented: μ=75)
  - `mmse` — Mini-Mental State Examination score (0–30); correlated inversely with severity
  - `cdr` — Clinical Dementia Rating (0, 0.5, 1, 2, 3)
  - `education_years` — random int 8–20, slight negative correlation with risk
  - `apoe4` — binary (0/1), higher in demented classes

---

## Preprocessing

### Images
- Resize: `224×224`
- Augmentations (train only): `RandomHorizontalFlip`, `RandomRotation(10°)`
- Normalize: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` (ImageNet stats)
- Convert grayscale MRI → 3-channel by repeating channel

### Tabular
- Continuous (`age`, `mmse`, `education_years`): `StandardScaler`
- Categorical (`apoe4`, `cdr` if treated as category): One-hot encode
- Missing values: None expected (synthetic data); add `fillna` guard anyway

---

## Model Dimensions

```
ViT feature vector:    768-dim
ResNet50 feature vector: 2048-dim
Tabular MLP output:    64-dim

Fused vector (ViT path):     768 + 64 = 832-dim
Fused vector (ResNet path): 2048 + 64 = 2112-dim

Classification head:
  Linear(832/2112 → 256) → ReLU → Dropout(0.3) → Linear(256 → 4)
```

---

## Training Config

| Param | Value |
|-------|-------|
| Batch size | 16 (GPU) / 8 (CPU fallback) |
| Optimizer | AdamW |
| LR (fusion head only) | 1e-3 |
| LR (backbone, if unfreezing) | 1e-5 |
| Scheduler | CosineAnnealingLR |
| Epochs | 20 (early stopping patience=5) |
| Loss | CrossEntropyLoss with class weights (handle imbalance) |
| Metrics | F1-macro, AUC-ROC macro, Confusion Matrix |

---

## LLM Report Prompt Template

```
System: You are a senior neurologist reviewing an AI-assisted MRI analysis.

User:
Patient data:
- Age: {age}
- MMSE Score: {mmse}/30
- CDR: {cdr}
- APOE4 allele: {apoe4}
- Education: {education_years} years

AI model prediction: {predicted_class} (confidence: {confidence:.1%})

Task: Write a concise 3-paragraph preliminary clinical summary:
1. Interpretation of AI findings in clinical context
2. Key risk factors from patient profile
3. Recommended next steps for the attending physician

IMPORTANT: Clearly state this is an AI-assisted analysis requiring physician verification.
```

---

## Key File Locations

| File | Purpose |
|------|---------|
| `data/raw/` | Downloaded Kaggle MRI images (unzipped) |
| `data/synthetic_tabular.csv` | Generated clinical metadata |
| `data/processed/` | Train/val/test split manifests |
| `src/data_loader.py` | PyTorch `AlzheimerDataset` class |
| `src/models/vision.py` | ViT/ResNet wrapper with frozen backbone |
| `src/models/mlp.py` | Tabular MLP encoder |
| `src/models/fusion.py` | Late-fusion classification head |
| `src/train.py` | Training & evaluation loop |
| `src/explain.py` | Grad-CAM / Attention Rollout |
| `api/main.py` | FastAPI inference endpoint |
| `ui/app.py` | Streamlit dashboard |
| `notebooks/01_eda.ipynb` | Exploratory data analysis |
| `notebooks/02_baseline.ipynb` | Baseline model experiments |

---

## Environment

- Python: 3.10+
- CUDA: Optional (CPU fallback supported)
- Virtual env: `python -m venv .venv && source .venv/bin/activate`
- Install: `pip install -r requirements.txt`
- Kaggle API key: Place `kaggle.json` at `~/.kaggle/kaggle.json`

---

## Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Class imbalance | Use `compute_class_weight` from sklearn; weighted CrossEntropy |
| Data leakage | Stratified split done once; never reshuffle after split |
| OOM on ViT + large batch | Keep batch ≤ 16; gradient accumulation if needed |
| LLM latency in demo | Pre-cache a few demo reports; show spinner in UI |
| Grayscale MRI ↔ 3-channel mismatch | Replicate channel in transforms pipeline |

---

*Last updated: 2026-05-10*
