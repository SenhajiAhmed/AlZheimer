# 🧠 Diamond-Lite — Alzheimer Multimodal AI: Walkthrough

> **Project root:** `/home/senhajiahmed/Desktop/Projects/AlZheimer/`
> **Goal:** Multimodal Alzheimer classification (MRI + Clinical data) with XAI & LLM reporting.
> **Timeline target:** 48h sprint.

---

## Phase T-0 — Environment & Data Preparation

- [ ] **Step 1 — Project structure initialized** (folders + placeholder files created)
- [ ] **Step 2 — Environment configured** (`requirements.txt` created, venv instructions documented)
- [ ] **Step 3 — Kaggle data download script ready** (`data/download_data.sh` created with instructions)
- [ ] **Step 4 — Synthetic tabular data generator written** (`data/generate_synthetic.py` created and executable)
- [ ] **Step 5 — Git repository initialized** (`.gitignore` + initial commit done)

---

## Phase 1 — Baseline Image Model

- [ ] Custom `Dataset` class written (`src/data_loader.py`)
- [ ] Pretrained ViT or ResNet loaded with frozen backbone
- [ ] Classification head replaced with Identity for feature extraction
- [ ] Training loop implemented (`src/train.py`)
- [ ] Baseline validation accuracy & F1 logged

---

## Phase 2 — Multimodal Fusion

- [ ] MLP for tabular data built (`src/models/mlp.py`)
- [ ] Late-fusion concatenation head built (`src/models/fusion.py`)
- [ ] Combined training loop working end-to-end
- [ ] F1-Score & Macro AUC-ROC evaluated on test set
- [ ] Confusion matrix plotted and saved

---

## Phase 3 — XAI & Generative Reporting

- [ ] Grad-CAM (ResNet) or Attention Rollout (ViT) implemented (`src/explain.py`)
- [ ] Heatmap overlay saved as image artifact
- [ ] LLM prompt template written
- [ ] LLM API integration working (returns structured clinical note)

---

## Phase 4 — UI & API

- [ ] FastAPI `/predict` endpoint built (`api/main.py`)
- [ ] Streamlit dashboard built (`ui/app.py`)
- [ ] MRI upload → prediction → heatmap → report flow working end-to-end
- [ ] Local demo tested and passing

---

## Phase 5 — Polish

- [ ] README.md finalized
- [ ] Deployed to Hugging Face Spaces or Streamlit Cloud
- [ ] Pitch narrative drafted
- [ ] Technical executive summary written (LaTeX optional)

---

*Last updated: 2026-05-10*
