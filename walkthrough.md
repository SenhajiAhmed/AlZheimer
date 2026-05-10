# 🧠 Diamond-Lite — Alzheimer Multimodal AI: Walkthrough

> **Project root:** `/home/senhajiahmed/Desktop/Projects/AlZheimer/`
> **Goal:** Multimodal Alzheimer classification (MRI + Clinical data) with XAI & LLM reporting.
> **Timeline target:** 48h sprint.

---

## Phase T-0 — Environment & Data Preparation

- [x] **Step 1 — Project structure initialized** (folders + placeholder files created)
- [x] **Step 2 — Environment configured** (`requirements.txt` created, venv instructions documented)
- [x] **Step 3 — Kaggle data download script ready** (`data/download_data.sh` created with instructions)
- [x] **Step 4 — Synthetic tabular data generator written** (`data/generate_synthetic.py` created and executable)
- [x] **Step 5 — Git repository initialized** (`.gitignore` + initial commit `0f166b7` done)

---

## Phase 1 — Baseline Image Model

- [x] Custom `Dataset` class written (`src/data_loader.py`)
- [x] Pretrained ViT or ResNet loaded with frozen backbone (`src/models/vision.py`)
- [x] Classification head replaced with Identity for feature extraction
- [x] Training loop implemented (`src/train.py`) — AMP, early stopping, checkpointing
- [x] Baseline validation accuracy & F1 logged (outputs saved to `outputs/`)

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

---

## 🗒 Next Session — Start Here

> **Phase 1** is next: implement `src/data_loader.py` (custom `AlzheimerDataset`), then fine-tune the frozen ViT/ResNet baseline.
> Run `data/download_data.sh` first if `data/raw/` is empty.

---

*Last updated: 2026-05-10 — T-0 complete ✅*
