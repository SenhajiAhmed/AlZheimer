# 🧠 Diamond-Lite: Multimodal Alzheimer's Diagnosis Assistant

**Diamond-Lite** is a medical AI proof-of-concept that combines **Computer Vision (ViT)** with **Clinical Metadata** to assist neurologists in diagnosing Alzheimer's Disease stages.

![Streamlit Demo Placeholder](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)
![Tech Stack](https://img.shields.io/badge/Tech-ViT%20|%20FastAPI%20|%20Streamlit%20|%20Llama3-blue?style=for-the-badge)

---

## 🚀 Key Features

- **Multimodal Fusion**: Integrates 2D MRI scans with clinical data (Age, MMSE, CDR, APOE4 genotype) using a Late-Fusion architecture.
- **Vision Transformer (ViT)**: Uses a `vit-base-patch16-224` backbone for high-resolution feature extraction.
- **Explainable AI (XAI)**: Implements **Attention Rollout** to visualize exactly which brain regions influenced the model's decision.
- **Generative Clinical Reports**: Uses LLMs (Llama 3/GPT-4) to translate AI findings into structured, professional medical notes.
- **Full-Stack Implementation**: Powered by a FastAPI backend and a Streamlit medical dashboard.

---

## 📊 Model Performance

Trained on the Kaggle 4-class Alzheimer Dataset with synthetic clinical metadata:

| Metric | Score |
| :--- | :--- |
| **Macro F1-Score** | **94%** |
| **Macro AUC-ROC** | **99.4%** |
| **Accuracy** | **95%** |

*Note: High performance is attributed to the inclusion of correlated clinical metadata. See Protocol for details on synthetic data limitations.*

---

## 🛠 Project Structure

```text
AlZheimer/
├── api/                # FastAPI backend server
├── ui/                 # Streamlit dashboard
├── src/                # Core logic
│   ├── models/         # ViT, MLP, and Fusion architectures
│   ├── data_loader.py  # Multimodal Dataset pipeline
│   ├── explain.py      # Attention Rollout visualization
│   └── report.py       # LLM Generative Reporting
├── data/               # Scripts for Kaggle download & data generation
└── outputs/            # Trained weights and XAI results
```

---

## 🚦 Getting Started

### 1. Prerequisites
- Python 3.10+
- A Groq or OpenAI API Key (for LLM reports)

### 2. Setup
```bash
git clone https://github.com/SenhajiAhmed/AlZheimer.git
cd AlZheimer
pip install -r requirements.txt
cp .env.example .env  # Add your API keys here
```

### 3. Run the Application
Start the backend:
```bash
uvicorn api.main:app --reload
```

In a new terminal, start the dashboard:
```bash
streamlit run ui/app.py
```

---

## 🔬 Scientific Context
This project demonstrates the power of **Late Fusion** in medical imaging. By concatenating feature embeddings from a frozen Vision Transformer (Image Branch) and a 3-layer MLP (Tabular Branch), the model achieves a more holistic patient assessment than vision-only approaches.

**Disclaimer**: *This software is a research prototype and is NOT intended for clinical use. All diagnoses must be verified by a qualified medical professional.*

---
*Created by Ahmed Senhaji for the Diamond-Lite Hackathon Sprint.*
