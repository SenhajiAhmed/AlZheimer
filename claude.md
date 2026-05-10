### 1. High-Level Architecture

The system operates as a pipeline:

1. **Input:** MRI Image + Tabular Clinical Data (Age, MMSE score, Genotype, Education).
2. **Vision Branch:** Pretrained Vision Transformer (ViT) or ResNet extracting an image feature vector.
3. **Clinical Branch:** Multi-Layer Perceptron (MLP) extracting a tabular feature vector.
4. **Fusion Center:** Concatenation followed by a lightweight classification head (Dense layers) outputting prediction logits (e.g., Normal vs. Mild Cognitive Impairment vs. Alzheimer's).
5. **Explainability:** Attention Rollout (for ViT) or Grad-CAM (for CNN) applied to the vision branch to generate heatmaps.
6. **Reporting:** Model logits, patient data, and explanations are passed to an LLM (e.g., via LangChain) to generate a localized, professional medical report.

### 2. Recommended Datasets

* **The Hackathon Cheat Code:** Avoid raw ADNI or OASIS-3 if you have less than a week. They require application approvals and massive preprocessing.
* **Go-to:** Search Kaggle for **"Alzheimer's Dataset (4 class of Images)"**. It contains pre-processed, 2D MRI slices categorized into NonDemented, VeryMildDemented, MildDemented, and ModerateDemented.
* **Tabular Mocking:** Since standard Kaggle datasets often lack tabular data, generate synthetic clinical metadata (Age, MMSE, CDR) mathematically correlated to the disease classes to prove the multimodal concept works.

### 3. DiaMond: What to Keep vs. Simplify

* **Keep:** The dual-branch architecture and the use of Vision Transformers for the image modality.
* **Simplify:** Drop the computationally heavy cross-attention mechanisms between modalities. Replace the 3D ViT with a 2D ViT (processing the middle slice or averaging a few slices). Drop PET entirely; clinical tabular data is much easier to process and highly relevant to doctors.

### 4. Implementation Phases

1. **T-0 (Prep):** Setup environment, download Kaggle data, generate synthetic tabular data, initialize Git repository.
2. **Phase 1 (Baseline Image Model):** Fine-tune a pretrained ViT/ResNet on the 2D MRI slices. Get a working baseline accuracy.
3. **Phase 2 (Multimodal Fusion):** Build the MLP for tabular data. Concatenate the features with the frozen Vision model outputs. Train the fusion head.
4. **Phase 3 (XAI & Generative Reporting):** Implement the Grad-CAM/Attention visualizer. Connect an LLM API to ingest the results.
5. **Phase 4 (UI & API):** Wrap the model in a FastAPI backend and connect a Streamlit frontend.
6. **Phase 5 (Polish):** Draft the pitch, record a demo video.

### 5. Project Structure

```text
diamond-lite/
├── data/                  # Raw and processed datasets
├── notebooks/             # EDA and experimentation
├── src/
│   ├── data_loader.py     # Custom PyTorch Dataset (Image + Tabular)
│   ├── models/            # Vision model, MLP, and Fusion network
│   ├── train.py           # Training loop and validation
│   └── explain.py         # Grad-CAM / Attention rollout logic
├── api/
│   └── main.py            # FastAPI endpoints
├── ui/
│   └── app.py             # Streamlit dashboard
├── requirements.txt
└── README.md

```

### 6. Data Preprocessing Pipeline

Keep it strictly to 2D for hackathon speed.

* **Image:** Resize to 224x224 (standard for ViT/ResNet). Apply basic augmentations (slight rotations, horizontal flips). Normalize using ImageNet mean and standard deviation.
* **Tabular:** Standardize continuous variables (Age, test scores) using `StandardScaler`. One-hot encode categorical variables.

### 7. Model Training Strategy

Do not train from scratch.

1. Load a pretrained model (e.g., `google/vit-base-patch16-224` from Hugging Face or a `ResNet50` from torchvision).
2. **Freeze** the entire backbone.
3. Replace the final classification head with an Identity layer to output the raw feature embedding (e.g., a 768-dimensional vector).
4. Only calculate gradients and update weights for your custom Fusion and Classification layers. This allows you to train on a standard GPU in minutes, not days.

### 8. Integrating Multimodal Fusion

Use late fusion (concatenation). Let $h_{img}$ be the visual embedding and $h_{tab}$ be the tabular embedding. The combined feature vector $z$ is:


$$z = \text{ReLU}(W \cdot [h_{img} ; h_{tab}] + b)$$


Feed $z$ into a final Softmax or Sigmoid classification layer.

### 9. Implementing Explainability

If you end up using a CNN (like ResNet) for speed, use the `pytorch-grad-cam` library. It’s plug-and-play. If you stick with ViT, use Attention Rollout. The goal is to overlay a heatmap on the original MRI indicating which regions (e.g., ventricles, hippocampus) drove the model's prediction. Save this overlaid image to serve to the frontend.

### 10. Evaluation

Focus on **F1-Score** and **Macro AUC-ROC**. Medical datasets are inherently imbalanced (more healthy patients than sick ones). Accuracy will mislead the judges. Plot a clean Confusion Matrix.

### 11. Backend API (Flask / FastAPI)

Given your experience with end-to-end Python deployments, FastAPI is excellent here for its asynchronous capabilities, but Flask works perfectly fine if you need to move fast.

* **Endpoint `/predict`:** Accepts a multipart form (Image file + JSON tabular data). Runs the inference, generates the XAI heatmap, triggers the LLM report generation, and returns a JSON payload with all three.

### 12. Frontend Dashboard

Use **Streamlit**. You can build a highly professional, interactive medical dashboard in under 100 lines of code. Create a sidebar for patient metadata input and a file uploader for the MRI. Dedicate the main view to displaying the original scan alongside the XAI heatmap, followed by the generated report.

### 13. Generating AI Medical Reports

This is your "Wow Factor." Take the outputs:

* Predicted Class (e.g., "Mild Cognitive Impairment")
* Confidence Score (e.g., "87%")
* Patient Data (e.g., "65 y/o, MMSE: 24")
Pass this into a prompt template using an LLM API (OpenAI, Anthropic, or an open-source model via Together AI).
* *Prompt Example:* "Act as a neurologist. Given the following AI analysis [DATA], draft a short, clinical preliminary summary. State clearly that this is an AI-assisted analysis and requires physician verification."

### 14. Deployment Strategy

* **Code & Pitch:** GitHub.
* **App:** Hugging Face Spaces (Streamlit + Docker) or Streamlit Community Cloud. It is free, requires zero infrastructure setup, and provides a public URL for the judges.

### 15. Timelines

* **24 Hours:** Skip the API. Build everything inside a monolithic Streamlit app. Use a pretrained ResNet (skip ViT), tabular fusion, and a basic LLM API call.
* **48 Hours:** Separate the FastAPI backend from the Streamlit frontend. Implement ViT with Attention Rollout. Add robust error handling.
* **1 Week:** Full architectural split. Train your own custom lightweight cross-attention layer instead of basic concatenation. Dockerize the entire stack. Draft a pristine, formal methodology document using LaTeX.

### 16. High-Impact Features for Judges

1. **The LLM Report:** Translating raw probabilities into natural language clinical notes bridges the gap between engineering and real-world utility.
2. **The XAI Visualizer:** Doctors do not trust black boxes. Visualizing the "why" is mandatory for modern healthcare AI.
3. **Speed:** A snappy, responsive UI that works live on stage.

### 17. Common Pitfalls & Debugging

* **Data Leakage:** Ensure absolutely no patient overlap between your train and test sets.
* **OOM Errors (Out of Memory):** If using ViTs, watch your batch size. Keep it small (8 or 16) if GPU RAM is limited.
* **Over-engineering:** Don't waste 10 hours trying to get a 3D CNN working. A 2D slice approach proves the concept just as well for a hackathon.

### 18. Compute Recommendations

Google Colab Pro (T4 or A100) is sufficient for training the lightweight fusion layers. For the demo, inference can run smoothly on a standard CPU if the backbone is frozen and weights are optimized.

### 19. Suggested Tech Stack

* **Modeling:** PyTorch, Hugging Face `transformers`, `torchvision`.
* **Explainability:** `pytorch-grad-cam`.
* **Backend/Frontend:** FastAPI, flask.
* **Generative Text:** LangChain + OpenAI API (or Groq for lightning-fast open-source LLM inference).
* **Data Manipulation:** Pandas, Scikit-learn.

### 20. The Pitch

Do not start by talking about Vision Transformers. Start with the *problem*: "Neurologists are overwhelmed, and early signs of dementia are easily missed. We built an end-to-end multimodal assistant..."
Show the live UI uploading an image and generating the report. Keep the technical architecture slide simple, focusing on the *fusion* aspect. Consider writing a sharp, one-page technical executive summary compiled in LaTeX to hand to the technical judges.

