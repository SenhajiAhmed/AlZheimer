"""
ui/app.py
=========
Streamlit Dashboard for Diamond-Lite.
Provides a clean, medical-grade interface for Alzheimer's diagnosis.
"""

import streamlit as st
import requests
from PIL import Image
import time

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Diamond-Lite | Alzheimer AI",
    page_icon="🧠",
    layout="wide"
)

API_URL = "http://localhost:8000"

# ── Styling ──────────────────────────────────────────────────────────────────

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .report-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ── Sidebar: Patient Metadata ────────────────────────────────────────────────

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2491/2491314.png", width=100)
    st.title("Patient Metadata")
    st.info("Enter clinical test scores and demographics below.")
    
    age = st.slider("Age", 50, 95, 65)
    mmse = st.slider("MMSE Score (0-30)", 0, 30, 24)
    cdr = st.select_slider("Clinical Dementia Rating (CDR)", options=[0.0, 0.5, 1.0, 2.0], value=0.0)
    edu = st.number_input("Years of Education", 1, 25, 12)
    apoe4 = st.selectbox("APOE4 Allele", options=[0, 1], format_func=lambda x: "Carrier (High Risk)" if x == 1 else "Non-carrier")
    
    st.divider()
    st.caption("v1.0.0-lite | Developed for Hackathon Demo")

# ── Main Content ─────────────────────────────────────────────────────────────

st.title("🧠 Diamond-Lite: Multimodal Alzheimer's Assistant")
st.write("Diagnostic support system combining MRI analysis with clinical context.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload MRI Scan")
    uploaded_file = st.file_uploader("Select a 2D MRI slice (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Original MRI Scan", use_column_width=True)

with col2:
    st.subheader("2. AI Analysis")
    if uploaded_file is not None:
        if st.button("Run Diagnostic Analysis"):
            with st.spinner("Analyzing scan and clinical data..."):
                # Prepare request to FastAPI
                files = {"image": uploaded_file.getvalue()}
                data = {
                    "age": age,
                    "mmse": mmse,
                    "cdr": cdr,
                    "education_years": edu,
                    "apoe4": apoe4
                }
                
                try:
                    # Trigger backend
                    response = requests.post(f"{API_URL}/predict", files=files, data=data)
                    
                    if response.status_code == 200:
                        res = response.json()
                        
                        # Display Results
                        st.success("Analysis Complete!")
                        
                        m1, m2 = st.columns(2)
                        m1.metric("Diagnosis", res["prediction"])
                        m2.metric("Confidence", f"{res['confidence']:.1%}")
                        
                        # Show Heatmap
                        st.divider()
                        st.write("**Explainable AI: Region of Interest (Attention Heatmap)**")
                        heatmap_url = f"{API_URL}{res['heatmap_url']}"
                        st.image(heatmap_url, caption="Regions of focus for the prediction", use_column_width=True)
                        
                        # Store report for display below
                        st.session_state["report"] = res["report"]
                        
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
    else:
        st.warning("Please upload an MRI image to start the analysis.")

# ── Clinical Report Section ──────────────────────────────────────────────────

if "report" in st.session_state:
    st.divider()
    st.subheader("📄 AI-Generated Clinical Preliminary Report")
    st.markdown(f'<div class="report-box">{st.session_state["report"].replace("\n", "<br>")}</div>', unsafe_allow_html=True)
    
    st.download_button(
        label="Download Report as Text",
        data=st.session_state["report"],
        file_name=f"report_patient_age_{age}.txt",
        mime="text/plain"
    )
