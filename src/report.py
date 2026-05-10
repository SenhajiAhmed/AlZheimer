"""
src/report.py
=============
Generative clinical reporting using LLMs (Groq or OpenAI).
Translates AI model outputs into structured natural language notes for physicians.

Usage:
    from src.report import generate_clinical_report
    report = generate_clinical_report(prediction_data, patient_meta)
"""

import os
from typing import Dict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

# ── LLM Configuration ────────────────────────────────────────────────────────

def get_llm():
    """Initialize the LLM based on environment variables."""
    backend = os.getenv("LLM_BACKEND", "groq").lower()
    
    if backend == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("⚠️ GROQ_API_KEY not found. LLM reporting will be disabled.")
            return None
        return ChatGroq(
            temperature=0.2,
            model_name=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            groq_api_key=api_key
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ OPENAI_API_KEY not found. LLM reporting will be disabled.")
            return None
        return ChatOpenAI(
            temperature=0.2,
            model_name=os.getenv("LLM_MODEL", "gpt-4-turbo"),
            openai_api_key=api_key
        )

# ── Report Generation ────────────────────────────────────────────────────────

REPORT_PROMPT = """
You are a senior neurologist reviewing an AI-assisted MRI analysis.

### PATIENT PROFILE
- Age: {age}
- MMSE Score: {mmse}/30 (Cognitive Test)
- CDR: {cdr} (Clinical Dementia Rating)
- APOE4 Genotype: {apoe4} (Genetic Risk)
- Education: {education_years} years

### AI ANALYSIS FINDINGS
- Predicted Diagnosis: {prediction}
- Confidence Score: {confidence:.1%}
- XAI Heatmap: {xai_status}

### TASK
Write a concise 3-paragraph preliminary clinical summary:
1. **Interpretation**: Explain the AI findings in the context of the patient's specific clinical data.
2. **Risk Assessment**: Identify the primary risk factors contributing to this diagnosis.
3. **Recommendation**: Suggest next steps for the attending physician (e.g., follow-up tests, PET scan, or specific counseling).

### GUIDELINES
- Use professional, clinical language.
- Be objective but direct.
- MANDATORY DISCLAIMER: Start the report with: "PRELIMINARY AI-ASSISTED ANALYSIS - FOR PHYSICIAN VERIFICATION ONLY."

Report:
"""

def generate_clinical_report(
    prediction: str,
    confidence: float,
    patient_data: Dict,
    has_heatmap: bool = True
) -> str:
    """
    Generate a clinical report using the configured LLM.
    
    Args:
        prediction:   The class name predicted by the model.
        confidence:   Model confidence (0.0 to 1.0).
        patient_data: Dict containing age, mmse, cdr, education_years, apoe4.
        has_heatmap:  Whether an XAI heatmap was successfully generated.
    """
    llm = get_llm()
    if not llm:
        return "LLM reporting is currently unavailable. Please check your API keys."

    prompt = ChatPromptTemplate.from_template(REPORT_PROMPT)
    chain = prompt | llm
    
    # Format apoe4 for better readability
    apoe4_str = "Carrier (High Risk)" if patient_data.get("apoe4", 0) == 1 else "Non-carrier"
    
    input_data = {
        "age": patient_data.get("age", "N/A"),
        "mmse": patient_data.get("mmse", "N/A"),
        "cdr": patient_data.get("cdr", "N/A"),
        "education_years": patient_data.get("education_years", "N/A"),
        "apoe4": apoe4_str,
        "prediction": prediction,
        "confidence": confidence,
        "xai_status": "Attention regions identified and visualized" if has_heatmap else "No visual explanation available"
    }

    try:
        response = chain.invoke(input_data)
        return response.content
    except Exception as e:
        return f"Error generating report: {str(e)}"

# ── Test Execution ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Mock data for testing
    sample_patient = {
        "age": 75,
        "mmse": 18,
        "cdr": 1.0,
        "education_years": 12,
        "apoe4": 1
    }
    
    print("✍️ Generating sample report...")
    report = generate_clinical_report(
        prediction="MildDemented",
        confidence=0.87,
        patient_data=sample_patient
    )
    print("\n" + "="*50)
    print(report)
    print("="*50)
