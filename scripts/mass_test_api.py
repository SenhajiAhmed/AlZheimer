import os
import time
import random
import requests
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report

API_URL = "http://localhost:8000/predict"

def run_mass_test(n_samples=20):
    print(f"🚀 Starting massive API test with {n_samples} samples...")
    print("This will send real HTTP requests to your FastAPI backend to verify the multimodal fusion is working end-to-end.\n")
    
    # Load synthetic tabular data
    csv_path = Path("data/synthetic_tabular.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    class_names = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
    
    grouped = df.groupby("class_label")
    y_true = []
    y_pred = []
    
    success_count = 0
    
    for i in range(n_samples):
        # Pick a random class ensuring balanced testing if possible
        true_class = class_names[i % 4]
        
        # Get a random clinical profile for this class
        try:
            row = grouped.get_group(true_class).sample(1).iloc[0]
        except KeyError:
            continue
            
        # Find a random image for this class
        img_dir = Path(f"data/raw/Alzheimer_s Dataset/test/{true_class}")
        if not img_dir.exists():
            print(f"Warning: Folder {img_dir} missing.")
            continue
            
        images = list(img_dir.glob("*.jpg"))
        if not images:
            continue
            
        img_path = random.choice(images)
        
        # Prepare multipart form data payload matching the API spec
        data = {
            "age": row["age"],
            "mmse": row["mmse"],
            "cdr": row["cdr"],
            "education_years": row["education_years"],
            "apoe4": row["apoe4"]
        }
        
        try:
            with open(img_path, "rb") as f:
                files = {"image": (img_path.name, f, "image/jpeg")}
                
                start_t = time.time()
                response = requests.post(API_URL, data=data, files=files, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                pred_class = result["prediction"]
                conf = result["confidence"]
                
                y_true.append(true_class)
                y_pred.append(pred_class)
                
                # Check if correct (note: with a dummy image, accuracy will be mostly meaningless, but we test the API!)
                mark = "✅" if pred_class == true_class else "❌"
                
                print(f"[{i+1:02d}/{n_samples}] {mark} True: {true_class[:12]:12s} | Pred: {pred_class[:12]:12s} | Conf: {conf:6.1%} | {time.time()-start_t:.1f}s")
                success_count += 1
                
        except requests.exceptions.RequestException as e:
            print(f"[{i+1:02d}/{n_samples}] ⚠️ API Error: {e}")
            
        # Small delay to prevent rate-limiting from Groq API
        time.sleep(1.5)
        
    print(f"\n✅ Tested {success_count} samples successfully.")
    
    if len(y_true) > 0:
        print("\n📊 --- API Classification Report ---")
        try:
            print(classification_report(y_true, y_pred, labels=class_names, target_names=class_names, zero_division=0))
        except ValueError as e:
            print(f"Could not generate report: {e}")

if __name__ == "__main__":
    # We test 20 samples by default to avoid hitting Groq's Free Tier Rate Limits too fast.
    run_mass_test(20)
