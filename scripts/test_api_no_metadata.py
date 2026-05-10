import os
import time
import random
import requests
from pathlib import Path
from sklearn.metrics import classification_report

API_URL = "http://localhost:8000/predict"

def run_vision_only_test(n_samples=20):
    print(f"👁️ Starting VISION-ONLY API test with {n_samples} samples...")
    print("Clinical metadata toggle is OFF. The model will rely purely on MRI features.\n")
    
    class_names = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
    base_path = Path("data/raw/Alzheimer_s Dataset/test")
    
    if not base_path.exists():
        print(f"Error: {base_path} not found.")
        return
        
    y_true = []
    y_pred = []
    
    success_count = 0
    
    for i in range(n_samples):
        # Balanced sampling
        true_class = class_names[i % 4]
        img_dir = base_path / true_class
        
        images = list(img_dir.glob("*.jpg"))
        if not images:
            continue
            
        img_path = random.choice(images)
        
        try:
            with open(img_path, "rb") as f:
                # OMITTING all form data fields (age, mmse, etc.)
                # This triggers the "MRI-only" logic in api/main.py
                files = {"image": (img_path.name, f, "image/jpeg")}
                
                start_t = time.time()
                response = requests.post(API_URL, files=files, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                pred_class = result["prediction"]
                conf = result["confidence"]
                
                y_true.append(true_class)
                y_pred.append(pred_class)
                
                mark = "✅" if pred_class == true_class else "❌"
                print(f"[{i+1:02d}/{n_samples}] {mark} True: {true_class[:12]:12s} | Pred: {pred_class[:12]:12s} | Conf: {conf:6.1%} | {time.time()-start_t:.1f}s")
                success_count += 1
                
        except requests.exceptions.RequestException as e:
            print(f"[{i+1:02d}/{n_samples}] ⚠️ API Error: {e}")
            
        time.sleep(1.5)
        
    print(f"\n✅ Tested {success_count} samples in Vision-Only mode.")
    
    if len(y_true) > 0:
        print("\n📊 --- VISION-ONLY Performance ---")
        print(classification_report(y_true, y_pred, labels=class_names, target_names=class_names, zero_division=0))

if __name__ == "__main__":
    run_vision_only_test(20)
