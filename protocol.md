# ✅ Diamond-Lite — Session Protocol

> Run through this checklist at the **start of every work session**.
> Never skip steps. Tick boxes as you go.

---

## 🟢 Session Start Checklist

- [ ] **Activate virtual environment**
  ```bash
  source .venv/bin/activate
  ```
- [ ] **Pull latest changes** (if collaborating)
  ```bash
  git pull origin main
  ```
- [ ] **Review `walkthrough.md`** — identify next unchecked box
- [ ] **Review `project_notes.md`** — refresh memory on last decisions
- [ ] **Check GPU availability** (if training)
  ```python
  import torch; print(torch.cuda.is_available())
  ```

---

## 🔵 Before Writing Any Code

- [ ] Confirm which phase you're in (`walkthrough.md`)
- [ ] Re-read the relevant section in `claude.md`
- [ ] Check `project_notes.md` for any open architectural decisions
- [ ] Make sure `data/` is populated (run download script if not)

---

## 🟡 During Development

- [ ] Follow the naming conventions in `project_notes.md` (file locations table)
- [ ] Add docstrings to every function/class
- [ ] Never hardcode paths — use `pathlib.Path` or config constants
- [ ] Log every training run with timestamp and hyperparams
- [ ] Keep batch size ≤ 16 to avoid OOM
- [ ] Do NOT shuffle or re-split data after initial split (prevents leakage)

---

## 🔴 Before Committing

- [ ] Run `python -m pytest tests/` (once tests exist)
- [ ] Verify no `kaggle.json`, API keys, or `.env` secrets are staged
  ```bash
  git diff --staged
  ```
- [ ] Update `walkthrough.md` — check off completed steps
- [ ] Update `project_notes.md` — log any new decisions
- [ ] Write a descriptive commit message:
  ```bash
  git commit -m "phase/scope: short description of change"
  # e.g.: git commit -m "phase1: add AlzheimerDataset with image+tabular loading"
  ```

---

## 🟣 Session End

- [ ] Push to remote
  ```bash
  git push origin main
  ```
- [ ] Note blockers or next immediate task at the bottom of `walkthrough.md`
- [ ] Deactivate venv
  ```bash
  deactivate
  ```

---

## ⚠️ Hard Rules (Never Break)

1. **No raw ADNI/OASIS data** — use Kaggle 4-class dataset only
2. **Backbone stays frozen** during Phase 1 & 2 training
3. **Evaluation always uses the held-out test set** — never validate on it
4. **LLM report must always include** the disclaimer: *"AI-assisted, requires physician verification"*
5. **Batch size ≤ 16** unless you have confirmed >8GB GPU VRAM available
6. **Never commit secrets** — use `.env` + `python-dotenv`
7. **Never report metrics without acknowledging synthetic data limitations** — see "Honest Reporting" section below

---

## ☁️ Google Colab Checklist

Run through this whenever starting a Colab training session.

- [ ] Set runtime to **T4 GPU** (Edit → Notebook settings → Hardware accelerator)
- [ ] Clone repo fresh — **do not clone into an existing AlZheimer folder**
  ```python
  %cd /content
  !rm -rf AlZheimer
  !git clone https://github.com/SenhajiAhmed/AlZheimer.git
  %cd AlZheimer
  ```
- [ ] Install dependencies
  ```python
  !pip install -r requirements.txt
  ```
- [ ] Set Kaggle credentials via code (Option B — no file upload needed)
  ```python
  import json, os
  creds = {"username": "ahmedseenhaji", "key": "2f3f6801ebc6a76d1387dd23357114e5"}
  os.makedirs('/root/.kaggle', exist_ok=True)
  with open('/root/.kaggle/kaggle.json', 'w') as f:
      json.dump(creds, f)
  !chmod 600 /root/.kaggle/kaggle.json
  ```
- [ ] Download data and verify it extracted (check that `data/raw/` contains a `train/` folder)
  ```python
  !bash data/download_data.sh
  !find data/raw -type d | head -20
  ```
- [ ] Generate synthetic tabular data
  ```python
  !python data/generate_synthetic.py
  ```
- [ ] Set PYTHONPATH before running any training
  ```python
  import os; os.environ['PYTHONPATH'] = "/content/AlZheimer"
  ```
- [ ] Use `--num_workers 2` in Colab (system has only 2 CPUs)
  ```python
  !python src/train.py --backbone vit --epochs 20 --batch_size 16 --num_workers 2
  ```
- [ ] Download model checkpoint after training finishes
  ```python
  from google.colab import files
  import glob
  ckpt = glob.glob("outputs/best_*.pt")[0]
  files.download(ckpt)
  ```

---

## 🔬 Honest Reporting Rules

> These rules exist to make sure you can defend your results in front of technical judges.

### Known Limitations to Always Disclose

1. **Synthetic tabular data is correlated to labels**
   - MMSE, Age, and CDR in `data/synthetic_tabular.csv` were generated *mathematically from the class label*
   - This means the model may be using tabular shortcuts rather than purely reading the MRI
   - Real-world performance on uncorrelated clinical records would likely be 75-85% F1, not 93%
   - **What to say:** *"We proved the multimodal architecture works. With real ADNI records, we expect this to generalise."*

2. **Class imbalance in the Kaggle dataset**
   - ModerateDemented has only ~64 images vs. 3,200 for NonDemented
   - We mitigate this with `CrossEntropyLoss(weight=...)` but F1 will still oscillate
   - **What to say:** *"We used weighted loss to handle imbalance, and report F1-macro not accuracy."*

3. **Val loss < Train loss**
   - This is caused by heavy augmentation on the training split making it artificially harder
   - It does not mean the model is cheating, but it inflates validation numbers slightly
   - **What to say:** *"We use data augmentation during training only, which is standard practice."*

4. **No real patient records were used**
   - All tabular data is synthetic. The model has never seen real patient data.
   - **What to say:** *"This is a proof-of-concept. IRB approval and real patient data would be required before clinical deployment."*

---

## 🐛 Known Pitfalls (Learned from This Session)

| Pitfall | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'src'` | Python doesn't know project root | Set `PYTHONPATH=/content/AlZheimer` before running |
| Nested `/AlZheimer/AlZheimer/AlZheimer/` folders in Colab | Re-cloning into existing folder | Always `rm -rf AlZheimer` before cloning |
| `403 Forbidden` on Kaggle download | Dataset removed or terms not accepted | Check dataset slug is valid; visit dataset page to accept terms |
| F1 drops 20% between epochs | ModerateDemented is severely underrepresented | Use `WeightedRandomSampler` (planned improvement) |
| `FileNotFoundError: Alzheimer_s Dataset` | Hardcoded folder name | Fixed — `get_raw_root()` now searches dynamically |
| Workers warning in Colab | Colab has 2 CPUs, we default to 4 workers | Pass `--num_workers 2` |

---

## 📐 Commit Message Convention

```
<phase>/<scope>: <short description>

phase: t0 | phase1 | phase2 | phase3 | phase4 | phase5 | fix | docs
scope: data | model | train | explain | api | ui | config | misc
```

---

*Last updated: 2026-05-10 — Updated after Phase 1 training session*
