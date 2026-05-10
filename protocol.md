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

---

## 📐 Commit Message Convention

```
<phase>/<scope>: <short description>

phase: t0 | phase1 | phase2 | phase3 | phase4 | phase5 | fix | docs
scope: data | model | train | explain | api | ui | config | misc
```

---

*Last updated: 2026-05-10*
