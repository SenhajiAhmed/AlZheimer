"""
generate_synthetic.py
=====================
Generates a synthetic clinical metadata CSV correlated to the 4 Alzheimer disease classes.
This proves the multimodal concept without real patient records.

Classes (matching Kaggle folder names):
  0 = NonDemented
  1 = VeryMildDemented
  2 = MildDemented
  3 = ModerateDemented

Columns generated:
  - patient_id       : unique identifier
  - class_label      : string class name
  - class_idx        : integer class index (0–3)
  - age              : age in years (Gaussian, shifts up with severity)
  - mmse             : Mini-Mental State Examination score (0–30, decreases with severity)
  - cdr              : Clinical Dementia Rating (0, 0.5, 1, 2)
  - education_years  : years of formal education (8–20)
  - apoe4            : APOE ε4 allele carrier (0 or 1, higher prevalence in demented)

Usage:
  python data/generate_synthetic.py
  python data/generate_synthetic.py --n_per_class 1000 --seed 99 --out data/synthetic_tabular.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# ── Per-class clinical distributions ──────────────────────────────────────────

CLASS_PARAMS = {
    "NonDemented": {
        "idx": 0,
        "age_mean": 62, "age_std": 7,
        "mmse_mean": 28.5, "mmse_std": 1.2,
        "cdr_probs": [0.95, 0.05, 0.00, 0.00],   # P(CDR=0), P(0.5), P(1), P(2)
        "apoe4_prob": 0.15,
        "edu_low": 12, "edu_high": 20,
    },
    "VeryMildDemented": {
        "idx": 1,
        "age_mean": 68, "age_std": 7,
        "mmse_mean": 26.0, "mmse_std": 1.8,
        "cdr_probs": [0.10, 0.85, 0.05, 0.00],
        "apoe4_prob": 0.30,
        "edu_low": 10, "edu_high": 18,
    },
    "MildDemented": {
        "idx": 2,
        "age_mean": 73, "age_std": 6,
        "mmse_mean": 21.0, "mmse_std": 3.0,
        "cdr_probs": [0.00, 0.15, 0.80, 0.05],
        "apoe4_prob": 0.45,
        "edu_low": 8, "edu_high": 16,
    },
    "ModerateDemented": {
        "idx": 3,
        "age_mean": 78, "age_std": 6,
        "mmse_mean": 13.0, "mmse_std": 4.0,
        "cdr_probs": [0.00, 0.00, 0.20, 0.80],
        "apoe4_prob": 0.60,
        "edu_low": 8, "edu_high": 14,
    },
}

CDR_VALUES = [0.0, 0.5, 1.0, 2.0]


def generate_class_samples(class_name: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    p = CLASS_PARAMS[class_name]

    age = rng.normal(p["age_mean"], p["age_std"], n).clip(50, 95).round(1)
    mmse = rng.normal(p["mmse_mean"], p["mmse_std"], n).clip(0, 30).round(1)
    cdr = rng.choice(CDR_VALUES, size=n, p=p["cdr_probs"])
    edu = rng.integers(p["edu_low"], p["edu_high"] + 1, n)
    apoe4 = rng.binomial(1, p["apoe4_prob"], n)

    return pd.DataFrame({
        "class_label": class_name,
        "class_idx": p["idx"],
        "age": age,
        "mmse": mmse,
        "cdr": cdr,
        "education_years": edu,
        "apoe4": apoe4,
    })


def generate(n_per_class: int = 800, seed: int = 42, out: str = "data/synthetic_tabular.csv") -> None:
    rng = np.random.default_rng(seed)
    frames = []

    for class_name in CLASS_PARAMS:
        df = generate_class_samples(class_name, n_per_class, rng)
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result.index.name = "patient_id"
    result = result.reset_index()

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f"✅  Generated {len(result)} synthetic patient records → {out_path}")
    print("\nClass distribution:")
    print(result["class_label"].value_counts().to_string())
    print("\nSample statistics:")
    print(result.groupby("class_label")[["age", "mmse", "cdr", "apoe4"]].mean().round(2).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Alzheimer clinical metadata")
    parser.add_argument("--n_per_class", type=int, default=800,
                        help="Number of synthetic patients per class (default: 800)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--out", type=str, default="data/synthetic_tabular.csv",
                        help="Output CSV path (default: data/synthetic_tabular.csv)")
    args = parser.parse_args()
    generate(args.n_per_class, args.seed, args.out)
