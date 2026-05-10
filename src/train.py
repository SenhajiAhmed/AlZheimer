"""
src/train.py
============
Full training and evaluation loop for the Diamond-Lite multimodal model.

Pipeline per forward pass:
    images  → VisionEncoder  → h_img  (B, img_dim)
    tabular → TabularEncoder → h_tab  (B, 64)
    [h_img ; h_tab] → FusionClassifier → logits (B, 4)

Metrics: Macro F1, Macro AUC-ROC, Confusion Matrix (saved to outputs/).

Usage:
    python src/train.py
    python src/train.py --backbone resnet --epochs 30 --batch_size 8
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data_loader import build_dataloaders
from src.models.fusion import FusionClassifier
from src.models.mlp import TabularEncoder
from src.models.vision import BACKBONE_CONFIGS, VisionEncoder

# ── Output directory ───────────────────────────────────────────────────────────

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_epoch_train(
    vision: VisionEncoder,
    tab_encoder: TabularEncoder,
    fusion: FusionClassifier,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler_amp,
) -> float:
    """Run one training epoch. Returns mean loss."""
    vision.eval()           # backbone stays frozen (eval mode = no dropout/BN update)
    tab_encoder.train()
    fusion.train()

    total_loss = 0.0

    for images, tabular, labels in tqdm(loader, desc="  train", leave=False):
        images  = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            with torch.no_grad():
                h_img = vision(images)          # frozen backbone
            h_tab  = tab_encoder(tabular)
            logits = fusion(h_img, h_tab)
            loss   = criterion(logits, labels)

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(tab_encoder.parameters()) + list(fusion.parameters()), max_norm=1.0
        )
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    vision: VisionEncoder,
    tab_encoder: TabularEncoder,
    fusion: FusionClassifier,
    loader,
    criterion: nn.Module,
    device: torch.device,
    class_names,
) -> dict:
    """Evaluate on a split. Returns dict with loss, f1, auc, preds, labels."""
    vision.eval()
    tab_encoder.eval()
    fusion.eval()

    all_logits, all_labels = [], []
    total_loss = 0.0

    for images, tabular, labels in tqdm(loader, desc="  eval ", leave=False):
        images  = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        h_img  = vision(images)
        h_tab  = tab_encoder(tabular)
        logits = fusion(h_img, h_tab)
        loss   = criterion(logits, labels)

        total_loss  += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits_cat = torch.cat(all_logits)           # (N, C)
    labels_cat = torch.cat(all_labels).numpy()   # (N,)
    probs      = torch.softmax(logits_cat, dim=1).numpy()
    preds      = logits_cat.argmax(dim=1).numpy()

    f1  = f1_score(labels_cat, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labels_cat, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")   # can fail if a class has no positive samples in small splits

    return {
        "loss":   total_loss / len(loader),
        "f1":     f1,
        "auc":    auc,
        "preds":  preds,
        "labels": labels_cat,
        "probs":  probs,
    }


def save_confusion_matrix(preds, labels, class_names, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    disp = ConfusionMatrixDisplay.from_predictions(
        labels, preds, display_labels=[c[:8] for c in class_names], cmap="Blues"
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"📊  Confusion matrix saved → {path}")


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    backbone: str = "vit",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    seed: int = 42,
    num_workers: int = 4,
) -> None:
    device = get_device()
    print(f"\n🖥  Device: {device}")
    print(f"🔧  Backbone: {backbone} | Epochs: {epochs} | BS: {batch_size} | LR: {lr}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        batch_size=batch_size, num_workers=num_workers, seed=seed
    )
    class_weights = meta["class_weights"].to(device)
    tabular_dim   = meta["tabular_dim"]
    class_names   = meta["class_names"]

    # ── Models ────────────────────────────────────────────────────────────────
    vision      = VisionEncoder(backbone=backbone, freeze=True).to(device)
    tab_encoder = TabularEncoder(input_dim=tabular_dim, embed_dim=64).to(device)
    fusion      = FusionClassifier(
        img_dim=BACKBONE_CONFIGS[backbone]["embed_dim"],
        tab_dim=64,
        num_classes=len(class_names),
    ).to(device)

    # ── Optimiser: only train fusion head + tabular encoder ───────────────────
    trainable_params = list(tab_encoder.parameters()) + list(fusion.parameters())
    optimizer  = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    amp_scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1    = 0.0
    patience_count = 0
    history        = []
    run_name       = f"{backbone}_{int(time.time())}"

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss = one_epoch_train(
            vision, tab_encoder, fusion, train_loader, optimizer, criterion, device, amp_scaler
        )
        val_metrics = evaluate(
            vision, tab_encoder, fusion, val_loader, criterion, device, class_names
        )
        scheduler.step()

        log = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_metrics["loss"], 4),
            "val_f1":     round(val_metrics["f1"], 4),
            "val_auc":    round(val_metrics["auc"], 4),
            "lr":         round(scheduler.get_last_lr()[0], 7),
        }
        history.append(log)

        print(
            f"  train_loss={log['train_loss']}  "
            f"val_loss={log['val_loss']}  "
            f"val_f1={log['val_f1']}  "
            f"val_auc={log['val_auc']}  "
            f"lr={log['lr']}"
        )

        # ── Checkpoint ───────────────────────────────────────────────────────
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_count = 0
            ckpt_path = OUTPUTS / f"best_{run_name}.pt"
            torch.save({
                "epoch":           epoch,
                "backbone":        backbone,
                "tab_encoder":     tab_encoder.state_dict(),
                "fusion":          fusion.state_dict(),
                "val_f1":          best_val_f1,
                "val_auc":         val_metrics["auc"],
                "tabular_dim":     tabular_dim,
            }, ckpt_path)
            print(f"  ✅  New best val F1={best_val_f1:.4f} — checkpoint saved → {ckpt_path}")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n⏹  Early stopping triggered (patience={patience})")
                break

    # ── Save training log ─────────────────────────────────────────────────────
    log_path = OUTPUTS / f"history_{run_name}.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n📈  Training history saved → {log_path}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n── Test Set Evaluation ──────────────────────────────────────────")
    test_metrics = evaluate(
        vision, tab_encoder, fusion, test_loader, criterion, device, class_names
    )
    print(f"  Test F1 (macro):    {test_metrics['f1']:.4f}")
    print(f"  Test AUC (macro):   {test_metrics['auc']:.4f}")
    print("\n" + classification_report(
        test_metrics["labels"], test_metrics["preds"], target_names=class_names
    ))

    cm_path = OUTPUTS / f"confusion_matrix_{run_name}.png"
    save_confusion_matrix(test_metrics["preds"], test_metrics["labels"], class_names, cm_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diamond-Lite multimodal Alzheimer model")
    parser.add_argument("--backbone",    type=str,   default="vit",  choices=["vit", "resnet"])
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=5)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--num_workers", type=int,   default=4)
    args = parser.parse_args()

    train(
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
    )
