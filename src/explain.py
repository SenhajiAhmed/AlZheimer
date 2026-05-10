"""
src/explain.py
==============
Implementation of Attention Rollout for Vision Transformers (ViT).
Generates heatmaps showing which regions of the MRI influenced the prediction.

Usage:
    python src/explain.py --image_path data/raw/.../sample.jpg --checkpoint outputs/best_vit_...pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models.fusion import FusionClassifier
from src.models.mlp import TabularEncoder
from src.models.vision import VisionEncoder

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def attention_rollout(attentions, discard_ratio=0.9, head_fusion="mean"):
    """
    Computes Attention Rollout.
    Ref: https://arxiv.org/abs/2005.00928
    """
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise ValueError("head_fusion must be 'mean', 'max', or 'min'")

            # Drop the lowest attentions to reduce noise
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            # flat[0, indices] = 0 # Not implemented for simplicity in this version

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + I) / 2
            a = a / a.sum(dim=-1).keepdim(True)

            result = torch.matmul(a, result)
    
    # Extract the attention from the [CLS] token to all other tokens
    mask = result[0, 0, 1:]
    # In case of 224x224 image and 16x16 patches, we have 14x14 patches
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


def generate_heatmap(image_path: Path, mask: np.ndarray, output_path: Path):
    """Overlay the attention mask on the original image."""
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (224, 224))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    cam = heatmap + np.float32(img) / 255
    cam = cam / np.max(cam)
    
    result = np.uint8(255 * cam)
    cv2.imwrite(str(output_path), result)
    print(f"🔥 Heatmap saved to {output_path}")


# ── Main Inference & Explain Logic ──────────────────────────────────────────

def explain(
    image_path: str,
    tabular_data: list,
    checkpoint_path: str,
    backbone: str = "vit",
):
    device = get_device()
    
    # ── Load Checkpoint ──────────────────────────────────────────────────────
    print(f"📦 Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # ── Initialize Models ────────────────────────────────────────────────────
    vision = VisionEncoder(backbone=backbone, freeze=True).to(device)
    tab_encoder = TabularEncoder(input_dim=ckpt["tabular_dim"]).to(device)
    fusion = FusionClassifier(
        img_dim=768 if backbone == "vit" else 2048,
        tab_dim=64,
        num_classes=4
    ).to(device)
    
    tab_encoder.load_state_dict(ckpt["tab_encoder"])
    fusion.load_state_dict(ckpt["fusion"])
    
    vision.eval()
    tab_encoder.eval()
    fusion.eval()

    # ── Preprocess ───────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    tab_tensor = torch.tensor([tabular_data], dtype=torch.float32).to(device)

    # ── Forward Pass with Attention ──────────────────────────────────────────
    with torch.no_grad():
        # Get attention weights from ViT
        attentions = vision.get_attention_weights(image_tensor)
        
        # Get prediction
        h_img = vision(image_tensor)
        h_tab = tab_encoder(tab_tensor)
        logits = fusion(h_img, h_tab)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    # ── Compute Rollout ──────────────────────────────────────────────────────
    mask = attention_rollout(attentions)
    
    # ── Save Result ──────────────────────────────────────────────────────────
    output_dir = Path("outputs/xai")
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = output_dir / f"attn_{Path(image_path).stem}.jpg"
    generate_heatmap(image_path, mask, heatmap_path)
    
    class_names = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
    print(f"🔮 Prediction: {class_names[pred_idx]} ({confidence:.2%})")
    
    return {
        "prediction": class_names[pred_idx],
        "confidence": confidence,
        "heatmap_path": str(heatmap_path)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    # Mock tabular data for the CLI test [age, mmse, cdr, edu, apoe4] (scaled)
    # Note: Real CLI should use the scaler, but for test we use raw scaled values
    args = parser.parse_args()
    
    explain(
        image_path=args.image_path,
        tabular_data=[0.5, -1.0, 1.0, 0.2, 1.0], # Sample normalized data
        checkpoint_path=args.checkpoint
    )
