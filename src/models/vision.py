"""
src/models/vision.py
====================
Vision backbone wrapper. Supports:
  - ViT  (google/vit-base-patch16-224 via Hugging Face transformers)
  - ResNet50 (torchvision)

The backbone is fully frozen by default. Only the final classification head
is replaced with an Identity layer to expose the raw feature embedding.

Usage:
    from src.models.vision import VisionEncoder
    model = VisionEncoder(backbone="vit")    # or "resnet"
    feats = model(images)   # shape: (B, 768) for ViT / (B, 2048) for ResNet
"""

import torch
import torch.nn as nn
from torchvision import models


# ── Backbone registry ──────────────────────────────────────────────────────────

BACKBONE_CONFIGS = {
    "vit":    {"embed_dim": 768},
    "resnet": {"embed_dim": 2048},
}


class VisionEncoder(nn.Module):
    """
    Frozen pretrained vision encoder.

    Args:
        backbone:  "vit" | "resnet"
        freeze:    If True (default), freezes all backbone parameters.
                   Set to False only for fine-tuning (use very low LR ~1e-5).
    """

    def __init__(self, backbone: str = "vit", freeze: bool = True):
        super().__init__()

        if backbone not in BACKBONE_CONFIGS:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose from: {list(BACKBONE_CONFIGS)}")

        self.backbone_name = backbone
        self.embed_dim = BACKBONE_CONFIGS[backbone]["embed_dim"]

        if backbone == "vit":
            self._build_vit(freeze)
        else:
            self._build_resnet(freeze)

    # ── ViT ───────────────────────────────────────────────────────────────────

    def _build_vit(self, freeze: bool) -> None:
        from transformers import ViTModel

        self.model = ViTModel.from_pretrained(
            "google/vit-base-patch16-224",
            add_pooling_layer=True,
        )
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        # Mark as feature extractor (output pooled [CLS] token)
        self._is_vit = True

    # ── ResNet50 ──────────────────────────────────────────────────────────────

    def _build_resnet(self, freeze: bool) -> None:
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if freeze:
            for param in base.parameters():
                param.requires_grad = False
        # Strip the final FC layer → exposes 2048-dim avg-pooled feature vector
        self.model = nn.Sequential(*list(base.children())[:-1])  # output: (B, 2048, 1, 1)
        self._is_vit = False

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224) normalised image tensor
        Returns:
            features: (B, embed_dim) float tensor
        """
        if self._is_vit:
            outputs = self.model(pixel_values=pixel_values)
            return outputs.pooler_output          # (B, 768)
        else:
            feats = self.model(pixel_values)      # (B, 2048, 1, 1)
            return feats.flatten(1)               # (B, 2048)

    def get_attention_weights(self, pixel_values: torch.Tensor):
        """
        Returns raw attention weights from the last ViT layer.
        Only valid for backbone='vit'. Used by Attention Rollout in explain.py.
        """
        if not self._is_vit:
            raise RuntimeError("get_attention_weights() is only available for ViT backbone.")
        outputs = self.model(pixel_values=pixel_values, output_attentions=True)
        return outputs.attentions   # tuple of (B, num_heads, seq_len, seq_len) per layer
