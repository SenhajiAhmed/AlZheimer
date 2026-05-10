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
    Pretrained vision encoder with optional partial unfreezing.

    Args:
        backbone:        "vit" | "resnet"
        freeze:          If True, freeze all backbone params. Overridden by unfreeze_last_n.
        unfreeze_last_n: Unfreeze only the last N transformer blocks (ViT only).
                         E.g. unfreeze_last_n=4 unfreezes encoder layers 8-11 + pooler.
                         Set to 0 to use the freeze flag as-is.
    """

    def __init__(self, backbone: str = "vit", freeze: bool = True, unfreeze_last_n: int = 0):
        super().__init__()

        if backbone not in BACKBONE_CONFIGS:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose from: {list(BACKBONE_CONFIGS)}")

        self.backbone_name = backbone
        self.embed_dim = BACKBONE_CONFIGS[backbone]["embed_dim"]

        if backbone == "vit":
            self._build_vit(freeze, unfreeze_last_n)
        else:
            self._build_resnet(freeze)

    # ── ViT ───────────────────────────────────────────────────────────────────

    def _build_vit(self, freeze: bool, unfreeze_last_n: int = 0) -> None:
        from transformers import ViTModel

        self.model = ViTModel.from_pretrained(
            "google/vit-base-patch16-224",
            add_pooling_layer=True,
            attn_implementation="eager"
        )
        # Freeze everything first
        for param in self.model.parameters():
            param.requires_grad = False

        if not freeze:
            # Unfreeze entire backbone
            for param in self.model.parameters():
                param.requires_grad = True
        elif unfreeze_last_n > 0:
            # Selectively unfreeze only the last N transformer encoder blocks + pooler
            total_layers = len(self.model.encoder.layer)  # 12 for ViT-Base
            unfreeze_from = max(0, total_layers - unfreeze_last_n)
            for i, layer in enumerate(self.model.encoder.layer):
                if i >= unfreeze_from:
                    for param in layer.parameters():
                        param.requires_grad = True
            # Also unfreeze the pooler
            for param in self.model.pooler.parameters():
                param.requires_grad = True

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
