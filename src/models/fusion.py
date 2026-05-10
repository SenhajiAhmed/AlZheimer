"""
src/models/fusion.py
====================
Late-fusion classification head.

Takes image embedding h_img and tabular embedding h_tab, concatenates them,
and passes through a two-layer classification head:

    z = ReLU(W · [h_img ; h_tab] + b)
    logits = Linear(z → num_classes)

Usage:
    from src.models.fusion import FusionClassifier
    head = FusionClassifier(img_dim=768, tab_dim=64, num_classes=4)
    logits = head(img_feats, tab_feats)   # (B, 4)
"""

import torch
import torch.nn as nn


class FusionClassifier(nn.Module):
    """
    Late-fusion head: concatenate + classify.

    Args:
        img_dim:     Dimension of the image feature vector (768 for ViT, 2048 for ResNet)
        tab_dim:     Dimension of the tabular embedding (default: 64)
        num_classes: Number of output classes (default: 4)
        hidden_dim:  Hidden dimension of the fusion layer (default: 256)
        dropout:     Dropout rate (default: 0.3)
    """

    def __init__(
        self,
        img_dim: int,
        tab_dim: int = 64,
        num_classes: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        fused_dim = img_dim + tab_dim

        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h_img: torch.Tensor, h_tab: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_img: (B, img_dim) image feature vector
            h_tab: (B, tab_dim) tabular embedding
        Returns:
            logits: (B, num_classes) — raw (un-softmaxed) class scores
        """
        z = torch.cat([h_img, h_tab], dim=1)   # (B, img_dim + tab_dim)
        return self.head(z)
