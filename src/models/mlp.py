"""
src/models/mlp.py
=================
Lightweight MLP encoder for tabular clinical features.

Architecture:
    Linear(input_dim → 128) → BatchNorm → ReLU → Dropout
    Linear(128 → 64)        → BatchNorm → ReLU → Dropout
    Linear(64  → embed_dim)

Usage:
    from src.models.mlp import TabularEncoder
    encoder = TabularEncoder(input_dim=5, embed_dim=64)
    feats = encoder(tabular_tensor)  # (B, 64)
"""

import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    """
    3-layer MLP that maps a tabular feature vector to a fixed-size embedding.

    Args:
        input_dim:  Number of tabular features (default: 5 — age, mmse, cdr, edu, apoe4)
        embed_dim:  Output embedding dimension (default: 64)
        dropout:    Dropout rate applied after each hidden layer (default: 0.3)
    """

    def __init__(self, input_dim: int = 5, embed_dim: int = 64, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, embed_dim),
        )
        self.embed_dim = embed_dim
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) float tensor of normalised tabular features
        Returns:
            embedding: (B, embed_dim)
        """
        return self.net(x)
