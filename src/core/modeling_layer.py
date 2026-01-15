from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn


class FusionModel_train(nn.Module):
    def __init__(
        self,
        urgency_levels: int,
        fusion_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # embedding dims (고정)
        audio_dim = 781
        text_dim = 768
        input_dim = audio_dim + text_dim

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 4-layer urgency head
        self.urgency_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, urgency_levels - 1)  # ordinal
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        audio_embed = batch["audio_embed"]
        text_embed = batch["text_embed"]

        fused = self.fusion(
            torch.cat([audio_embed, text_embed], dim=-1)
        )

        return {
            "urgency": self.urgency_head(fused)
        }
