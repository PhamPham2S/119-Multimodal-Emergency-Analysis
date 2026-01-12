from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from audio.encoder import AudioEncoder
from text.encoder import TextEncoder


class FusionModel(nn.Module):
    def __init__(
        self,
        audio_model: str,
        text_model: str,
        urgency_levels: int,
        sentiment_levels: int,
        pooling: str = "attn",
        fusion_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # hard sharing으로 진행
        self.audio_encoder = AudioEncoder(audio_model, pooling=pooling)
        self.text_encoder = TextEncoder(text_model)
        audio_dim = self.audio_encoder.model.config.hidden_size
        text_dim = self.text_encoder.model.config.hidden_size
        self.fusion = nn.Sequential(
            nn.LayerNorm(audio_dim + text_dim),
            nn.Linear(audio_dim + text_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.urgency_head = nn.Linear(fusion_dim, urgency_levels - 1)
        self.sentiment_head = nn.Linear(fusion_dim, sentiment_levels)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        audio_embed = self.audio_encoder(batch["input_values"], batch["audio_mask"])
        text_embed = self.text_encoder(batch["input_ids"], batch["text_mask"])
        shared = self.fusion(torch.cat([audio_embed, text_embed], dim=-1))
        return {
            "urgency_logits": self.urgency_head(shared),
            "sentiment_logits": self.sentiment_head(shared),
            "urgency": batch["urgency"],
            "sentiment": batch["sentiment"],
        }