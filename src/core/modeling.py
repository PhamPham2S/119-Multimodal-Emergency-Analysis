from __future__ import annotations

from typing import Dict, Optional

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

        # Encoder 초기화
        self.audio_encoder = AudioEncoder(audio_model, pooling=pooling)
        self.text_encoder = TextEncoder(text_model)

        # 임베딩 차원
        audio_dim = self.audio_encoder.model.config.hidden_size
        text_dim = self.text_encoder.model.config.hidden_size

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.LayerNorm(audio_dim + text_dim),
            nn.Linear(audio_dim + text_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.urgency_head = nn.Linear(fusion_dim, urgency_levels - 1)  # ordinal
        self.sentiment_head = nn.Linear(fusion_dim, sentiment_levels)  # class

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        audio_key: str = "input_values",
        audio_mask_key: Optional[str] = "audio_mask",
        text_key: str = "input_ids",
        text_mask_key: Optional[str] = "text_mask",
    ) -> Dict[str, torch.Tensor]:
        """
        batch dict 기반 forward.
        더미/실제 encoder 모두 호환되도록 키 지정 가능
        """

        # Audio embedding
        audio_mask = batch[audio_mask_key] if audio_mask_key in batch else None
        audio_embed = self.audio_encoder(batch[audio_key], audio_mask)

        # Text embedding
        text_mask = batch[text_mask_key] if text_mask_key in batch else None
        text_embed = self.text_encoder(batch[text_key], text_mask)

        # Fusion
        shared = self.fusion(torch.cat([audio_embed, text_embed], dim=-1))

        # Output
        return {
            "urgency_logits": self.urgency_head(shared),
            "sentiment_logits": self.sentiment_head(shared),
        }

class FusionModel_train(nn.Module):
    def __init__(
        self,
        urgency_levels: int,
        sentiment_levels: int,
        pooling: str = "attn",
        fusion_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # 임베딩 차원
        audio_dim = 781
        text_dim = 768

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.LayerNorm(audio_dim + text_dim),
            nn.Linear(audio_dim + text_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.urgency_head = nn.Linear(fusion_dim, urgency_levels - 1)  # ordinal
        self.sentiment_head = nn.Linear(fusion_dim, sentiment_levels)  # class

    def forward(self, batch):
        audio_embed = batch["audio_embed"]   # (B, D_a)
        text_embed = batch["text_embed"]     # (B, D_t)

        shared = self.fusion(
            torch.cat([audio_embed, text_embed], dim=-1)
        )

        return {
            "urgency": self.urgency_head(shared),
            "sentiment": self.sentiment_head(shared),
        }
