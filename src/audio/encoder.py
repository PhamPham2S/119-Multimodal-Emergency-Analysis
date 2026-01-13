# encoder.py
from __future__ import annotations
from typing import Optional, Union

import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except ImportError:
    raise SystemExit("transformers is required")


def masked_mean(
    sequence: torch.Tensor,
    mask: torch.Tensor,
    dim: int = 1,
    keepdim: bool = False,
) -> torch.Tensor:
    """마스크를 고려한 평균 pooling"""

    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)

    sequence = sequence.masked_fill(mask == 0, 0.0)
    sum_seq = torch.sum(sequence, dim=dim, keepdim=keepdim)
    sum_mask = torch.sum(mask, dim=dim, keepdim=keepdim).clamp(min=1e-9)

    return sum_seq / sum_mask


class AttentivePooling(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        scores = self.proj(sequence).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class AudioEncoder(nn.Module):
    def __init__(
        self,
        model: Union[str, nn.Module],
        pooling: str = "attn",
    ) -> None:
        super().__init__()

        # =========================
        # Dummy encoder 분기
        # =========================
        if isinstance(model, nn.Module):
            self.model = model
            self.is_dummy = True

            hidden_size = getattr(model, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("DummyAudioEncoder must have `hidden_size` attribute")

            self.pooling_mode = None
            self.pool = None
            self.proj = nn.Identity()
            return

        # =========================
        # HuggingFace encoder
        # =========================
        self.is_dummy = False
        self.model = AutoModel.from_pretrained(model)
        hidden_size = self.model.config.hidden_size

        self.pooling_mode = pooling
        if pooling == "mean":
            self.pool = None
        elif pooling == "attn":
            self.pool = AttentivePooling(hidden_size)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.proj = nn.Linear(hidden_size, hidden_size)

    def _feature_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if hasattr(self.model, "_get_feature_vector_attention_mask"):
            return self.model._get_feature_vector_attention_mask(seq_len, attention_mask)
        return None

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # =========================
        # Dummy forward
        # =========================
        if self.is_dummy:
            return self.model(input_values)

        # =========================
        # HuggingFace forward
        # =========================
        outputs = self.model(input_values, attention_mask=attention_mask)
        sequence = outputs.last_hidden_state

        feat_mask = self._feature_mask(attention_mask, sequence.shape[1])

        if self.pooling_mode == "mean":
            pooled = masked_mean(sequence, feat_mask)
        else:
            pooled = self.pool(sequence, feat_mask)

        return self.proj(pooled)
