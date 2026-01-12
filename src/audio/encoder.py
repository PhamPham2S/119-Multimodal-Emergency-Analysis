from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit("transformers is required to run this pipeline") from exc

from core.utils import masked_mean


class AttentivePooling(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, sequence: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        scores = self.proj(sequence).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class AudioEncoder(nn.Module):
    def __init__(self, model_name: str, pooling: str = "attn") -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        hidden = self.model.config.hidden_size
        if pooling == "mean":
            self.pool = None
        elif pooling == "attn":
            self.pool = AttentivePooling(hidden)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")
        self.pooling = pooling

    def _feature_mask(self, attention_mask: torch.Tensor, seq_len: int) -> Optional[torch.Tensor]:
        if hasattr(self.model, "_get_feature_vector_attention_mask"):
            return self.model._get_feature_vector_attention_mask(seq_len, attention_mask)
        return None

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_values, attention_mask=attention_mask)
        sequence = outputs.last_hidden_state
        feat_mask = self._feature_mask(attention_mask, sequence.shape[1])
        if self.pooling == "mean":
            if feat_mask is None:
                return sequence.mean(dim=1)
            return masked_mean(sequence, feat_mask)
        return self.pool(sequence, feat_mask)
