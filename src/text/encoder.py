from __future__ import annotations

import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit("transformers is required to run this pipeline") from exc

from core.utils import masked_mean


class TextEncoder(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled
        return masked_mean(outputs.last_hidden_state, attention_mask)
