from __future__ import annotations

from typing import Union, Optional

import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit("transformers is required to run this pipeline") from exc

from core.utils import masked_mean


class TextEncoder(nn.Module):
    def __init__(self, model: Union[str, nn.Module]) -> None:
        super().__init__()

        # Dummy encoder 분기
        if isinstance(model, nn.Module):
            self.model = model
            self.is_dummy = True

            hidden_size = getattr(model, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("DummyTextEncoder must have `hidden_size` attribute")

            self.proj = nn.Identity()
            return

        # HuggingFace encoder
        self.is_dummy = False
        self.model = AutoModel.from_pretrained(model)

        hidden_size = self.model.config.hidden_size
        self.proj = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Dummy forward
        if self.is_dummy:
            return self.model(input_ids, attention_mask)

        # HuggingFace forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return self.proj(pooled)

        return self.proj(masked_mean(outputs.last_hidden_state, attention_mask))
