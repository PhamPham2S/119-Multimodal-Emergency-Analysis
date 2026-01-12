from __future__ import annotations

import torch


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).type_as(sequence)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return (sequence * mask).sum(dim=1) / denom
