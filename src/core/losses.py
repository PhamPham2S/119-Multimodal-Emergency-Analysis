from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def ordinal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    thresholds = logits.shape[1]
    levels = torch.arange(thresholds, device=targets.device)
    targets = targets.unsqueeze(-1)
    binary = (targets > levels).float()
    return nn.functional.binary_cross_entropy_with_logits(logits, binary)


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    urgency_weight: float = 1.0,
    sentiment_weight: float = 1.0,
) -> torch.Tensor:
    outputs = model(batch)
    loss_urg = ordinal_loss(outputs["urgency_logits"], batch["urgency"])
    loss_sent = nn.functional.cross_entropy(outputs["sentiment_logits"], batch["sentiment"])
    return urgency_weight * loss_urg + sentiment_weight * loss_sent
