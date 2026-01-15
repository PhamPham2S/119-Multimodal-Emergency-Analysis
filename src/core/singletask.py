from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn


class SingleTaskLoss(nn.Module):
    def __init__(
        self,
        loss_fn: nn.Module,
        task_name: str,
    ) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.task_name = task_name

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        loss = self.loss_fn(
            outputs[self.task_name],
            targets[self.task_name],
        )

        return {
            self.task_name: loss,
            "total": loss,
        }
