from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class MultiTaskLossController(nn.Module):
    """
    Multi-task loss controller for urgency / sentiment.

    Responsibilities:
    - urgency-only warmup
    - static or scheduled loss weighting
    - optional uncertainty-based weighting
    """

    def __init__(
        self,
        warmup_epochs: int = 0,
        urgency_weight: float = 1.0,
        sentiment_weight: float = 1.0,
        use_uncertainty: bool = False,
    ) -> None:
        super().__init__()

        self.warmup_epochs = warmup_epochs
        self.base_weights = {
            "urgency": urgency_weight,
            "sentiment": sentiment_weight,
        }

        self.use_uncertainty = use_uncertainty

        if use_uncertainty:
            # log(sigma^2) 형태로 두는 것이 수치적으로 안정적
            self.log_vars = nn.ParameterDict(
                {
                    "urgency": nn.Parameter(torch.zeros(1)),
                    "sentiment": nn.Parameter(torch.zeros(1)),
                }
            )

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        epoch: int,
    ) -> torch.Tensor:
        """
        Args:
            losses: {"urgency": loss_urg, "sentiment": loss_sent}
            epoch: current epoch index
        """

        # 1. Warm-up: urgency only
        if epoch < self.warmup_epochs:
            return losses["urgency"]

        # 2. Uncertainty-based weighting
        if self.use_uncertainty:
            total = 0.0
            for task, loss in losses.items():
                log_var = self.log_vars[task]
                precision = torch.exp(-log_var)
                total = total + precision * loss + log_var
            return total

        # 3. Static / scheduled weighting
        total_loss = 0.0
        for task, loss in losses.items():
            weight = self.base_weights.get(task, 1.0)
            total_loss = total_loss + weight * loss

        return total_loss

    def get_current_weights(self) -> Dict[str, float]:
        """
        For logging / debugging.
        """
        if self.use_uncertainty:
            return {
                task: float(torch.exp(-log_var).detach().cpu())
                for task, log_var in self.log_vars.items()
            }
        return self.base_weights.copy()
