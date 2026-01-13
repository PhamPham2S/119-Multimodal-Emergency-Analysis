from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientMonitor:
    """
    Monitor task-wise gradient statistics on shared parameters.
    """

    def __init__(self, model: nn.Module) -> None:
        # shared parameters: requires_grad=True 인 모든 파라미터
        self.shared_params = [
            p for p in model.parameters() if p.requires_grad
        ]

    @torch.no_grad()
    def _flatten(self, grads):
        return torch.cat([g.reshape(-1) for g in grads if g is not None])

    def compute(
        self,
        task_losses: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Returns:
            {
                "urgency_grad_norm": float,
                "sentiment_grad_norm": float,
                "grad_cosine_similarity": float
            }
        """

        grads = {}

        for task, loss in task_losses.items():
            grad_list = torch.autograd.grad(
                loss,
                self.shared_params,
                retain_graph=True,
                allow_unused=True,
            )
            flat = self._flatten(grad_list)
            grads[task] = flat

        stats = {}

        # Gradient norms
        for task, g in grads.items():
            stats[f"{task}_grad_norm"] = g.norm(p=2).item()

        # Cosine similarity
        if "urgency" in grads and "sentiment" in grads:
            g1 = grads["urgency"]
            g2 = grads["sentiment"]

            if g1.numel() > 0 and g2.numel() > 0:
                cos = F.cosine_similarity(g1, g2, dim=0)
                stats["grad_cosine_similarity"] = cos.item()
            else:
                stats["grad_cosine_similarity"] = 0.0

        return stats


class GradMonitor(GradientMonitor):
    """
    Alias for external use.
    Keeps train / debug code concise.
    """
    pass
