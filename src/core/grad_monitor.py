from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientMonitor:
    """
    Monitor task-wise gradient statistics on shared parameters.
    Computes per-task gradient norms and cosine similarity between tasks.
    """

    def __init__(self, model: nn.Module) -> None:
        # 공유 파라미터: requires_grad=True인 모든 파라미터
        self.shared_params = [p for p in model.parameters() if p.requires_grad]

    @torch.no_grad()
    def _flatten(self, grads):
        """
        Flatten gradients into a single 1D tensor.
        None gradients (unused parameters) are replaced with zeros to ensure consistent length.
        """
        return torch.cat([
            g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
            for g, p in zip(grads, self.shared_params)
        ])

    def compute(self, task_losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute gradient norms for each task and cosine similarity between tasks.

        Args:
            task_losses (Dict[str, torch.Tensor]): dictionary of task_name -> loss_tensor

        Returns:
            Dict[str, float]: {
                "task_grad_norm": float for each task,
                "grad_cosine_similarity": float between 'urgency' and 'sentiment' if both exist
            }
        """
        grads = {}

        # 각 task별 gradient 계산
        for task, loss in task_losses.items():
            grad_list = torch.autograd.grad(
                loss,
                self.shared_params,
                retain_graph=True,  # backward 연속 호출 가능하도록 그래프 유지
                allow_unused=True,  # 사용되지 않은 파라미터 gradient는 None
            )
            grads[task] = self._flatten(grad_list)

        # Gradient norms
        stats = {f"{task}_grad_norm": g.norm(p=2).item() for task, g in grads.items()}

        # Cosine similarity (urgency <-> sentiment)
        if "urgency" in grads and "sentiment" in grads:
            g1 = grads["urgency"]
            g2 = grads["sentiment"]
            stats["grad_cosine_similarity"] = F.cosine_similarity(g1, g2, dim=0).item()

        return stats


class GradMonitor(GradientMonitor):
    """
    Alias for external use.
    Keeps train / debug code concise.
    """
    pass
