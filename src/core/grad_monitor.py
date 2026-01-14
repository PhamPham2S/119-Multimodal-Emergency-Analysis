from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientMonitor:
    """
    공유 파라미터에 대한 task별 gradient 통계 계산.
    - 각 task의 gradient L2 norm
    - 두 task 간 gradient cosine similarity (urgency vs sentiment)
    """

    def __init__(self, model: nn.Module) -> None:
        # requires_grad=True인 모든 파라미터만 추출
        self.shared_params = [p for p in model.parameters() if p.requires_grad]

    @torch.no_grad()
    def _flatten_grads(self, grads):
        """
        gradient 리스트를 1D 텐서로 합치기
        None인 gradient는 0으로 대체
        """
        flat = []
        for g, p in zip(grads, self.shared_params):
            if g is None:
                flat.append(torch.zeros_like(p).reshape(-1))
            else:
                flat.append(g.reshape(-1))
        return torch.cat(flat)

    def compute(self, task_losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        각 task별 gradient norm과 urgency/sentiment 간 cosine similarity 계산
        """
        grads = {}

        # 각 task별 gradient 계산
        for task_name, loss in task_losses.items():
            grad_list = torch.autograd.grad(
                loss,
                self.shared_params,
                retain_graph=True,
                allow_unused=True
            )
            grads[task_name] = self._flatten_grads(grad_list)

        # gradient norm
        stats = {f"{task}_grad_norm": g.norm().item() for task, g in grads.items()}

        # urgency와 sentiment가 모두 있을 경우 cosine similarity 계산
        if "urgency" in grads and "sentiment" in grads:
            g1, g2 = grads["urgency"], grads["sentiment"]
            # batch가 없을 경우도 대비해서 1D vector 그대로 계산
            stats["grad_cosine_similarity"] = F.cosine_similarity(g1, g2, dim=0).item()

        return stats


class GradMonitor(GradientMonitor):
    """
    외부 코드에서 사용할 때 이름 간단히 하기 위한 alias
    """
    pass
