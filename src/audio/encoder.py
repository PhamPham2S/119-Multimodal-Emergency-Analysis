# encoder.py
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except ImportError:
    raise SystemExit("transformers is required")

def masked_mean(sequence: torch.Tensor, mask: torch.Tensor, dim: int = 1, keepdim: bool = False) -> torch.Tensor:
    """마스크를 고려하여 평균을 구하는 유틸리티 함수 (차원 충돌 수정됨)"""
    
    # 🚨 [수정 핵심] 마스크 차원 확장: (Batch, Time) -> (Batch, Time, 1)
    if mask.dim() == 2:
        mask_expanded = mask.unsqueeze(-1)
    else:
        mask_expanded = mask
        
    # 이제 차원이 (Batch, Time, 1)이라서 (Batch, Time, 768)과 잘 맞습니다.
    sequence = sequence.masked_fill(mask_expanded == 0, 0.0)
    
    # 합계 계산
    sum_seq = torch.sum(sequence, dim=dim, keepdim=keepdim)
    
    # 마스크 합계 (유효 길이)
    sum_mask = torch.sum(mask_expanded, dim=dim, keepdim=keepdim)
    sum_mask = sum_mask.clamp(min=1e-9) # 0 나누기 방지
    
    return sum_seq / sum_mask

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
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", pooling: str = "attn") -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size
        
        self.pooling_mode = pooling
        if pooling == "mean":
            self.pool = None
        elif pooling == "attn":
            self.pool = AttentivePooling(hidden_size)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.proj = nn.Linear(hidden_size, hidden_size)

    def _feature_mask(self, attention_mask: torch.Tensor, seq_len: int) -> Optional[torch.Tensor]:
        if attention_mask is None: return None
        if hasattr(self.model, "_get_feature_vector_attention_mask"):
            return self.model._get_feature_vector_attention_mask(seq_len, attention_mask)
        return None

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_values, attention_mask=attention_mask)
        sequence = outputs.last_hidden_state 
        feat_mask = self._feature_mask(attention_mask, sequence.shape[1])
        
        if self.pooling_mode == "mean":
            pooled_output = masked_mean(sequence, feat_mask)
        else:
            pooled_output = self.pool(sequence, feat_mask)
            
        return self.proj(pooled_output)