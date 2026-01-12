# io.py
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover
    raise SystemExit("soundfile is required to run this pipeline") from exc

try:
    from scipy.signal import resample_poly
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scipy is required to run this pipeline") from exc


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """오디오 샘플링 레이트를 정밀하게 변환합니다."""
    if orig_sr == target_sr:
        return waveform
    factor = math.gcd(orig_sr, target_sr)
    up = target_sr // factor
    down = orig_sr // factor
    return resample_poly(waveform, up, down).astype("float32")


def load_audio(path: Union[str, Path], target_sr: int) -> torch.Tensor:
    """
    오디오 파일을 로드하여 정규화된 1D 텐서로 반환합니다.
    - Mono 변환, Resampling, Peak Normalization 포함
    """
    path_str = str(path)
    
    # 1. Load & Mono Mix
    data, sr = sf.read(path_str, always_2d=True)
    data = data.mean(axis=1).astype("float32")
    
    # 2. Resample
    if sr != target_sr:
        data = resample_audio(data, sr, target_sr)
        
    # 3. Standardization (Wav2Vec 2.0 맞춤형 정규화)
    # 기존 Peak Normalization 대신 평균 0, 표준편차 1로 맞춤
    waveform = torch.from_numpy(data)
    
    if waveform.abs().max() > 0: # 무음 파일이 아닐 때만
        mean = waveform.mean()
        std = waveform.std()
        # 1e-7은 0으로 나누기 방지용 아주 작은 수
        waveform = (waveform - mean) / (std + 1e-7)
        
    return waveform


def get_audio_paths(directory: Union[str, Path], extension: str = ".wav") -> List[Path]:
    """디렉토리 내의 특정 확장자 파일 경로를 모두 찾습니다."""
    target_dir = Path(directory)
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory not found: {target_dir}")
    
    # extension이 *.wav 형태가 아니라면 붙여줌
    pattern = extension if extension.startswith("*") else f"*{extension}"
    return list(target_dir.rglob(pattern))


def save_feature(feature: Union[np.ndarray, torch.Tensor], save_path: Union[str, Path]) -> None:
    """추출된 특징 벡터를 .npy 파일로 저장합니다."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(feature, torch.Tensor):
        data = feature.detach().cpu().numpy()
    else:
        data = feature
        
    np.save(str(save_path), data)