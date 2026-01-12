from __future__ import annotations

import math
from pathlib import Path

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
    if orig_sr == target_sr:
        return waveform
    factor = math.gcd(orig_sr, target_sr)
    up = target_sr // factor
    down = orig_sr // factor
    return resample_poly(waveform, up, down).astype("float32")


def load_audio(path: Path, target_sr: int) -> torch.Tensor:
    data, sr = sf.read(str(path), always_2d=True)
    data = data.mean(axis=1).astype("float32")
    if sr != target_sr:
        data = resample_audio(data, sr, target_sr)
    waveform = torch.from_numpy(data)
    peak = waveform.abs().max().clamp(min=1e-6)
    return waveform / peak
