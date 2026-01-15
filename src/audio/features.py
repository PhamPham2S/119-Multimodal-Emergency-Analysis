from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise SystemExit("librosa is required to extract handcrafted audio features") from exc


def extract_handcrafted_features(
    waveform: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    """Extract 13 handcrafted audio features (see audio/embedding.py)."""
    if waveform.numel() == 0:
        return torch.zeros(13, dtype=torch.float32)

    y = waveform.detach().cpu().numpy().astype("float32")
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        f0, _, voiced_probs = librosa.pyin(y, fmin=60, fmax=1000, frame_length=2048)
        f0_clean = f0[~np.isnan(f0)]
        if f0_clean.size > 0:
            f0_mean = float(np.mean(f0_clean))
            f0_std = float(np.std(f0_clean))
        else:
            f0_mean = 0.0
            f0_std = 0.0
        voiced_prob = float(np.mean(voiced_probs)) if voiced_probs is not None else 0.0

        spectrum = np.abs(librosa.stft(y))
        cent = float(np.mean(librosa.feature.spectral_centroid(S=spectrum, sr=sample_rate)))
        bw = float(np.mean(librosa.feature.spectral_bandwidth(S=spectrum, sr=sample_rate)))
        contrast = float(np.mean(librosa.feature.spectral_contrast(S=spectrum, sr=sample_rate)))
        flat = float(np.mean(librosa.feature.spectral_flatness(S=spectrum)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=spectrum, sr=sample_rate)))

        rms = float(np.mean(librosa.feature.rms(y=y)))
        onset = float(np.mean(librosa.onset.onset_strength(y=y, sr=sample_rate)))

        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))
        tonality = 1.0 - flat

        features: Sequence[float] = (
            f0_mean,
            f0_std,
            voiced_prob,
            cent,
            bw,
            contrast,
            flat,
            rolloff,
            rms,
            onset,
            zcr_mean,
            zcr_std,
            tonality,
        )
        values = np.array(features, dtype="float32")
    except Exception:
        values = np.zeros(13, dtype="float32")

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(values)
