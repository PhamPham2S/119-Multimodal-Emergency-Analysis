from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover
    raise SystemExit("soundfile is required for inference") from exc

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("transformers is required for inference") from exc

from audio.io import load_audio, resample_audio
from core.data_pipeline import build_records
from core.modeling import FusionModel

DEFAULT_URGENCY_ORDER = ("\uD558", "\uC911", "\uC0C1")
DEFAULT_SENTIMENT_ORDER = (
    "\uAE30\uD0C0\uBD80\uC815",
    "\uB2F9\uD669/\uB09C\uCC98",
    "\uBD88\uC548/\uAC70\uC815",
    "\uC911\uB9BD",
)


@dataclass
class InferenceResult:
    urgency_label: str
    urgency_logits: List[float]
    urgency_probs: List[float]
    sentiment_label: str
    sentiment_logits: List[float]
    sentiment_probs: List[float]


@dataclass
class LabelOrders:
    urgency_order: Tuple[str, ...]
    sentiment_order: Tuple[str, ...]


def load_label_orders(model_dir: Path, data_root: Path | None) -> LabelOrders:
    labels_path = model_dir / "labels.json"
    if labels_path.exists():
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        urgency = tuple(payload.get("urgency_order", DEFAULT_URGENCY_ORDER))
        sentiment = tuple(payload.get("sentiment_order", DEFAULT_SENTIMENT_ORDER))
        return LabelOrders(urgency_order=urgency, sentiment_order=sentiment)

    sentiments: Sequence[str] = []
    if data_root and data_root.exists():
        records = build_records(data_root)
        sentiments = sorted({r.sentiment for r in records if r.sentiment})
    if not sentiments:
        sentiments = DEFAULT_SENTIMENT_ORDER
    return LabelOrders(urgency_order=DEFAULT_URGENCY_ORDER, sentiment_order=tuple(sentiments))


def load_audio_from_bytes(data: bytes, target_sr: int) -> torch.Tensor:
    array, sr = sf.read(io.BytesIO(data), always_2d=True)
    array = array.mean(axis=1).astype("float32")
    if sr != target_sr:
        array = resample_audio(array, sr, target_sr)
    waveform = torch.from_numpy(array)
    peak = waveform.abs().max().clamp(min=1e-6)
    return waveform / peak


class InferenceEngine:
    def __init__(
        self,
        audio_model: str,
        text_model: str,
        model_dir: Path,
        label_orders: LabelOrders,
        sample_rate: int = 16000,
        max_text_len: int = 256,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = FusionModel(
            audio_model=audio_model,
            text_model=text_model,
            urgency_levels=len(label_orders.urgency_order),
            sentiment_levels=len(label_orders.sentiment_order),
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.sample_rate = sample_rate
        self.max_text_len = max_text_len
        self.label_orders = label_orders
        self._load_weights(model_dir)
        self.model.eval()

    def _load_weights(self, model_dir: Path) -> None:
        fusion_path = model_dir / "fusion_linear.pt"
        urgency_path = model_dir / "urgency_head.pt"
        sentiment_path = model_dir / "sentiment_head.pt"

        if fusion_path.exists():
            self.model.fusion[1].load_state_dict(torch.load(fusion_path, map_location=self.device))
        else:
            print(f"[warn] Missing {fusion_path}, using random init for fusion linear.")

        if urgency_path.exists():
            self.model.urgency_head.load_state_dict(torch.load(urgency_path, map_location=self.device))
        else:
            print(f"[warn] Missing {urgency_path}, using random init for urgency head.")

        if sentiment_path.exists():
            self.model.sentiment_head.load_state_dict(torch.load(sentiment_path, map_location=self.device))
        else:
            print(f"[warn] Missing {sentiment_path}, using random init for sentiment head.")

    def _tokenize(self, text: str) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        return {"input_ids": encoded["input_ids"], "text_mask": encoded["attention_mask"]}

    def _predict(self, waveform: torch.Tensor, text: str) -> InferenceResult:
        input_values = waveform.unsqueeze(0).to(self.device)
        audio_mask = torch.ones_like(input_values, dtype=torch.long)
        tokens = self._tokenize(text)
        batch = {
            "input_values": input_values,
            "audio_mask": audio_mask.to(self.device),
            "input_ids": tokens["input_ids"].to(self.device),
            "text_mask": tokens["text_mask"].to(self.device),
        }
        with torch.no_grad():
            outputs = self.model(batch)
        urgency_logits = outputs["urgency_logits"].squeeze(0)
        sentiment_logits = outputs["sentiment_logits"].squeeze(0)

        urgency_idx = int((urgency_logits > 0).sum().item())
        sentiment_idx = int(torch.argmax(sentiment_logits).item())

        urgency_probs = torch.sigmoid(urgency_logits).cpu().tolist()
        sentiment_probs = torch.softmax(sentiment_logits, dim=0).cpu().tolist()

        return InferenceResult(
            urgency_label=self.label_orders.urgency_order[urgency_idx],
            urgency_logits=urgency_logits.cpu().tolist(),
            urgency_probs=urgency_probs,
            sentiment_label=self.label_orders.sentiment_order[sentiment_idx],
            sentiment_logits=sentiment_logits.cpu().tolist(),
            sentiment_probs=sentiment_probs,
        )

    def predict_from_path(self, audio_path: Path, text: str) -> InferenceResult:
        waveform = load_audio(audio_path, self.sample_rate)
        return self._predict(waveform, text)

    def predict_from_bytes(self, audio_bytes: bytes, text: str) -> InferenceResult:
        waveform = load_audio_from_bytes(audio_bytes, self.sample_rate)
        return self._predict(waveform, text)
