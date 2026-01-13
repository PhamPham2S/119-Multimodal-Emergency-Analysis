from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from ..audio.data_io import load_audio
from ..text.processing import extract_text

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("transformers is required to run this pipeline") from exc


@dataclass(frozen=True)
class Record:
    audio_path: Path
    text: str
    urgency: str
    sentiment: str


@dataclass(frozen=True)
class LabelMapper:
    urgency_order: Tuple[str, ...]
    sentiment_order: Tuple[str, ...]
    urgency_to_id: Dict[str, int]
    sentiment_to_id: Dict[str, int]

    @classmethod
    def from_records(
        cls,
        records: Sequence[Record],
        urgency_order: Optional[Sequence[str]] = None,
        sentiment_order: Optional[Sequence[str]] = None,
    ) -> "LabelMapper":
        if urgency_order is None:
            raise ValueError("urgency_order must be provided for ordinal training")
        urgency = tuple(urgency_order)
        sentiments = sentiment_order
        if sentiments is None:
            unique = sorted({r.sentiment for r in records if r.sentiment})
            sentiments = unique
        sentiment = tuple(sentiments)
        return cls(
            urgency_order=urgency,
            sentiment_order=sentiment,
            urgency_to_id={k: i for i, k in enumerate(urgency)},
            sentiment_to_id={k: i for i, k in enumerate(sentiment)},
        )

    def encode(self, urgency: str, sentiment: str) -> Tuple[int, int]:
        return self.urgency_to_id[urgency], self.sentiment_to_id[sentiment]


def build_records(data_root: Path) -> List[Record]:
    audio_index = {p.stem: p for p in data_root.rglob("*.wav")}
    records: List[Record] = []
    for label_path in data_root.rglob("*.json"):
        audio_path = audio_index.get(label_path.stem)
        if audio_path is None:
            continue
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        urgency = payload.get("urgencyLevel")
        sentiment = payload.get("sentiment")
        if not urgency or not sentiment:
            continue
        text = extract_text(payload)
        records.append(
            Record(
                audio_path=audio_path,
                text=text,
                urgency=str(urgency),
                sentiment=str(sentiment),
            )
        )
    return records


class AudioTextDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Record],
        label_mapper: LabelMapper,
        sample_rate: int,
    ) -> None:
        self.records = list(records)
        self.label_mapper = label_mapper
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int, int]:
        record = self.records[idx]
        audio = load_audio(record.audio_path, self.sample_rate)
        urg_id, sent_id = self.label_mapper.encode(record.urgency, record.sentiment)
        return audio, record.text, urg_id, sent_id


def make_collate(
    tokenizer: AutoTokenizer,
    max_text_len: int,
):
    def collate(batch: Sequence[Tuple[torch.Tensor, str, int, int]]) -> Dict[str, torch.Tensor]:
        audios, texts, urg, sent = zip(*batch)
        lengths = torch.tensor([a.shape[-1] for a in audios], dtype=torch.long)
        max_len = int(lengths.max().item())
        padded = torch.zeros(len(audios), max_len, dtype=torch.float)
        audio_mask = torch.zeros(len(audios), max_len, dtype=torch.long)
        for i, audio in enumerate(audios):
            end = audio.shape[-1]
            padded[i, :end] = audio
            audio_mask[i, :end] = 1
        text_inputs = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        return {
            "input_values": padded,
            "audio_mask": audio_mask,
            "input_ids": text_inputs["input_ids"],
            "text_mask": text_inputs["attention_mask"],
            "urgency": torch.tensor(urg, dtype=torch.long),
            "sentiment": torch.tensor(sent, dtype=torch.long),
        }

    return collate


def build_dataloader(
    data_root: Path(),
    text_model: str,
    sample_rate: int,
    max_text_len: int,
    urgency_order: Sequence[str],
    sentiment_order: Optional[Sequence[str]] = None,
    batch_size: int = 4,
) -> Tuple[DataLoader, LabelMapper]:
    records = build_records(data_root)
    label_mapper = LabelMapper.from_records(
        records,
        urgency_order=urgency_order,
        sentiment_order=sentiment_order,
    )
    dataset = AudioTextDataset(records, label_mapper, sample_rate)
    tokenizer = AutoTokenizer.from_pretrained(text_model)
    collate = make_collate(tokenizer, max_text_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    return loader, label_mapper
