from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# src 경로 추가
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.data_pipeline import build_dataloader
from core.modeling import FusionModel
from audio.encoder import build_audio_encoder
from text.encoder import build_text_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audio+Text urgency pipeline")
    parser.add_argument("--data-root", type=Path, default=Path("data/Sample"))
    parser.add_argument("--audio-model", type=str, default="facebook/hubert-base-ls960")
    parser.add_argument("--text-model", type=str, default="beomi/KcELECTRA-base")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-text-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # dataloader 생성
    loader, label_mapper = build_dataloader(
        data_root=args.data_root,
        text_model=args.text_model,
        sample_rate=args.sample_rate,
        max_text_len=args.max_text_len,
        urgency_order=("하", "중", "상"),
        sentiment_order=None,
        batch_size=args.batch_size,
    )

    # audio encoder 생성
    audio_model = build_audio_encoder(
        model_name=args.audio_model,
        sample_rate=args.sample_rate,
    )

    # text encoder 생성
    text_model = build_text_encoder(
        model_name=args.text_model,
        max_length=args.max_text_len,
    )

    # fusion model 생성
    model = FusionModel(
        audio_model=audio_model,
        text_model=text_model,
        urgency_levels=len(label_mapper.urgency_order),
        sentiment_levels=len(label_mapper.sentiment_order),
    )

    batch = next(iter(loader))

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_values=batch["input_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )

    # ordinal urgency decoding
    urgency_ids = (outputs["urgency_logits"] > 0).sum(dim=1).tolist()
    sentiment_ids = outputs["sentiment_logits"].argmax(dim=1).tolist()

    for idx, (urg_id, sent_id) in enumerate(zip(urgency_ids, sentiment_ids)):
        print(
            f"\nsample={idx} "
            f"urgency_logits={outputs['urgency_logits'][idx]} "
            f"sentiment_logits={outputs['sentiment_logits'][idx]}"
        )

        print(
            f"sample={idx} "
            f"urgency_ids={urg_id} "
            f"sentiment_ids={sent_id}"
        )

        urgency = label_mapper.urgency_order[urg_id]
        sentiment = label_mapper.sentiment_order[sent_id]

        print(
            f"sample={idx} "
            f"urgency_pred={urgency} "
            f"sentiment_pred={sentiment}"
        )

        print(
            f"\nsample={idx} "
            f"urgency_ids_y={outputs['urgency'][idx]} "
            f"sentiment_ids_y={outputs['sentiment'][idx]}"
        )

        print(
            f"sample={idx} "
            f"urgency_y={label_mapper.urgency_order[outputs['urgency'][idx]]} "
            f"sentiment_y={label_mapper.sentiment_order[outputs['sentiment'][idx]]}"
        )


if __name__ == "__main__":
    main()
