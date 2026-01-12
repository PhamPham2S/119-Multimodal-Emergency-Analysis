from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.data_pipeline import build_dataloader
from core.losses import train_step
from core.modeling import FusionModel


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
    loader, label_mapper = build_dataloader(
        data_root=args.data_root,
        text_model=args.text_model,
        sample_rate=args.sample_rate,
        max_text_len=args.max_text_len,
        urgency_order=("\uD558", "\uC911", "\uC0C1"),
        sentiment_order=None,
        batch_size=args.batch_size,
    )
    model = FusionModel(
        audio_model=args.audio_model,
        text_model=args.text_model,
        urgency_levels=len(label_mapper.urgency_order),
        sentiment_levels=len(label_mapper.sentiment_order),
    )
    batch = next(iter(loader))
    loss = train_step(model, batch)
    print(f"batch_size={batch['input_values'].shape[0]} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
