from pathlib import Path
import argparse
import yaml
import sys

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.core.data_pipeline import build_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/local.yaml")
    parser.add_argument("--data_root", type=str, help="Override data_root from config")
    args = parser.parse_args()

    # config 경로 해석 (항상 프로젝트 루트 기준)
    cfg_path = Path(__file__).resolve().parent / args.config

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = Path(args.data_root) if args.data_root else Path(cfg["data_root"])

    loader, label_mapper = build_dataloader(
        data_root=data_root,
        text_model=cfg["text_model"],
        sample_rate=cfg["sample_rate"],
        max_text_len=cfg["max_text_len"],
        urgency_order=["상", "중", "하"],
        sentiment_order=["당황/난처", "불안/걱정", "중립", "기타부정"],
        batch_size=cfg["batch_size"],
    )

    batch = next(iter(loader))
    print("Audio batch shape:", batch["input_values"].shape)
    print("Text input_ids shape:", batch["input_ids"].shape)


if __name__ == "__main__":
    main()
