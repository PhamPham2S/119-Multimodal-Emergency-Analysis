from pathlib import Path
import sys

# Path 설정
SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from core.modeling import FusionModel
from core.multitask import MultiTaskLoss, MultiTaskLossController
from core.losses import ordinal_loss
from core.grad_monitor import GradMonitor

class DummyAudioEncoder(nn.Module):
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_values):
        batch_size = input_values.size(0)
        return torch.zeros(batch_size, self.hidden_size, device=input_values.device)


# Dummy components (디버그용)
class DummyTextEncoder(nn.Module):
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.size(0)
        return torch.zeros(batch_size, self.hidden_size, device=input_ids.device)


class DummyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "input_values": torch.randn(16000),      # audio dummy
            "input_ids": torch.randint(0, 1000, (32,)),
            "attention_mask": torch.ones(32),
            "urgency": torch.tensor(1),               # ordinal label
            "sentiment": torch.tensor(2),             # class label
        }


def collate_fn(batch):
    return {
        "input_values": torch.stack([b["input_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "urgency": torch.stack([b["urgency"] for b in batch]),
        "sentiment": torch.stack([b["sentiment"] for b in batch]),
    }


# Model
text_model = DummyTextEncoder(hidden_size=768)
audio_model = DummyAudioEncoder(hidden_size=768)

model = FusionModel(
    audio_model= audio_model,
    text_model= text_model,
    urgency_levels=3,       # 하 / 중 / 상
    sentiment_levels=4,     # 감정 클래스 수
)

# Loss
controller = MultiTaskLossController(
    warmup_epochs=0,        # debug에서는 바로 multitask
    urgency_weight=1.0,     # 일단 둘 다 1.0
    sentiment_weight =1.0,
    use_uncertainty=False,
)

criterion = MultiTaskLoss(
    urgency_loss_fn=ordinal_loss, # ordinal loss 사용
    sentiment_loss_fn=nn.CrossEntropyLoss(),
    controller=controller,
)

# Gradient monitor
grad_monitor = GradMonitor(model)

# Data
dataset = DummyDataset()
loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=collate_fn,
)

batch = next(iter(loader))

# Forward
outputs = model(
    input_values=batch["input_values"],
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
)

# Loss
loss_dict = criterion(
    outputs=outputs,
    targets={
        "urgency": batch["urgency"],
        "sentiment": batch["sentiment"],
    },
    epoch=0,
)

print("Loss dict:", loss_dict)

# Gradient analysis
grad_stats = grad_monitor.compute(
    {
        "urgency": loss_dict["urgency"],
        "sentiment": loss_dict["sentiment"],
    }
)

print("Gradient stats:", grad_stats)
