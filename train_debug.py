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

# Dummy encoders
class DummyAudioEncoder(nn.Module):
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = type("Config", (), {"hidden_size": hidden_size})()

    def forward(self, input_values, attention_mask=None):
        batch_size = input_values.size(0)
        return torch.zeros(batch_size, self.hidden_size, device=input_values.device, dtype=torch.float)


class DummyTextEncoder(nn.Module):
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = type("Config", (), {"hidden_size": hidden_size})()

    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.size(0)
        return torch.zeros(batch_size, self.hidden_size, device=input_ids.device, dtype=torch.float)


# Dummy dataset
class DummyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "input_values": torch.randn(16000, dtype=torch.float),  # audio dummy
            "input_ids": torch.randint(0, 1000, (32,), dtype=torch.long),
            "audio_mask": torch.ones(16000, dtype=torch.float),
            "text_mask": torch.ones(32, dtype=torch.float),
            "urgency": torch.tensor(1, dtype=torch.float),          # ordinal label float
            "sentiment": torch.tensor(2, dtype=torch.long),         # class label LongTensor
        }


def collate_fn(batch):
    return {
        "input_values": torch.stack([b["input_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "audio_mask": torch.stack([b["audio_mask"] for b in batch]),
        "text_mask": torch.stack([b["text_mask"] for b in batch]),
        "urgency": torch.stack([b["urgency"] for b in batch]),
        "sentiment": torch.stack([b["sentiment"] for b in batch]),
    }


# Dummy FusionModel
class DummyFusionModel(nn.Module):
    def __init__(self, audio_model, text_model, urgency_levels=3, sentiment_levels=4):
        super().__init__()
        self.audio_model = audio_model
        self.text_model = text_model
        self.urgency_fc = nn.Linear(audio_model.hidden_size + text_model.hidden_size, urgency_levels)
        self.sentiment_fc = nn.Linear(audio_model.hidden_size + text_model.hidden_size, sentiment_levels)

    def forward(self, batch):
        audio_feat = self.audio_model(batch["input_values"], batch.get("audio_mask"))
        text_feat = self.text_model(batch["input_ids"], batch.get("text_mask"))
        fused = torch.cat([audio_feat, text_feat], dim=-1)
        return {
            "urgency": self.urgency_fc(fused).float(),      # logits float
            "sentiment": self.sentiment_fc(fused),          # logits float
        }

# Model
# dummy data to test
text_model = DummyTextEncoder(hidden_size=768)
audio_model = DummyAudioEncoder(hidden_size=768)

model = DummyFusionModel(
    audio_model=audio_model,
    text_model=text_model,
    urgency_levels=3,       # 하 / 중 / 상
    sentiment_levels=4,     # 감정 클래스 수
)

# Loss
controller = MultiTaskLossController(
    warmup_epochs=0,
    urgency_weight=1.0,
    sentiment_weight=1.0,
    use_uncertainty=False,
)

criterion = MultiTaskLoss(
    urgency_loss_fn=ordinal_loss,
    sentiment_loss_fn=nn.CrossEntropyLoss(),
    controller=controller,
)

grad_monitor = GradMonitor(model)
dataset = DummyDataset()
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
batch = next(iter(loader))
outputs = model(batch)

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
