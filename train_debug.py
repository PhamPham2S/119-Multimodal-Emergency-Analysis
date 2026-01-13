from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.modeling import FusionModel
from core.multitask import MultiTaskLoss, MultiTaskLossController
from core.losses import ordinal_loss
from core.grad_monitor import GradMonitor

import torch.nn as nn


model = FusionModel(...)

controller = MultiTaskLossController(
    warmup_epochs=3,
    urgency_weight=1.0,
    sentiment_weight=0.5,
    use_uncertainty=False,
)
criterion = MultiTaskLoss(
    urgency_loss_fn=ordinal_loss,
    sentiment_loss_fn=nn.CrossEntropyLoss(),
    controller=controller,
)
grad_monitor = GradMonitor(model)

batch = next(iter(loader))

outputs = model(
    input_values=batch["input_values"],
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
)

loss_dict = criterion(
    outputs=outputs,
    targets={
        "urgency": batch["urgency"],
        "sentiment": batch["sentiment"],
    }
)

total_loss = loss_dict["total"]
print(loss_dict)
