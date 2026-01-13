# train_sample.py
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

# src 경로 추가
SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# 파이프라인, 모델, loss, grad monitor 불러오기
from core.data_pipeline import build_dataloader
from core.modeling import FusionModel
from core.multitask import MultiTaskLoss, MultiTaskLossController
from core.grad_monitor import GradMonitor
from core.losses import ordinal_loss

# 학습 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-4

# 데이터 로더 (Sample 데이터 사용)
data_dir = Path(__file__).resolve().parent / "data" / "Sample"
train_loader = build_dataloader(data_dir, batch_size=BATCH_SIZE, shuffle=True)

# 모델 초기화
model = FusionModel(
    audio_dim=768,      # 예시 값, 실제 encoder 출력 차원 확인
    text_dim=768,       # 예시 값, 실제 encoder 출력 차원 확인
    hidden_dim=256,
    num_urgency_classes=3,
    num_sentiment_classes=4
).to(DEVICE)

# MultiTask Loss 설정
controller = MultiTaskLossController(
    warmup_epochs=1,
    urgency_weight=1.0,
    sentiment_weight=0.5,
    use_uncertainty=False
)
criterion = MultiTaskLoss(
    urgency_loss_fn=ordinal_loss,
    sentiment_loss_fn=nn.CrossEntropyLoss(),
    controller=controller
)

# 옵티마이저
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# GradMonitor 초기화
grad_monitor = GradMonitor(model)

# 학습 루프
for epoch in range(EPOCHS):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        audio = batch['audio'].to(DEVICE)
        text = batch['text'].to(DEVICE)
        urgency_label = batch['urgency'].to(DEVICE)
        sentiment_label = batch['sentiment'].to(DEVICE)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(audio, text, attention_mask=attention_mask)

        loss = criterion(outputs, urgency_label, sentiment_label)
        loss.backward()

        # gradient 모니터링
        grad_monitor.log()  # task별 gradient norm, cosine similarity 출력

        optimizer.step()

        if batch_idx % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

print("학습 완료")
