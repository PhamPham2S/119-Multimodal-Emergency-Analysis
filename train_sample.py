# train_sample.py (수정 완료)
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
train_loader, label_mapper = build_dataloader(
    data_root=data_dir,
    text_model="beomi/KcELECTRA-base-v2022",  # KC ELECTRA 모델 지정
    sample_rate=16000,
    max_text_len=128,
    urgency_order=["상", "중", "하"],
    sentiment_order=["당황/난처", "불안/걱정", "중립", "기타부정"],
    batch_size=BATCH_SIZE
)

# 모델 초기화
model = FusionModel(
    audio_model="facebook/wav2vec2-base-960h",  # 예시 오디오 모델
    text_model="beomi/KcELECTRA-base-v2022",          # KC ELECTRA
    urgency_levels=3,                            # urgency 클래스 수
    sentiment_levels=4,                          # sentiment 클래스 수
    fusion_dim=256,
    dropout=0.2
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
        # batch 그대로 model에 전달
        for k in batch:
            batch[k] = batch[k].to(DEVICE)
        
        outputs = model(batch)  # FusionModel forward 호출

        loss = criterion(outputs, batch['urgency'], batch['sentiment'])
        loss.backward()

        # gradient 모니터링
        grad_monitor.log()  # task별 gradient norm, cosine similarity 출력

        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

print("학습 완료")
