# 🚨 멀티모달 기반 119 신고 긴급도 판별

## 👥 Team & Roles

- **전체(2)**  
  - 문제 정의, 모델 아키텍처 설계 및 고도화
  - 실험 설계, 결과 해석, 발표 자료 정리  

- **오디오 담당**  
  - 음성 전처리 및 인코더 구축  
  - HuBERT/wav2vec2 기반 오디오 인코더 실험  

- **텍스트 담당**  
  - STT 텍스트 처리 기준 정의  
  - 텍스트 인코더 구축 및 분석  

- **학습 담당**  
  - Dataset / DataLoader 구성  
  - 멀티태스크 학습 루프 및 loss 설계  

---

## 📌 Overview
본 프로젝트는 **119 신고 음성 및 STT 텍스트를 활용하여 신고의 긴급도를 판별**하는 멀티모달 분류 모델을 구현하는 것을 목표로 한다.

- **Input**:  
  - 신고 음성 파일 (`.wav`)  
  - 음성을 변환한 STT 텍스트  (`json`)

- **Output**:  
  - `urgency`: 하 / 중 / 상  
  - `sentiment`: 당황/난처, 불안/걱정, 중립, 기타부정

신고 상황에서는 텍스트 정보만으로는 포착하기 어려운 **비언어적 신호(톤, 속도, 끊김, 호흡 등)**가 중요한 단서가 되므로, 음성과 텍스트를 함께 사용하는 멀티모달 접근을 채택하였다.

---

## 🧠 Problem Formulation
### 왜 Regression이 아닌가?
**긴급도(urgency)**는 연속적인 수치가 아닌 **의사결정 단계**이다.

- 실제 대응은  
  - 즉시 출동  
  - 상황 확인  
  - 비긴급  
  과 같은 **이산적 판단**으로 이루어진다.
- `2.7`과 같은 긴급도 값은 현실적인 의미가 없다.

따라서 본 프로젝트에서는 긴급도를 **Ordinal Classification (하 < 중 < 상)** 문제로 정의하였다.

---

## 📊 Dataset
- **출처**: AIHub – 119 지능형 신고접수 음성 인식 데이터
- **구성**
  - 음성 파일 (wav)
  - STT 텍스트
  - 긴급도 및 감정상태 라벨

### 데이터 전략

---

## 🔄 Pipeline Overview
1. **Audio Pipeline**
   - 음성 로드 및 리샘플링
   - wav2vec2 기반 음성 인코딩

2. **Text Pipeline**
   - STT 텍스트 입력
   - KcELECTRA를 통한 임베딩 생성

3. **Multimodal Fusion**
   - 음성/텍스트 임베딩을 결합하여 하나의 표현으로 통합

4. **Prediction**
   - 얕은 MLP Head를 통해
     - 긴급도(하/중/상)
     - 감정 상태
     별도 예측

---

## 🏗 Model Architecture
<div align="center">
  <img src="https://github.com/user-attachments/assets/914a9fab-0141-47dc-8fe0-ea837d7dbd26" width="413"/>
</div>

### Audio Encoder
- **Model**: wav2vec2 (pretrained)
- **Output**:
  - 프레임 단위 시퀀스 표현  
    - `A_frames ∈ (T × D)`
- 이후 pooling을 통해 고정 길이 벡터로 변환

### Text Encoder
- **Model**: KcELECTRA (pretrained)
- **Output**:
  - 토큰 단위 표현 `T_tokens ∈ (N × D)`
  - CLS 토큰 기반 문장 표현

### Pooling
- 시퀀스 표현을 판단에 적합한 벡터로 요약
- Attention Pooling 사용

### Prediction Head
- **얕은 MLP 구조**
Linear → ReLU → Dropout → Linear


---

## ⚙ Training Strategy
- **Encoder Freeze 전략**
- Audio / Text 인코더는 고정
- Fusion 및 Head만 학습
- **Loss**
- Urgency: Ordinal 또는 Weighted Cross Entropy
- Sentiment: Cross Entropy
- **Class imbalance 대응**
- class weight 적용

### 학습 파라미터 규모
- 전체 모델 파라미터: 200M+
- **실제 학습 파라미터**: 약 **2.6M**

---

## 🧪 Experiments
### Baselines
- Audio-only
- Text-only

### Proposed Model
- Audio + Text 멀티모달 모델

### Evaluation Metrics
- Macro F1-score
- Urgency 단계별 혼동 행렬

---

## 📈 Results (Summary)
- 멀티모달 모델이 audio-only, text-only 대비 **일관된 성능 향상**
- 특히 다음 상황에서 강점 확인:
- STT 오류가 심한 경우
- 텍스트는 평범하지만 음성에 긴급 신호가 있는 경우

---

## 🔍 Error Analysis & Insights
- **조용하지만 위급한 상황**
- 텍스트 정보만으로는 탐지 어려움
- **위급 단어 사용 + 차분한 음성**
- 멀티모달 결합을 통해 과잉 판단 감소

→ 음성과 텍스트의 **불일치 자체가 중요한 신호**가 될 수 있음을 확인

---

## 📂 Project Structure

NewJeans-5
├── service/
| ├── frontend/
| ├── backend/
| ├── plan_frontend.md
├── src/
| ├── core/  # 전체 아키텍처 및 파이프라인
| ├── audio/ # 음성 전처리 및 오디오 인코더
| ├── text/  # 텍스트 인코더 및 처리
| └── train/ # 학습, loss, dataset
└── requirements.txt

---

## ▶ How to Run
```bash
# 학습
python scripts/train.py

# 평가
python scripts/eval.py
⚠ Limitations & Future Work
긴 통화에 대한 turn-level modeling 미적용

더 많은 데이터 확보 시 encoder fine-tuning 가능성

실시간 시스템 적용을 위한 경량화 필요

✨ Key Takeaway
긴급도 판별은 “무엇을 말했는가”보다
“어떻게 말했는가”가 더 중요한 순간이 존재한다.
본 프로젝트는 멀티모달 접근을 통해 그 차이를 효과적으로 포착하고자 했다.
