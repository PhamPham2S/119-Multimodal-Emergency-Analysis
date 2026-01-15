import os
import glob
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from tqdm.auto import tqdm

# ==========================================
# 1. 환경 설정
# ==========================================
from google.colab import drive
drive.mount('/content/drive')

CONFIG = {
    'base_dir': '/content/drive/MyDrive',
    'extract_root': './dataset_unified',           # 압축 풀 경로
    'save_root': '/content/drive/MyDrive/Final_Embeddings_781', # 결과 저장 경로
    'zips': ['TS_서울_구급.zip', 'TS_서울_구조.zip', 'TS_서울_기타.zip', 'TS_서울_화재.zip'],
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

os.makedirs(CONFIG['save_root'], exist_ok=True)
print(f"Using Device: {CONFIG['device']}")

# ==========================================
# 2. 데이터 준비
# ==========================================
def prepare_data():
    all_data = []
    print(">>> 데이터 준비 중...")
    for zip_name in CONFIG['zips']:
        category = zip_name.replace('.zip', '')
        zip_path = os.path.join(CONFIG['base_dir'], zip_name)
        extract_path = os.path.join(CONFIG['extract_root'], category)
        
        # 압축 해제
        if not os.path.exists(extract_path):
            if os.path.exists(zip_path):
                print(f"압축 해제: {zip_name}")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(extract_path)
            else:
                print(f"[Skip] 파일 없음: {zip_path}")
                continue
        
        # 파일 리스트 확보
        files = glob.glob(os.path.join(extract_path, "**/*.wav"), recursive=True)
        if not files: files = glob.glob(os.path.join(extract_path, "**/*.mp3"), recursive=True)
        
        print(f"  - {category}: {len(files)}개")
        for f in files:
            all_data.append({'path': f, 'filename': os.path.splitext(os.path.basename(f))[0]})
            
    return pd.DataFrame(all_data)

df = prepare_data()
if len(df) == 0: raise RuntimeError("데이터가 없습니다.")

# ==========================================
# 3. 데이터셋 (13개 피처 추출 구현)
# ==========================================
class UnifiedAudioDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.df = dataframe
        self.processor = processor
        self.sr = 16000

    def __len__(self):
        return len(self.df)

    def extract_evidence_features(self, y):
        # [핵심] 근거 기반 13개 피처 추출
        try:
            # 1. Prosodic (3)
            f0, _, voiced_probs = librosa.pyin(y, fmin=60, fmax=1000, frame_length=2048)
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) > 0:
                f0_mean, f0_std = np.mean(f0_clean), np.std(f0_clean)
            else:
                f0_mean, f0_std = 0.0, 0.0
            voiced_prob = np.mean(voiced_probs)
            
            # 2. Timbre (5)
            S = np.abs(librosa.stft(y))
            cent = np.mean(librosa.feature.spectral_centroid(S=S, sr=self.sr))
            bw = np.mean(librosa.feature.spectral_bandwidth(S=S, sr=self.sr))
            contrast = np.mean(librosa.feature.spectral_contrast(S=S, sr=self.sr))
            flat = np.mean(librosa.feature.spectral_flatness(S=S))
            rolloff = np.mean(librosa.feature.spectral_rolloff(S=S, sr=self.sr))
            
            # 3. Energy (2)
            rms = np.mean(librosa.feature.rms(y=y))
            onset = np.mean(librosa.onset.onset_strength(y=y, sr=self.sr))
            
            # 4. Texture (3)
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)
            zcr_std = np.std(zcr)
            tonality = 1.0 - flat # HNR 대용
            
            features = np.array([
                f0_mean, f0_std, voiced_prob,
                cent, bw, contrast, flat, rolloff,
                rms, onset,
                zcr_mean, zcr_std, tonality
            ])
        except:
            features = np.zeros(13)
            
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(features, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            speech, orig_sr = torchaudio.load(row['path'])
            if orig_sr != self.sr:
                speech = torchaudio.transforms.Resample(orig_sr, self.sr)(speech)
            if speech.shape[0] > 1: speech = speech.mean(dim=0, keepdim=True)
            speech = speech.squeeze().numpy()
        except:
            speech = np.zeros(16000, dtype=np.float32)

        if len(speech) < 1600 or np.isnan(speech).any():
            speech = np.random.normal(0, 0.001, 16000).astype(np.float32)

        # 전체 길이 처리 (Padding=True, Truncation=False)
        inputs = self.processor(speech, sampling_rate=self.sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.squeeze(0)
        
        # 마스크 생성
        if 'attention_mask' in inputs:
            attention_mask = inputs.attention_mask.squeeze(0)
        else:
            attention_mask = torch.ones_like(input_values, dtype=torch.long)

        return {
            'input_values': input_values,
            'attention_mask': attention_mask,
            'handcrafted': self.extract_evidence_features(speech),
            'filename': row['filename']
        }

# ==========================================
# 4. 단일 모델 (768 + 13 = 781차원)
# ==========================================
class UnifiedEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone (768)
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec.feature_extractor._freeze_parameters()
        
        # Normalization Layers (매우 중요)
        self.w2v_norm = nn.LayerNorm(768)  # Wav2Vec용
        self.hc_norm = nn.LayerNorm(13)    # Handcrafted용

    def forward(self, input_values, attention_mask, handcrafted):
        # 1. Wav2Vec Extraction
        outputs = self.wav2vec(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # 2. Mask Resizing & Pooling (전체 길이 대응)
        target_len = hidden_states.size(1)
        mask_float = attention_mask.unsqueeze(1).float()
        resized_mask = F.interpolate(mask_float, size=target_len, mode='nearest').squeeze(1)
        expanded_mask = resized_mask.unsqueeze(-1).expand(hidden_states.size())
        
        sum_hidden = torch.sum(hidden_states * expanded_mask, dim=1)
        sum_mask = torch.sum(expanded_mask, dim=1)
        w2v_emb = sum_hidden / torch.clamp(sum_mask, min=1e-9)
        
        # 3. Wav2Vec Scaling
        w2v_scaled = self.w2v_norm(w2v_emb)
        
        # 4. Handcrafted Scaling (Log + LayerNorm)
        # 물리량(Hz 등)을 Log로 눌러주고, LayerNorm으로 분포 통일
        hc_log = torch.log1p(handcrafted)
        hc_scaled = self.hc_norm(hc_log)
        
        # 5. Concat (768 + 13 = 781)
        final_vec = torch.cat((w2v_scaled, hc_scaled), dim=1)
        
        # 6. Final L2 Normalize (방향성 통일)
        return F.normalize(final_vec, p=2, dim=1)

# ==========================================
# 5. 실행 (Batch=1)
# ==========================================
def run_extraction():
    print(f"\n>>> [통합 모델] 임베딩 추출 시작 (781차원)...")
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    dataset = UnifiedAudioDataset(df, processor)
    # Batch Size 1: 전체 길이를 다 넣기 위해 필수
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = UnifiedEmbeddingModel().to(CONFIG['device'])
    model.eval()
    
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = batch['input_values'].to(CONFIG['device'])
            mask = batch['attention_mask'].to(CONFIG['device'])
            hc = batch['handcrafted'].to(CONFIG['device'])
            fname = batch['filename'][0]
            
            # 추출
            embedding = model(inputs, mask, hc) # [1, 781]
            
            # 저장
            np.save(os.path.join(CONFIG['save_root'], f"{fname}.npy"), embedding.cpu().numpy()[0])
            count += 1
            
    print(f">>> 작업 완료: 총 {count}개 저장됨.")
    print(f">>> 저장 경로: {CONFIG['save_root']}")
    print(f">>> 최종 차원: 781 (768 Wav2Vec + 13 Evidence)")

if __name__ == "__main__":
    run_extraction()