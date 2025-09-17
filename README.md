**## 1. Config 작성법 예시

모델 학습/예측에 필요한 모든 설정은 JSON 파일(`config/*.json`)로 관리합니다.

예시: `config/lstm_config.json`
```json
{
  "model_name": "lstm",
  "input_size": 1,
  "hidden_size": 64,
  "num_layers": 2,
  "dropout": 0.2,
  "bidirectional": false,
  "output_size": 5,

  "loss": "MSELoss",
  "optimizer": "Adam",
  "learning_rate": 0.001,
  "batch_size": 32,
  "epochs": 20,

  "preprocessed_data_path": "data/lstm_dataset.pt"
}

model_name: 사용할 모델 이름 (MODEL_REGISTRY에서 매핑)

input_size: 입력 feature 차원 (변동률 → 1)

output_size: 예측할 미래 길이 (5분 → 5)

loss: 손실 함수 (MSELoss, L1Loss 등)

optimizer: 최적화 알고리즘 (Adam, AdamW, SGD 등)

preprocessed_data_path: 학습용 데이터셋 경로 (torch.save((X,y))로 저장된 파일)



## 2. 파이프라인 실행법 예시

모델 학습 
python train.py --config config/ 원하는 모델의 config.json
ex)
python train.py --config config/lstm_config.json

ml flow
mlflow ui --port 5000
접속시 시각화된 학습 곡선 확인 가능



## 3. 모델 학습 후 백테스팅 모델 불러오기
from model import MODEL_REGISTRY
import torch

config = {...}
ModelClass = MODEL_REGISTRY[config["model_name"]]

model = ModelClass(**config)
model.load_state_dict(torch.load(f"model/{config['model_name']}_final.pth", map_location="cpu"))
model.eval()


========== event detect ==========
# 원하는 전략 클라스
from event.strategy.ema_cross import EmaCrossStrategy
from event.interface import DetectEvent

# event 찾는 전략. 이 클래스만 교체하면 된다
strategy = EmaCrossStrategy()
# interface *이 코드는 고정*
detector = DetectEvent(strategy)

# 이 위치에서 raw data 가져오기
raw = detector.load_raw_data("data/raw_data/minute_data.npy")
# raw data 안에서 event 탐지
events = detector.find_events(raw)
# 이 위치에 이 이름으로 저장
detector.save_events(events, "data/precessed_data/events.npy")



## 백테스트 실행법

python backtest.py --config config/backtest_config.json
