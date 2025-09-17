import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional, output_size, **kwargs):
        """
        input_size: 입력 feature 차원 (예: 변동률이면 1)
        hidden_size: LSTM hidden 차원
        num_layers: LSTM layer 수
        dropout: 드롭아웃 비율
        bidirectional: 양방향 여부
        output_size: 출력 차원 (예: 이후 5분 변동률이면 5)
        kwargs: config에 있는 나머지 값들 무시하기 위해 받음
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (hn, _) = self.lstm(x)
        hn = hn[-1]  # 마지막 레이어 hidden state
        out = self.fc(hn)  # (batch, output_size)
        return out



# -------------------
# 모델 레지스트리 등록
# -------------------
MODEL_REGISTRY = {
    "lstm": LSTMModel,
}