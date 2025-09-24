# model/tf.py
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_heads,
        dropout,
        output_size,
        seq_len,             
        max_seq_len,
        pred_len,
        **kwargs
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_size = output_size


        # 입력 feature → embedding
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Positional Embedding (hidden_size와 동일 차원)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_size))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 출력 projection
        self.fc = nn.Linear(hidden_size, pred_len * output_size)

        self.attn_pool = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embedding[:, :x.size(1), :]
        h = self.transformer(x)     # (B, L, H)

        # 각 시점마다 중요도 점수 구하기
        attn_score = torch.softmax(self.attn_pool(h), dim=1)  # (B, L, 1)

        # 중요도를 가중합해서 context 벡터 생성
        context = torch.sum(h * attn_score, dim=1)            # (B, H)

        out = self.fc(context)                                # (B, pred_len*output_size)
        out = out.view(x.size(0), self.pred_len, self.output_size)
        return out


# MODEL_REGISTRY에 등록
MODEL_REGISTRY = {
    "tft": TransformerModel,
}
