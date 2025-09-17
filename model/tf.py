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
        **kwargs
    ):
        super().__init__()
        self.seq_len = seq_len

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
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        # ✅ seq_len=32 강제 체크
        assert x.dim() == 3, f"expected 3D tensor (B, L, C), got {x.shape}"
        assert x.size(1) == self.seq_len, f"seq_len must be {self.seq_len}, got {x.size(1)}"

        x = self.input_proj(x)                 # (B, L, H)
        L = x.size(1)

        pos = self.pos_embedding[:, :L, :]     # (1, L, H)
        x = x + pos

        h = self.transformer(x)                # (B, L, H)
        h_last = h[:, -1, :]                   # (B, H)
        return self.fc(h_last)                 # (B, output_size)


# MODEL_REGISTRY에 등록
MODEL_REGISTRY = {
    "tf": TransformerModel,
}
