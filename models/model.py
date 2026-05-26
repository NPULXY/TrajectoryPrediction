"""
Transformer 时序预测模型。
支持 Encoder-Only（直接回归）和 Encoder-Decoder 两种模式。
"""

import torch
import torch.nn as nn
import math

from config import (
    MAX_DIM, INPUT_STEPS, OUTPUT_STEPS,
    D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
    DIM_FEEDFORWARD, DROPOUT, USE_DECODER
)


class PositionalEncoding(nn.Module):
    """正弦位置编码（用于无学习参数时的备选方案）。"""

    def __init__(self, d_model, max_len=20, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """x: (B, T, D)"""
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class TrajectoryTransformer(nn.Module):
    """
    轨迹预测 Transformer 模型。

    输入:  (batch, 10, max_dim) 观测序列
    输出:  (batch, 10, max_dim) 预测序列
    """

    def __init__(
        self,
        input_dim=MAX_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        use_decoder=USE_DECODER,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_decoder = use_decoder

        # 输入投影：将 max_dim 维特征映射到 d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # 可学习位置编码（针对 10 个输入时间步）
        self.src_pos_encoding = nn.Parameter(
            torch.randn(1, INPUT_STEPS, d_model) * 0.02
        )
        if use_decoder:
            self.tgt_pos_encoding = nn.Parameter(
                torch.randn(1, OUTPUT_STEPS, d_model) * 0.02
            )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer Decoder（可选）
        if use_decoder:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, input_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.zeros_(p)

    def forward(self, x):
        """
        Args:
            x: (B, 10, max_dim) 观测序列

        Returns:
            (B, 10, max_dim) 未来预测序列
        """
        B = x.shape[0]

        # 投影到 d_model 并加位置编码
        src = self.input_proj(x) + self.src_pos_encoding       # (B, 10, d_model)

        # Encoder
        memory = self.encoder(src)                              # (B, 10, d_model)

        if self.use_decoder:
            # 使用可学习的输出查询 + Decoder 交叉注意力
            tgt = torch.zeros(B, OUTPUT_STEPS, self.d_model, device=x.device)
            tgt = tgt + self.tgt_pos_encoding
            output = self.decoder(tgt, memory)                  # (B, 10, d_model)
        else:
            # 直接从 encoder 输出回归
            output = memory

        return self.output_head(output)                         # (B, 10, max_dim)


def create_model(device=None):
    """工厂函数：根据配置创建模型并移至指定设备。"""
    model = TrajectoryTransformer()
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    # 快速自检
    from config import DEVICE
    model = create_model(DEVICE)
    x = torch.randn(4, 10, 24).to(DEVICE)
    y = model(x)
    print(f"输入:  {x.shape}")
    print(f"输出:  {y.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {total_params:,} (可训练: {trainable_params:,})")
