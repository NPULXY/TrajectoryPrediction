"""
轨迹预测模型：LSTM Encoder-Decoder + 残差连接。
Encoder 将 10 步输入编码为隐状态，Decoder 自回归生成 10 步修正量（delta），
最终预测 = 持久预测 + delta，保证至少等于朴素基线。
"""

import torch
import torch.nn as nn

from config import (
    MAX_DIM, INPUT_STEPS, OUTPUT_STEPS,
    D_MODEL, DROPOUT,
)


class TrajectoryLSTM(nn.Module):
    """
    LSTM Encoder-Decoder 轨迹预测模型。

    输入:  (B, 10, max_dim) 观测序列
    输出:  (B, 10, max_dim) 预测序列
    """

    def __init__(
        self,
        input_dim=MAX_DIM,
        hidden_size=D_MODEL,
        num_layers=3,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ── Encoder LSTM ──
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # ── Decoder LSTM ──
        self.decoder_lstm = nn.LSTM(
            input_size=input_dim + hidden_size,  # concat(prev_delta, context)
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # ── 上下文向量投影（从 encoder 最终状态生成） ──
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )

        # ── 初始 decoder 输入：从 encoder 最后隐状态预测第一步 delta ──
        self.init_delta = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_dim),
        )

        # ── Decoder 输出 → delta ──
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p)
                elif "bias" in name:
                    p.data.fill_(0)
                    n = p.size(0)
                    p.data[n // 4: n // 2].fill_(1)  # forget gate
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.zeros_(p)

    def forward(self, x, target=None, teacher_forcing_ratio=0.0):
        """
        Args:
            x:      (B, 10, max_dim) 观测序列
            target: (B, 10, max_dim) 目标序列（仅 teacher forcing 时需要）
            teacher_forcing_ratio: teacher forcing 概率

        Returns:
            (B, 10, max_dim) 未来预测序列
        """
        B = x.shape[0]
        device = x.device

        # ── Encoder: 编码输入序列 ──
        encoder_out, (h_n, c_n) = self.encoder_lstm(x)  # h_n: (layers, B, hidden)

        # 上下文向量：encoder 最终隐状态
        context = self.context_proj(h_n[-1])  # (B, hidden)

        # ── 持久预测（基线）──
        persistence = x[:, -1:, :].repeat(1, OUTPUT_STEPS, 1)  # (B, 10, 24)

        # ── 初始 delta ──
        first_delta = self.init_delta(h_n[-1]).unsqueeze(1)  # (B, 1, 24)

        # Decoder 初始隐状态
        hidden = (h_n, c_n)

        # ── 自回归生成 ──
        deltas = []
        prev_delta = first_delta

        for t in range(OUTPUT_STEPS):
            # Decoder 输入: concat(prev_delta, context)
            dec_input = torch.cat([
                prev_delta,
                context.unsqueeze(1).expand(-1, prev_delta.shape[1], -1)
            ], dim=-1)  # (B, 1, input_dim + hidden)

            out, hidden = self.decoder_lstm(dec_input, hidden)
            delta = self.output_proj(out)  # (B, 1, input_dim)
            deltas.append(delta)

            # 下一时间步的输入
            use_tf = self.training and torch.rand(1).item() < teacher_forcing_ratio
            if use_tf and target is not None:
                prev_delta = (target[:, t:t+1, :] - persistence[:, t:t+1, :]).detach()
            else:
                prev_delta = delta.detach()

        deltas = torch.cat(deltas, dim=1)  # (B, 10, input_dim)

        return persistence + deltas


def create_model(device=None):
    """工厂函数：根据配置创建模型并移至指定设备。

    当 PHYSICS_ENABLED=True 时返回物理信息条件 LSTM，
    否则返回标准 TrajectoryLSTM（向后兼容）。
    """
    try:
        from config import PHYSICS_ENABLED, CONDITION_EMBED_DIM
    except ImportError:
        PHYSICS_ENABLED = False
        CONDITION_EMBED_DIM = 8

    if PHYSICS_ENABLED:
        from models.pinn_lstm import PhysicsInformedTrajectoryLSTM
        model = PhysicsInformedTrajectoryLSTM(condition_embed_dim=CONDITION_EMBED_DIM)
    else:
        model = TrajectoryLSTM()

    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    from config import DEVICE
    model = create_model(DEVICE)
    x = torch.randn(4, 10, 24).to(DEVICE)
    y_target = torch.randn(4, 10, 24).to(DEVICE)
    y_train = model(x, y_target, teacher_forcing_ratio=0.5)
    y_eval = model(x)
    print(f"输入:  {x.shape}")
    print(f"训练模式: {y_train.shape}")
    print(f"推理模式: {y_eval.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {total_params:,}")
