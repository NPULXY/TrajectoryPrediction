"""
v5 Transformer-PI-LSTM 轨迹预测模型（完全算法变更）。

核心创新（vs v3 PI-LSTM）：
1. **Transformer encoder** 替代 LSTM（自注意力，长距离依赖）
2. **多尺度监督**：每步位置 + 末步位置 + 速度一致性
3. **outlier-aware 损失**：Huber-like 软裁剪，对极端样本鲁棒
4. **位置/速度解耦 head**（沿用 v3 设计）
5. **保留 PI-LSTM 物理一致性**：Δv 估计 + 条件门控

架构：
  输入 (B, 10, 24) + mask
    → 输入投影 Linear(24 → d_model=256) + pos_embed
    → Δv 估计 (CW 逆推, 可微)
    → 条件嵌入 (mode_embed + Δv MLP)
    → 条件注入 (gate) 调节 hidden
    → Transformer Encoder (4 层, 8 heads, d_ff=512, dropout=0.1)
    → Multi-head attention pooling
    → 位置 head: (B, 10, max_N*3) + 速度 head: (B, 10, max_N*3)
    → 持久预测 + delta
    → Δv 估计 (vel 差分) → 物理一致性
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    MAX_DIM, INPUT_STEPS, OUTPUT_STEPS, DROPOUT,
)
from models.physics_loss import (
    N_MEAN, compute_cw_matrix, compute_cw_B_eff,
)


# ==================== Δv 估计模块（与 v3 同）====================

class DeltaVEstimator(nn.Module):
    def __init__(self, n=N_MEAN, dt=1.0):
        super().__init__()
        self.n = n
        self.dt = dt
        Phi = compute_cw_matrix(n, dt)
        B_eff = compute_cw_B_eff(Phi)
        BtB = B_eff.T @ B_eff
        BtB_inv = torch.linalg.inv(BtB)
        B_pinv = BtB_inv @ B_eff.T
        self.register_buffer("Phi", Phi)
        self.register_buffer("B_eff", B_eff)
        self.register_buffer("B_pinv", B_pinv)

    def forward(self, states, mask):
        B, T, D = states.shape
        max_N = D // 6
        states_rs = states.reshape(B, T, max_N, 6)
        x_curr = states_rs[:, :-1, :, :]
        x_next = states_rs[:, 1:, :, :]
        residual = x_next - (x_curr @ self.Phi.T)
        delta_v = residual @ self.B_pinv.T
        valid = mask.reshape(B, max_N, 6).any(dim=-1)
        valid_3d = valid.unsqueeze(-1).expand(-1, -1, 3)
        delta_v = delta_v * valid_3d.unsqueeze(1)
        delta_v = delta_v.reshape(B, T - 1, max_N * 3)
        return delta_v


class ConditionBuilder(nn.Module):
    def __init__(self, embed_dim=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.mode_embed = nn.Embedding(2, embed_dim)
        nn.init.orthogonal_(self.mode_embed.weight)
        self.dv_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(4, embed_dim) if False else nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, delta_v_global, mode_labels=None):
        B, T, _ = delta_v_global.shape
        if mode_labels is None:
            mode_labels = torch.ones(B, T, dtype=torch.long, device=delta_v_global.device)
        mode_feat = self.mode_embed(mode_labels)
        dv_feat = self.dv_mlp(delta_v_global)
        return mode_feat + dv_feat


# ==================== Transformer-PI 模型 ====================

class TransformerPI(nn.Module):
    def __init__(
        self,
        input_dim=MAX_DIM,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        dropout=DROPOUT,
        condition_embed_dim=8,
        max_N=4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.condition_embed_dim = condition_embed_dim
        self.max_N = max_N

        self.dv_estimator = DeltaVEstimator()
        self.condition_builder = ConditionBuilder(embed_dim=condition_embed_dim)

        # 输入投影 (24 维 → d_model=256)
        self.input_proj = nn.Linear(input_dim, d_model)

        # 位置嵌入 (10 步)
        self.pos_embed = nn.Parameter(torch.randn(1, INPUT_STEPS, d_model) * 0.02)

        # 条件融合 gate
        lstm_input_dim = max_N * (6 + condition_embed_dim)
        self.cond_proj = nn.Linear(condition_embed_dim, d_model)
        self.cond_gate = nn.Sequential(
            nn.Linear(condition_embed_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Final LayerNorm
        self.final_ln = nn.LayerNorm(d_model)

        # 多头注意力 pooling（query=1个，与 encoder 输出做 attention）
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 位置专用 head
        self.pos_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_steps_in := OUTPUT_STEPS * max_N * 3),
        )

        # 速度专用 head
        self.vel_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, OUTPUT_STEPS * max_N * 3),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "transformer_encoder" in name:
                continue  # PyTorch default initialization is fine
            if "mode_embed" in name:
                continue
            if "pos_embed" in name or "pool_query" in name:
                continue  # already small init
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_input_with_condition(self, x, condition_input, mask):
        """构造 transformer 输入：状态投影 + 条件门控"""
        B, T, _ = x.shape
        # 状态投影
        state_emb = self.input_proj(x)  # (B, T, d_model)
        # 加位置嵌入
        state_emb = state_emb + self.pos_embed[:, :T]
        # 条件门控
        cond_per_step = self.cond_proj(condition_input)  # (B, T, d_model)
        gate = self.cond_gate(condition_input)  # (B, T, d_model)
        state_emb = state_emb * gate + cond_per_step  # 加门控的融合

        # padding mask (True 表示 padding)
        # mask: (B, max_dim) bool, True 表示有效
        # 转换为 transformer 期望的格式 (True 表示 padding)
        # 对每个样本，如果有任何特征 padding（mask 中有 False），整个时间步 padding
        # 用 mask.any(dim=-1) 聚合
        key_padding_mask = ~mask.any(dim=-1)  # (B,)
        # 扩展到所有时间步
        key_padding_mask = key_padding_mask.unsqueeze(1).expand(-1, T)  # (B, T)

        return state_emb, key_padding_mask

    def forward(self, x, target=None, teacher_forcing_ratio=0.0, return_dv=True, mask=None):
        B = x.shape[0]
        device = x.device

        if mask is None:
            mask = torch.ones(B, self.input_dim, dtype=torch.bool, device=device)

        # 1. Δv 估计 + 全局聚合
        dv_all_input = self.dv_estimator(x, mask)  # (B, 9, max_N*3)
        max_N = self.max_N
        dv_rs = dv_all_input.reshape(B, 9, max_N, 3)
        valid_agents = mask.reshape(B, max_N, 6).any(dim=-1).float()
        valid_count = valid_agents.sum(dim=-1).clamp(min=1)
        dv_global = (dv_rs * valid_agents.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
        dv_global = dv_global / valid_count.unsqueeze(-1).unsqueeze(-1)

        # 2. 条件向量
        condition_input = self.condition_builder(dv_global)  # (B, 9, embed_dim)
        last_cond = condition_input[:, -1:, :]
        condition_input = torch.cat([condition_input, last_cond], dim=1)  # (B, 10, embed_dim)

        # 3. Transformer 输入
        emb, key_padding_mask = self._build_input_with_condition(x, condition_input, mask)
        # emb: (B, 10, d_model), key_padding_mask: (B, 10)

        # 4. Transformer Encoder
        encoder_out = self.transformer_encoder(emb, src_key_padding_mask=key_padding_mask)
        encoder_out = self.final_ln(encoder_out)  # (B, 10, d_model)

        # 5. 注意力 pooling
        # pool_query: (1, 1, d_model) -> (B, 1, d_model)
        q = self.pool_query.expand(B, -1, -1)
        # 使用 cross-attention: (B, 1, d_model) <- (B, 10, d_model)
        attn_scores = torch.matmul(q, encoder_out.transpose(-1, -2)) / (self.d_model ** 0.5)
        # padding mask (B, 10)
        attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1), -1e9)
        attn = F.softmax(attn_scores, dim=-1)  # (B, 1, 10)
        context = torch.matmul(attn, encoder_out).squeeze(1)  # (B, d_model)

        # 6. 位置 + 速度 head
        delta_pos = self.pos_head(context).reshape(B, OUTPUT_STEPS, max_N, 3)
        delta_vel = self.vel_head(context).reshape(B, OUTPUT_STEPS, max_N, 3)
        delta = torch.cat([delta_pos, delta_vel], dim=-1).reshape(B, OUTPUT_STEPS, self.input_dim)

        # 7. 持久预测 + delta
        persistence = x[:, -1:, :].repeat(1, OUTPUT_STEPS, 1)
        pred = persistence + delta

        # 8. Δv 计算
        if return_dv:
            pred_vel_full = pred.reshape(B, OUTPUT_STEPS, max_N, 6)[..., 3:6]
            dv_pred = pred_vel_full[:, 1:] - pred_vel_full[:, :-1]  # (B, 9, max_N, 3)
            valid_3d = valid_agents.unsqueeze(-1).expand(-1, -1, 3).unsqueeze(1)
            dv_pred_flat = dv_pred.reshape(B, 9, max_N * 3)
            dv_pred_flat = dv_pred_flat * valid_3d.reshape(B, 1, max_N * 3)
            dv_all = torch.cat([dv_all_input, dv_pred_flat], dim=1)  # (B, 18, max_N*3)
            return pred, dv_all
        else:
            return pred


def create_transformer_pi(device=None, condition_embed_dim=8):
    model = TransformerPI(condition_embed_dim=condition_embed_dim)
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    from config import DEVICE
    print("=" * 60)
    print("v5 Transformer-PI 模型自检")
    print("=" * 60)
    model = create_transformer_pi(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {total_params:,}")
    x = torch.randn(4, 10, 24).to(DEVICE)
    mask = torch.ones(4, 24, dtype=torch.bool).to(DEVICE)
    pred, dv_all = model(x, return_dv=True, mask=mask)
    print(f"pred: {pred.shape}, dv_all: {dv_all.shape}")
    print(f"pred mean: {pred.mean().item():.6f}, std: {pred.std().item():.6f}")
    print("\n自检通过。")