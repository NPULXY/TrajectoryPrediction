"""
v7 PI-LSTM 变体：不同容量模型（用于集成学习多样性）

- v7a: hidden=256, 3 层（小模型，预测更平滑）
- v7b: hidden=320, 4 层（中等模型）
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    MAX_DIM, INPUT_STEPS, OUTPUT_STEPS, DROPOUT, CW_DT_H,
)
from models.physics_loss import (
    N_MEAN, compute_cw_matrix, compute_cw_B_eff,
)


class DeltaVEstimator(nn.Module):
    def __init__(self, n=N_MEAN, dt=CW_DT_H):   # 2026-09-10: 步长由 1.0 修正为实测值 60.0
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
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, delta_v_global, mode_labels=None):
        B, T, _ = delta_v_global.shape
        if mode_labels is None:
            mode_labels = torch.ones(B, T, dtype=torch.long, device=delta_v_global.device)
        mode_feat = self.mode_embed(mode_labels)
        dv_feat = self.dv_mlp(delta_v_global)
        return mode_feat + dv_feat


class PI_LSTM_V7(nn.Module):
    """
    v7 变体：可配置 hidden / 层数，用于集成学习多样性
    """

    def __init__(self, input_dim=MAX_DIM, hidden_size=256, num_layers=3, dropout=DROPOUT,
                 condition_embed_dim=8, max_N=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.condition_embed_dim = condition_embed_dim
        self.max_N = max_N

        self.dv_estimator = DeltaVEstimator()
        self.condition_builder = ConditionBuilder(embed_dim=condition_embed_dim)

        lstm_input_dim = max_N * (6 + condition_embed_dim)

        self.encoder_lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.encoder_ln = nn.LayerNorm(hidden_size)

        self.context_query = nn.Parameter(torch.randn(hidden_size) * 0.02)

        # 位置/速度解耦 head
        head_dim = hidden_size
        self.pos_head = nn.Sequential(
            nn.Linear(head_dim, head_dim), nn.GELU(), nn.LayerNorm(head_dim), nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim), nn.GELU(), nn.LayerNorm(head_dim), nn.Dropout(dropout),
            nn.Linear(head_dim, OUTPUT_STEPS * max_N * 3),
        )
        self.vel_head = nn.Sequential(
            nn.Linear(head_dim, head_dim), nn.GELU(), nn.LayerNorm(head_dim), nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim), nn.GELU(), nn.LayerNorm(head_dim), nn.Dropout(dropout),
            nn.Linear(head_dim, OUTPUT_STEPS * max_N * 3),
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
                    p.data[n // 4: n // 2].fill_(1)
            elif "mode_embed" in name or "context_query" in name:
                continue
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_expanded_input(self, states, condition, mask):
        B, T, _ = states.shape
        max_N = self.max_N
        embed_dim = self.condition_embed_dim
        states_rs = states.reshape(B, T, max_N, 6)
        valid_agents = mask.reshape(B, max_N, 6).any(dim=-1)
        cond_expanded = condition.unsqueeze(2).expand(-1, -1, max_N, -1)
        cond_expanded = cond_expanded * valid_agents.unsqueeze(1).unsqueeze(-1).float()
        expanded = torch.cat([states_rs, cond_expanded], dim=-1)
        expanded = expanded.reshape(B, T, -1)
        return expanded

    def _aggregate_context(self, encoder_out):
        scores = torch.matmul(encoder_out, self.context_query)
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (encoder_out * attn).sum(dim=1)

    def forward(self, x, target=None, teacher_forcing_ratio=0.0, return_dv=True, mask=None):
        B = x.shape[0]
        device = x.device
        if mask is None:
            mask = torch.ones(B, self.input_dim, dtype=torch.bool, device=device)

        dv_all_input = self.dv_estimator(x, mask)
        max_N = self.max_N
        dv_rs = dv_all_input.reshape(B, 9, max_N, 3)
        valid_agents = mask.reshape(B, max_N, 6).any(dim=-1).float()
        valid_count = valid_agents.sum(dim=-1).clamp(min=1)
        dv_global = (dv_rs * valid_agents.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
        dv_global = dv_global / valid_count.unsqueeze(-1).unsqueeze(-1)

        condition_input = self.condition_builder(dv_global)
        last_cond = condition_input[:, -1:, :]
        condition_input = torch.cat([condition_input, last_cond], dim=1)

        expanded_input = self._build_expanded_input(x, condition_input, mask)
        encoder_out, _ = self.encoder_lstm(expanded_input)
        encoder_out = self.encoder_ln(encoder_out)
        context = self._aggregate_context(encoder_out)

        delta_pos = self.pos_head(context).reshape(B, OUTPUT_STEPS, max_N, 3)
        delta_vel = self.vel_head(context).reshape(B, OUTPUT_STEPS, max_N, 3)
        delta = torch.cat([delta_pos, delta_vel], dim=-1).reshape(B, OUTPUT_STEPS, self.input_dim)
        persistence = x[:, -1:, :].repeat(1, OUTPUT_STEPS, 1)
        pred = persistence + delta

        if return_dv:
            pred_vel_full = pred.reshape(B, OUTPUT_STEPS, max_N, 6)[..., 3:6]
            dv_pred = pred_vel_full[:, 1:] - pred_vel_full[:, :-1]
            valid_3d = valid_agents.unsqueeze(-1).expand(-1, -1, 3).unsqueeze(1)
            dv_pred_flat = dv_pred.reshape(B, 9, max_N * 3)
            dv_pred_flat = dv_pred_flat * valid_3d.reshape(B, 1, max_N * 3)
            dv_all = torch.cat([dv_all_input, dv_pred_flat], dim=1)
            return pred, dv_all
        return pred


def create_pinn_lstm_v7(hidden_size=256, num_layers=3, device=None):
    model = PI_LSTM_V7(hidden_size=hidden_size, num_layers=num_layers)
    if device is not None:
        model = model.to(device)
    return model