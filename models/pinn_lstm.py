"""
物理信息条件 LSTM 轨迹预测模型 v3 (PI-LSTM v3)。

相比 v5 (PhysicsInformedTrajectoryLSTM) 的重大改进：
1. 加大容量：hidden 256→384, 3层→4层
2. 加 LayerNorm 稳定训练
3. 去除自回归 decoder：一次性输出 10 步（消除误差累积）
4. 位置/速度解耦 head：精细化预测
5. 注意力上下文聚合：替代最后隐状态
6. 保留 PI-LSTM 核心：Δv 估计 + 条件门控 + Δv alignment

架构 (v3):
  输入 (B, 10, 24)
    → Δv 估计模块 (CW 逆推，可微)
    → 条件向量 = mode_embed(1) + MLP(Δv_global)
    → 扩展输入 = state + cond per agent → 56 维
    → Encoder LSTM (4层, hidden=384) + LayerNorm
    → 注意力上下文聚合（最后 3 步平均 + 可学习权重）
    → 位置专用 head: MLP(384) → (B, 10*12)
    → 速度专用 head: MLP(384) → (B, 10*12)
    → delta = concat(pos, vel) (B, 10, 24)
    → 持久预测 + delta (B, 10, 24)
    → Δv 计算 (vel 差分) → 物理一致性
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    MAX_DIM, INPUT_STEPS, OUTPUT_STEPS,
    D_MODEL, DROPOUT,
)
from models.physics_loss import (
    N_MEAN, compute_cw_matrix, compute_cw_B_eff, estimate_delta_v_from_states,
)


# ==================== Δv 估计模块 ====================

class DeltaVEstimator(nn.Module):
    """
    从状态序列中估计速度增量 Δv 的可微模块（全向量化实现）。
    """

    def __init__(self, n=N_MEAN, dt=1.0, use_learnable_correction=False):
        super().__init__()
        self.n = n
        self.dt = dt
        self.use_learnable_correction = use_learnable_correction

        Phi = compute_cw_matrix(n, dt)
        B_eff = compute_cw_B_eff(Phi)
        BtB = B_eff.T @ B_eff
        BtB_inv = torch.linalg.inv(BtB)
        B_pinv = BtB_inv @ B_eff.T  # (3, 6)

        self.register_buffer("Phi", Phi)
        self.register_buffer("B_eff", B_eff)
        self.register_buffer("B_pinv", B_pinv)

        if use_learnable_correction:
            self.correction_net = nn.Sequential(
                nn.Linear(3, 8),
                nn.GELU(),
                nn.Linear(8, 3),
            )
        else:
            self.correction_net = None

    def forward(self, states, mask):
        B, T, D = states.shape
        max_N = D // 6

        states_rs = states.reshape(B, T, max_N, 6)
        x_curr = states_rs[:, :-1, :, :]
        x_next = states_rs[:, 1:, :, :]

        residual = x_next - (x_curr @ self.Phi.T)
        delta_v = residual @ self.B_pinv.T

        if self.correction_net is not None:
            delta_v = delta_v + self.correction_net(delta_v)

        valid = mask.reshape(B, max_N, 6).any(dim=-1)
        valid_3d = valid.unsqueeze(-1).expand(-1, -1, 3)
        delta_v = delta_v * valid_3d.unsqueeze(1)

        delta_v = delta_v.reshape(B, T - 1, max_N * 3)

        return delta_v


class ConditionBuilder(nn.Module):
    """
    构建条件 LSTM 的条件向量 c_t = mode_embed(1) + MLP(Δv_global)
    """

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
        condition = mode_feat + dv_feat
        return condition


# ==================== v3 物理信息条件 LSTM ====================

class PhysicsInformedTrajectoryLSTM(nn.Module):
    """
    v3: 大容量 + LayerNorm + 非自回归 + 位置/速度解耦 head
    """

    def __init__(
        self,
        input_dim=MAX_DIM,
        hidden_size=384,
        num_layers=4,
        dropout=DROPOUT,
        condition_embed_dim=8,
        max_N=4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.condition_embed_dim = condition_embed_dim
        self.max_N = max_N

        self.dv_estimator = DeltaVEstimator(use_learnable_correction=False)
        self.condition_builder = ConditionBuilder(embed_dim=condition_embed_dim)

        lstm_input_dim = max_N * (6 + condition_embed_dim)

        # ── Encoder LSTM ──
        self.encoder_lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        # LayerNorm 稳定训练
        self.encoder_ln = nn.LayerNorm(hidden_size)

        # ── 上下文聚合（最后 3 步加权平均）──
        self.context_query = nn.Parameter(torch.randn(hidden_size) * 0.02)

        # ── 位置专用 head: hidden → (10 * max_N * 3) ──
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, OUTPUT_STEPS * max_N * 3),
        )

        # ── 速度专用 head: hidden → (10 * max_N * 3) ──
        self.vel_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, OUTPUT_STEPS * max_N * 3),
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
            elif "mode_embed" in name:
                continue
            elif "encoder_ln" in name or "pos_head" in name or "vel_head" in name:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
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
        """用可学习 query 注意力聚合 encoder 输出"""
        # encoder_out: (B, T, hidden)
        # query: (hidden,)
        # attention: (B, T) = softmax(encoder_out @ query)
        scores = torch.matmul(encoder_out, self.context_query)  # (B, T)
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        context = (encoder_out * attn).sum(dim=1)  # (B, hidden)
        return context

    def forward(self, x, target=None, teacher_forcing_ratio=0.0, return_dv=True, mask=None, baseline=None):
        """
        Args:
            x: (B, 10, max_dim) 观测序列
            target: (B, 10, max_dim) 目标序列（保留兼容，本版本不使用）
            teacher_forcing_ratio: 保留兼容，本版本不使用
            return_dv: 是否返回 Δv 估计
            mask: (B, max_dim) bool
            baseline: (B, OUTPUT_STEPS, max_dim) 标准化空间的 CW baseline（v7 残差学习模式）
                     若提供：pred = baseline + delta（学残差）
                     若不提供：pred = persistence + delta（v3 默认模式）
        """
        B = x.shape[0]
        device = x.device

        if mask is None:
            mask = torch.ones(B, self.input_dim, dtype=torch.bool, device=device)

        # 1. Δv 估计
        dv_all_input = self.dv_estimator(x, mask)  # (B, 9, max_N*3)

        # 2. 全局 Δv 聚合
        max_N = self.max_N
        dv_rs = dv_all_input.reshape(B, 9, max_N, 3)
        valid_agents = mask.reshape(B, max_N, 6).any(dim=-1).float()
        valid_count = valid_agents.sum(dim=-1).clamp(min=1)
        dv_global = (dv_rs * valid_agents.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
        dv_global = dv_global / valid_count.unsqueeze(-1).unsqueeze(-1)  # (B, 9, 3)

        # 3. 条件向量
        condition_input = self.condition_builder(dv_global)  # (B, 9, embed_dim)
        last_cond = condition_input[:, -1:, :]
        condition_input = torch.cat([condition_input, last_cond], dim=1)  # (B, 10, embed_dim)

        # 4. 扩展输入
        expanded_input = self._build_expanded_input(x, condition_input, mask)  # (B, 10, lstm_input_dim)

        # 5. Encoder LSTM
        encoder_out, (h_n, c_n) = self.encoder_lstm(expanded_input)  # (B, 10, hidden)
        encoder_out = self.encoder_ln(encoder_out)

        # 6. 上下文聚合
        context = self._aggregate_context(encoder_out)  # (B, hidden)

        # 7. 位置 + 速度 head（一次性输出 10 步）
        delta_pos = self.pos_head(context).reshape(B, OUTPUT_STEPS, max_N, 3)
        delta_vel = self.vel_head(context).reshape(B, OUTPUT_STEPS, max_N, 3)
        delta = torch.cat([delta_pos, delta_vel], dim=-1).reshape(B, OUTPUT_STEPS, self.input_dim)

        # 8. 持久预测 + delta（v3 模式：persistence = X_now 末步）
        #    v7 残差学习模式：若 baseline 不为 None，则 anchor = baseline（CW 演化）
        #    此时 model 只学 (y - baseline) 的残差，target 应该是 residual
        #    final_pred = baseline + delta_residual
        if baseline is not None:
            # v7 残差学习模式：anchor = baseline
            pred = baseline + delta
        else:
            # 默认 v3 模式：anchor = persistence
            persistence = x[:, -1:, :].repeat(1, OUTPUT_STEPS, 1)
            pred = persistence + delta

        # 9. Δv 计算（从预测速度差分）
        if return_dv:
            # pred velocity: pred[:, :, 3::6] for each agent
            # 简化为: 取每个 agent 的 velocity (位置基 3:6)
            # 但 24 维是 (x,y,z,vx,vy,vz) × max_N，要按 agent 切片
            pred_vel_full = pred.reshape(B, OUTPUT_STEPS, max_N, 6)[..., 3:6]  # (B, 10, max_N, 3)
            dv_pred = pred_vel_full[:, 1:] - pred_vel_full[:, :-1]  # (B, 9, max_N, 3) - vel 增量
            # 但 dv_pred 与 CW 估计的 Δv 含义不同：CW 估计是 1s 内的脉冲 Δv，vel diff 是 1s 内的速度差
            # 在 CW 模型下：vel_diff = dv_pulse + 自由演化带来的速度变化
            # 这里为对齐 dv_alignment：直接将 dv_pred 与 dv_all_input 比较（仅前 9 步）
            dv_pred_flat = dv_pred.reshape(B, 9, max_N * 3)
            # 用 mask 处理 padding
            valid_3d = valid_agents.unsqueeze(-1).expand(-1, -1, 3).unsqueeze(1)  # (B, 1, max_N, 3)
            dv_pred_flat = dv_pred_flat * valid_3d.reshape(B, 1, max_N * 3)
            dv_all = torch.cat([dv_all_input, dv_pred_flat], dim=1)  # (B, 18, max_N*3)
            return pred, dv_all
        else:
            return pred


def create_pinn_model(device=None, condition_embed_dim=8):
    """创建物理信息条件 LSTM 模型 v3"""
    model = PhysicsInformedTrajectoryLSTM(condition_embed_dim=condition_embed_dim)
    if device is not None:
        model = model.to(device)
    return model


def create_model(device=None):
    """
    工厂函数 —— 根据配置创建模型。
    """
    try:
        from config import PHYSICS_ENABLED, CONDITION_EMBED_DIM
    except ImportError:
        PHYSICS_ENABLED = True
        CONDITION_EMBED_DIM = 8

    if PHYSICS_ENABLED:
        return create_pinn_model(device, CONDITION_EMBED_DIM)
    else:
        from models.model import TrajectoryLSTM
        model = TrajectoryLSTM()
        if device is not None:
            model = model.to(device)
        return model


if __name__ == "__main__":
    from config import DEVICE

    print("=" * 60)
    print("物理信息条件 LSTM v3 模型自检")
    print("=" * 60)

    model = create_pinn_model(DEVICE, condition_embed_dim=8)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {total_params:,}")

    x = torch.randn(4, 10, 24).to(DEVICE)
    mask = torch.ones(4, 24, dtype=torch.bool).to(DEVICE)

    pred, dv_all = model(x, return_dv=True, mask=mask)
    print(f"pred: {pred.shape}, dv_all: {dv_all.shape}")
    print(f"pred mean: {pred.mean().item():.6f}, std: {pred.std().item():.6f}")
    print(f"dv_all 前 9 步 abs.mean: {dv_all[:, :9].abs().mean().item():.6f}")
    print(f"dv_all 后 9 步 abs.mean: {dv_all[:, 9:].abs().mean().item():.6f}")
    print("\n自检通过。")