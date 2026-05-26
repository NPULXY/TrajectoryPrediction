"""
物理信息条件 LSTM 轨迹预测模型 (Physics-Informed Conditional LSTM)。

核心改进:
- 条件门控: 以机动模式嵌入 + Δv 机动特征作为额外条件输入，使 LSTM 自适应调整门控行为
- Δv 估计模块: 利用 CW 方程从状态序列中可微地估计速度增量
- 多输出: 同时输出预测轨迹和 Δv 估计，支持物理损失计算

架构:
  输入 (B, 10, 24)
    → Δv 估计模块 (CW 逆推 + 可学习修正)
    → 条件向量 = 模式嵌入 + MLP(Δv)
    → 扩展输入 = concat(状态, 条件) per agent
    → Encoder LSTM (3层, hidden=256)
    → 上下文向量 + 初始 delta
    → Decoder LSTM (3层) 自回归生成 10 步 delta
    → 输出 = 持久预测 + delta
    → 输出 Δv 估计 (用于物理损失)
"""

import math
import torch
import torch.nn as nn

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
    从状态序列中估计速度增量 Δv 的可微模块。

    方法: CW 逆推（解析解）+ 可选的轻量学习修正。
    由于 CW 逆推已经是可微的（对状态求导），此模块天然支持梯度传播。
    """

    def __init__(self, n=N_MEAN, dt=1.0, use_learnable_correction=False):
        super().__init__()
        self.n = n
        self.dt = dt
        self.use_learnable_correction = use_learnable_correction

        Phi = compute_cw_matrix(n, dt)
        B_eff = compute_cw_B_eff(Phi)
        self.register_buffer("Phi", Phi)          # (6, 6)
        self.register_buffer("B_eff", B_eff)      # (6, 3)

        if use_learnable_correction:
            # 小型修正网络（可选）
            self.correction_net = nn.Sequential(
                nn.Linear(3, 8),
                nn.GELU(),
                nn.Linear(8, 3),
            )
        else:
            self.correction_net = None

    def forward(self, states, mask):
        """
        估计输入序列中每对相邻步之间的 Δv。

        Args:
            states: (B, T, max_dim) 状态序列
            mask:   (B, max_dim) 有效特征掩码

        Returns:
            delta_v: (B, T-1, max_N*3) 每步的速度增量估计
        """
        B, T, D = states.shape
        max_N = D // 6
        device = states.device

        Phi = self.Phi.to(device)
        B_eff = self.B_eff.to(device)

        delta_v = torch.zeros(B, T - 1, max_N * 3, device=device)

        for b in range(B):
            n_agents = int(mask[b].sum().item()) // 6
            if n_agents == 0:
                continue
            for a in range(n_agents):
                base = a * 6
                s = states[b, :, base:base + 6]  # (T, 6)
                dv = estimate_delta_v_from_states(s[:-1], s[1:], Phi, B_eff)  # (T-1, 3)

                if self.correction_net is not None:
                    dv = dv + self.correction_net(dv)

                delta_v[b, :, a * 3:(a + 1) * 3] = dv

        return delta_v


# ==================== 条件向量构建 ====================

class ConditionBuilder(nn.Module):
    """
    构建条件 LSTM 的条件向量 c_t。

    c_t = mode_embed(m_t) + maneuver_mlp(Δv_t)

    其中:
    - mode_embed: 可学习的模式嵌入 (2 × d_embed)，两行对应非机动/机动
    - maneuver_mlp: 将 3D Δv 映射到 d_embed 维
    """

    def __init__(self, embed_dim=8, dv_mean=0.0, dv_std=0.001):
        super().__init__()
        self.embed_dim = embed_dim

        # 模式嵌入: 2 行 (非机动 / 机动)
        self.mode_embed = nn.Embedding(2, embed_dim)
        nn.init.orthogonal_(self.mode_embed.weight)

        # Δv → 机动特征 MLP
        self.dv_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, delta_v_global, mode_labels=None):
        """
        Args:
            delta_v_global: (B, T_cond, 3) 全局 Δv（各 agent 平均）
            mode_labels:    (B, T_cond) 模式标签，None 则默认全为 1（机动）

        Returns:
            condition: (B, T_cond, embed_dim) 条件向量
        """
        B, T, _ = delta_v_global.shape

        if mode_labels is None:
            mode_labels = torch.ones(B, T, dtype=torch.long, device=delta_v_global.device)

        mode_feat = self.mode_embed(mode_labels)        # (B, T, embed_dim)
        dv_feat = self.dv_mlp(delta_v_global)            # (B, T, embed_dim)
        condition = mode_feat + dv_feat                   # (B, T, embed_dim)

        return condition


# ==================== 物理信息条件 LSTM 模型 ====================

class PhysicsInformedTrajectoryLSTM(nn.Module):
    """
    物理信息条件 LSTM 轨迹预测模型。

    继承原有 Encoder-Decoder + 残差连接架构，新增:
    - Δv 估计模块（可微 CW 逆推）
    - 条件向量构建（模式嵌入 + 机动特征）
    - 扩展 LSTM 输入（状态 + 条件 per agent）
    - Δv 输出（用于物理损失计算）
    """

    def __init__(
        self,
        input_dim=MAX_DIM,
        hidden_size=D_MODEL,
        num_layers=3,
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

        # ── Δv 估计模块 ──
        self.dv_estimator = DeltaVEstimator(use_learnable_correction=False)

        # ── 条件向量构建 ──
        self.condition_builder = ConditionBuilder(embed_dim=condition_embed_dim)

        # ── LSTM 输入维度（状态 + 条件 per agent）──
        lstm_input_dim = max_N * (6 + condition_embed_dim)

        # ── Encoder LSTM ──
        self.encoder_lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # ── 上下文向量投影 ──
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )

        # ── Decoder LSTM ──
        # 输入: concat(prev_delta_with_cond, context)
        decoder_input_dim = max_N * (6 + condition_embed_dim) + hidden_size
        self.decoder_lstm = nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # ── 初始 decoder delta ──
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

        # ── Δv 预测头（用于 decoder 步的 Δv 估计）──
        self.dv_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, max_N * 3),
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重，与原始模型一致的自定义初始化。"""
        for name, p in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p)
                elif "bias" in name:
                    p.data.fill_(0)
                    n = p.size(0)
                    p.data[n // 4: n // 2].fill_(1)  # forget gate bias = 1
            elif "mode_embed" in name:
                continue  # 已在 ConditionBuilder 中初始化
            elif "correction_net" in name:
                continue  # 使用默认初始化
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.zeros_(p)

    def _build_expanded_input(self, states, condition, mask):
        """
        将原始状态与条件向量拼接，形成扩展的 LSTM 输入。

        Args:
            states:    (B, T, max_dim) 原始状态
            condition: (B, T, embed_dim) 条件向量
            mask:      (B, max_dim) 有效特征掩码

        Returns:
            expanded: (B, T, max_N * (6 + embed_dim))
        """
        B, T, _ = states.shape
        max_N = self.max_N
        embed_dim = self.condition_embed_dim

        states_rs = states.reshape(B, T, max_N, 6)  # (B, T, max_N, 6)

        # 检查每个 agent 是否有效
        valid_agents = mask.reshape(B, max_N, 6).any(dim=-1)  # (B, max_N)

        # 扩展条件: (B, T, embed_dim) → (B, T, max_N, embed_dim)
        cond_expanded = condition.unsqueeze(2).expand(-1, -1, max_N, -1)

        # 将无效 agent 的条件置零
        cond_expanded = cond_expanded * valid_agents.unsqueeze(1).unsqueeze(-1).float()

        # 拼接: (B, T, max_N, 6+embed_dim)
        expanded = torch.cat([states_rs, cond_expanded], dim=-1)

        # 展平: (B, T, max_N * (6+embed_dim))
        expanded = expanded.reshape(B, T, -1)

        return expanded

    def _get_condition_for_decoder(self, condition_input, mask):
        """
        为 Decoder 获取条件向量。使用 Encoder 最后一刻的条件。

        Args:
            condition_input: (B, T_in, embed_dim) Encoder 的条件序列
            mask:            (B, max_dim)

        Returns:
            condition_dec: (B, embed_dim) Decoder 的条件向量
        """
        # 取最后一个有效条件
        return condition_input[:, -1, :]  # (B, embed_dim)

    def forward(self, x, target=None, teacher_forcing_ratio=0.0, return_dv=True, mask=None):
        """
        Args:
            x:      (B, 10, max_dim) 观测序列
            target: (B, 10, max_dim) 目标序列（teacher forcing 时需要）
            teacher_forcing_ratio: teacher forcing 概率
            return_dv: 是否返回 Δv 估计（训练时需要）
            mask:   (B, max_dim) bool 有效特征掩码，用于正确处理 padding

        Returns:
            若 return_dv=True: (pred, dv_all) 其中 pred (B,10,max_dim), dv_all (B,19,max_N*3)
            若 return_dv=False: pred (B,10,max_dim)
        """
        B = x.shape[0]
        device = x.device

        # ── 构造 mask（若未提供则默认全有效）──
        if mask is None:
            mask = torch.ones(B, self.input_dim, dtype=torch.bool, device=device)

        # ── 步骤 1: 从输入序列估计 Δv ──
        dv_all_input = self.dv_estimator(x, mask)  # (B, 9, max_N*3)

        # 计算全局 Δv（各 agent 平均，处理 padding）
        max_N = self.max_N
        dv_rs = dv_all_input.reshape(B, 9, max_N, 3)  # (B, 9, max_N, 3)
        valid_agents = mask.reshape(B, max_N, 6).any(dim=-1).float()  # (B, max_N)
        valid_count = valid_agents.sum(dim=-1).clamp(min=1)  # (B,)
        dv_global = (dv_rs * valid_agents.unsqueeze(1).unsqueeze(-1)).sum(dim=2)  # (B, 9, 3)
        dv_global = dv_global / valid_count.unsqueeze(-1).unsqueeze(-1)  # (B, 9, 3)

        # ── 步骤 2: 构建条件向量（每个输入步）──
        condition_input = self.condition_builder(dv_global)  # (B, 9, embed_dim)

        # 为第10个输入步补充条件（使用第9个Δv的条件）
        last_cond = condition_input[:, -1:, :]  # (B, 1, embed_dim)
        condition_input = torch.cat([condition_input, last_cond], dim=1)  # (B, 10, embed_dim)

        # ── 步骤 3: 扩展 LSTM 输入 ──
        expanded_input = self._build_expanded_input(x, condition_input, mask)  # (B, 10, lstm_input_dim)

        # ── 步骤 4: Encoder ──
        encoder_out, (h_n, c_n) = self.encoder_lstm(expanded_input)

        # 上下文向量
        context = self.context_proj(h_n[-1])  # (B, hidden)

        # ── 步骤 5: Decoder 初始化 ──
        persistence = x[:, -1:, :].repeat(1, OUTPUT_STEPS, 1)  # (B, 10, 24)
        first_delta = self.init_delta(h_n[-1]).unsqueeze(1)  # (B, 1, 24)

        # Decoder 条件（使用最后一个输入步的条件）
        condition_dec = self._get_condition_for_decoder(condition_input, mask)  # (B, embed_dim)
        condition_dec_expanded = condition_dec.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, embed_dim)

        hidden = (h_n, c_n)

        # ── 步骤 6: 自回归生成 ──
        deltas = []
        dv_preds = []
        prev_delta_flat = first_delta.squeeze(1)  # (B, 24)

        for t in range(OUTPUT_STEPS):
            # 将 delta 扩展到包含条件
            prev_delta_rs = prev_delta_flat.reshape(B, max_N, 6)  # (B, max_N, 6)
            cond_rs = condition_dec_expanded.expand(-1, -1, max_N, -1).squeeze(1)  # (B, max_N, embed_dim)
            delta_with_cond = torch.cat([prev_delta_rs, cond_rs], dim=-1)  # (B, max_N, 6+embed_dim)
            delta_with_cond_flat = delta_with_cond.reshape(B, -1)  # (B, max_N*(6+embed_dim))

            # Decoder 输入: concat(delta_with_cond, context)
            dec_input = torch.cat([
                delta_with_cond_flat.unsqueeze(1),
                context.unsqueeze(1).expand(-1, 1, -1)
            ], dim=-1)  # (B, 1, decoder_input_dim)

            out, hidden = self.decoder_lstm(dec_input, hidden)
            delta = self.output_proj(out)  # (B, 1, input_dim)
            deltas.append(delta)

            # 预测该步的 Δv
            if return_dv:
                dv_pred = self.dv_predictor(out)  # (B, 1, max_N*3)
                dv_preds.append(dv_pred)

            # 下一时间步的输入
            use_tf = self.training and torch.rand(1).item() < teacher_forcing_ratio
            if use_tf and target is not None:
                prev_delta_flat = (target[:, t:t+1, :] - persistence[:, t:t+1, :]).detach().squeeze(1)
            else:
                prev_delta_flat = delta.detach().squeeze(1)

        deltas = torch.cat(deltas, dim=1)  # (B, 10, input_dim)
        pred = persistence + deltas

        if return_dv:
            dv_preds = torch.cat(dv_preds, dim=1)   # (B, 10, max_N*3)
            # 拼接输入 Δv 和预测 Δv: (B, 9+10, max_N*3) = (B, 19, max_N*3)
            dv_all = torch.cat([dv_all_input, dv_preds], dim=1)
            return pred, dv_all
        else:
            return pred


# ==================== 工厂函数 ====================

def create_pinn_model(device=None, condition_embed_dim=8):
    """创建物理信息条件 LSTM 模型。"""
    model = PhysicsInformedTrajectoryLSTM(condition_embed_dim=condition_embed_dim)
    if device is not None:
        model = model.to(device)
    return model


def create_model(device=None):
    """
    工厂函数 —— 根据配置创建模型。

    向后兼容: 当 PHYSICS_ENABLED=False 时返回原始模型；
    否则返回物理信息条件 LSTM。
    """
    try:
        from config import PHYSICS_ENABLED, CONDITION_EMBED_DIM
    except ImportError:
        PHYSICS_ENABLED = True
        CONDITION_EMBED_DIM = 8

    if PHYSICS_ENABLED:
        return create_pinn_model(device, CONDITION_EMBED_DIM)
    else:
        # 回退到原始模型
        from models.model import TrajectoryLSTM
        model = TrajectoryLSTM()
        if device is not None:
            model = model.to(device)
        return model


if __name__ == "__main__":
    from config import DEVICE

    print("=" * 60)
    print("物理信息条件 LSTM 模型自检")
    print("=" * 60)

    model = create_pinn_model(DEVICE, condition_embed_dim=8)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {total_params:,}")

    # 测试前向传播
    x = torch.randn(4, 10, 24).to(DEVICE)
    y_target = torch.randn(4, 10, 24).to(DEVICE)

    # 训练模式（含 Δv 输出）
    pred_train, dv_train = model(x, y_target, teacher_forcing_ratio=0.5)
    print(f"训练模式 - pred: {pred_train.shape}, dv: {dv_train.shape}")

    # 推理模式（不含 Δv）
    pred_eval = model(x, return_dv=False)
    print(f"推理模式 - pred: {pred_eval.shape}")

    # 验证 Δv 范围
    print(f"Δv 范围: [{dv_train.min().item():.6f}, {dv_train.max().item():.6f}] km/s")
    print(f"Δv 范围 (m/s): [{dv_train.min().item()*1000:.2f}, {dv_train.max().item()*1000:.2f}]")

    print("\n自检通过。")
