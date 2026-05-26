"""
物理损失模块：CW 方程残差、Δv 一致性约束、平滑正则化。
所有损失函数均支持变长样本（通过 mask 排除 padding 维度）。
"""

import math
import torch
import torch.nn as nn


# ==================== 轨道物理常数 ====================
MU = 398600.0               # 地球引力常数 (km³/s²)
R_ORBIT = 6371.0 + 480.0    # 轨道半径 (km)
N_MEAN = math.sqrt(MU / R_ORBIT**3)  # 平均轨道角速度 ≈ 0.001134 rad/s


def compute_cw_matrix(n, dt, dtype=torch.float32):
    """
    计算 CW 状态转移矩阵 Φ(dt)，形状 (6, 6)。

    适用于 LVLH 坐标系下的近圆轨道相对运动。

    Args:
        n:   平均轨道角速度 (rad/s)
        dt:  时间步长 (s)
        dtype: 输出张量类型

    Returns:
        Phi: (6, 6) 状态转移矩阵
    """
    nt = n * dt
    sin_nt = math.sin(nt)
    cos_nt = math.cos(nt)

    Phi = torch.zeros(6, 6, dtype=dtype)

    # 第一行: x 位置
    Phi[0, 0] = 4.0 - 3.0 * cos_nt
    Phi[0, 3] = sin_nt / n
    Phi[0, 4] = 2.0 * (1.0 - cos_nt) / n

    # 第二行: y 位置
    Phi[1, 0] = 6.0 * (sin_nt - nt)
    Phi[1, 1] = 1.0
    Phi[1, 3] = 2.0 * (cos_nt - 1.0) / n
    Phi[1, 4] = (4.0 * sin_nt - 3.0 * nt) / n

    # 第三行: z 位置
    Phi[2, 2] = cos_nt
    Phi[2, 5] = sin_nt / n

    # 第四行: x 速度
    Phi[3, 0] = 3.0 * n * sin_nt
    Phi[3, 3] = cos_nt
    Phi[3, 4] = 2.0 * sin_nt

    # 第五行: y 速度
    Phi[4, 0] = 6.0 * n * (cos_nt - 1.0)
    Phi[4, 3] = -2.0 * sin_nt
    Phi[4, 4] = 4.0 * cos_nt - 3.0

    # 第六行: z 速度
    Phi[5, 2] = -n * sin_nt
    Phi[5, 5] = cos_nt

    return Phi


def compute_cw_B_eff(Phi):
    """
    从 CW 状态转移矩阵提取 Δv 传播矩阵。

    Δv 对后继状态的影响: x_{t+1} = Φ * (x_t + B·Δv) = Φ·x_t + Φ·B·Δv
    其中 B = [0_{3×3}; I_{3×3}], 即 B·Δv = [0,0,0, Δvx,Δvy,Δvz]^T
    因此 B_eff = Φ·B = Φ 的速度列 (第3,4,5列，0-indexed: 3,4,5)。

    Args:
        Phi: (6, 6) CW 状态转移矩阵

    Returns:
        B_eff: (6, 3) Δv 传播矩阵
    """
    return Phi[:, 3:6].clone()


def estimate_delta_v_from_states(x_curr, x_next, Phi, B_eff):
    """
    根据相邻两步状态和 CW 矩阵反推速度增量 Δv。

    求解: B_eff · Δv = x_next - Φ · x_curr
    使用最小二乘法（每agent独立求解）。

    Args:
        x_curr: (*, 6) 当前状态 [x,y,z, vx,vy,vz]
        x_next: (*, 6) 下一状态
        Phi:    (6, 6) CW 状态转移矩阵
        B_eff:  (6, 3) Δv 传播矩阵

    Returns:
        delta_v: (*, 3) 估计的速度增量
    """
    residual = x_next - (x_curr @ Phi.T)  # (*, 6)
    # 最小二乘: Δv = (B_eff^T · B_eff)^{-1} · B_eff^T · residual
    BtB = B_eff.T @ B_eff  # (3, 3)
    BtB_inv = torch.linalg.inv(BtB)  # (3, 3)
    B_pinv = BtB_inv @ B_eff.T  # (3, 6)
    delta_v = residual @ B_pinv.T  # (*, 3)
    return delta_v


class PhysicsLoss(nn.Module):
    """
    物理信息损失模块。

    提供:
    - cw_residual_loss: CW 方程单步递推残差（需与 Δv 估计配合）
    - delta_v_consistency_loss: 模型 Δv 估计与 CW 逆推 Δv 的一致性
    - delta_v_bound_loss: Δv 幅值边界惩罚 (|Δv| ≤ 3 m/s)
    - smoothness_loss: 轨迹平滑性正则化（最小化 jerk）
    - boundary_loss: 预测状态范围约束
    """

    def __init__(
        self,
        n=N_MEAN,
        dt_h=1.0,
        dt_T=60.0,
        delta_v_limit=3.0,   # m/s → km/s
        device="cpu",
    ):
        super().__init__()
        self.n = n
        self.dt_h = dt_h
        self.dt_T = dt_T
        # Δv 上限: 3 m/s = 0.003 km/s
        self.delta_v_limit = delta_v_limit / 1000.0

        # 预计算 CW 矩阵
        Phi_h = compute_cw_matrix(n, dt_h)
        Phi_T = compute_cw_matrix(n, dt_T)
        B_eff_h = compute_cw_B_eff(Phi_h)
        B_eff_T = compute_cw_B_eff(Phi_T)

        self.register_buffer("Phi_h", Phi_h)        # (6, 6)
        self.register_buffer("Phi_T", Phi_T)        # (6, 6)
        self.register_buffer("B_eff_h", B_eff_h)    # (6, 3)
        self.register_buffer("B_eff_T", B_eff_T)    # (6, 3)

        self.device = device

    def _reshape_for_agents(self, x, mask):
        """
        将 (B, T, max_dim) 张量按有效 agent 重组为 (B, T, max_N, 6)。

        Args:
            x:    (B, T, max_dim)
            mask: (B, max_dim) bool

        Returns:
            x_reshaped: (B, T, max_N, 6)
            n_valid:    (B,) 每个样本的有效特征数
        """
        B, T, D = x.shape
        max_N = D // 6
        x_rs = x.reshape(B, T, max_N, 6)  # (B, T, max_N, 6)
        n_valid = mask.sum(dim=-1) // 6   # (B,) 各样本有效 agent 数
        return x_rs, n_valid

    def cw_residual_loss(self, states, delta_v, mask):
        """
        CW 方程单步递推残差损失。

        对连续步计算: r = x_{t+1} - (Φ_h · x_t + B_eff_h · Δv_t)
        物理含义: 预测状态变化减去 CW 自由演化 + 机动贡献，理想情况应为零。

        Args:
            states:  (B, T, max_dim) 状态序列（输入或预测）
            delta_v: (B, T-1, N*3) 每步的速度增量估计（由模型输出）
            mask:    (B, max_dim) 有效特征掩码

        Returns:
            标量 CW 残差 MSE
        """
        B, T, D = states.shape
        max_N = D // 6

        Ph = self.Phi_h.to(states.device)
        Bh = self.B_eff_h.to(states.device)

        total_loss = 0.0
        count = 0

        for b in range(B):
            n_agents = int(mask[b].sum().item()) // 6
            if n_agents == 0:
                continue

            for a in range(n_agents):
                base = a * 6
                # 提取该 agent 的状态序列
                s = states[b, :, base:base + 6]  # (T, 6)
                dv = delta_v[b, :, a * 3:(a + 1) * 3]  # (T-1, 3)

                # CW 自由演化: s_t → Φ_h · s_t
                s_free = s[:-1] @ Ph.T  # (T-1, 6)
                # 机动贡献: B_eff_h · Δv_t
                dv_effect = dv @ Bh.T  # (T-1, 6)
                # 预测的下一状态: s_{t+1}
                s_next_pred = s_free + dv_effect  # (T-1, 6)
                s_next_true = s[1:]  # (T-1, 6)

                diff = (s_next_pred - s_next_true) ** 2
                total_loss += diff.mean()
                count += 1

        return total_loss / max(count, 1)

    def cw_self_consistency_loss(self, states, delta_v, mask):
        """
        CW 自洽损失（简化版）: 约束状态变化本身与机动贡献的一致性。

        L = MSE( x_{t+1} - Φ_h·x_t , B_eff_h·Δv_t )

        即: 模型声称的 Δv 应该能解释观测到的状态变化（扣除 CW 自由演化后）。

        Args:
            states:  (B, T, max_dim)
            delta_v: (B, T-1, N*3) 模型估计的每步 Δv
            mask:    (B, max_dim)

        Returns:
            标量损失
        """
        B, T, D = states.shape

        Ph = self.Phi_h.to(states.device)
        Bh = self.B_eff_h.to(states.device)

        total_loss = 0.0
        count = 0

        for b in range(B):
            n_agents = int(mask[b].sum().item()) // 6
            if n_agents == 0:
                continue

            for a in range(n_agents):
                base = a * 6
                s = states[b, :, base:base + 6]  # (T, 6)
                dv = delta_v[b, :, a * 3:(a + 1) * 3]  # (T-1, 3)

                # 状态变化（扣除 CW 自由演化）
                s_free = s[:-1] @ Ph.T  # (T-1, 6)
                state_change = s[1:] - s_free  # (T-1, 6)
                # Δv 贡献
                dv_effect = dv @ Bh.T  # (T-1, 6)

                diff = (state_change - dv_effect) ** 2
                total_loss += diff.mean()
                count += 1

        return total_loss / max(count, 1)

    def delta_v_consistency_loss(self, delta_v_model, states, mask):
        """
        Δv 一致性损失: 模型估计的 Δv 与 CW 逆推 Δv 之间的 MSE。

        使用 CW 逆推从相邻状态计算"真实"Δv，与模型估计对比。

        Args:
            delta_v_model: (B, T-1, N*3) 模型估计的 Δv
            states:        (B, T, max_dim) 观测状态序列
            mask:          (B, max_dim)

        Returns:
            标量损失
        """
        B, _, D = states.shape

        Ph = self.Phi_h.to(states.device)
        Bh = self.B_eff_h.to(states.device)

        total_loss = 0.0
        count = 0

        for b in range(B):
            n_agents = int(mask[b].sum().item()) // 6
            if n_agents == 0:
                continue

            for a in range(n_agents):
                base = a * 6
                s = states[b, :, base:base + 6]  # (T, 6)
                dv_model = delta_v_model[b, :, a * 3:(a + 1) * 3]  # (T-1, 3)

                # CW 逆推 Δv
                dv_cw = estimate_delta_v_from_states(
                    s[:-1], s[1:], Ph, Bh
                )  # (T-1, 3)

                diff = (dv_model - dv_cw) ** 2
                total_loss += diff.mean()
                count += 1

        return total_loss / max(count, 1)

    def delta_v_bound_loss(self, delta_v):
        """
        Δv 幅值边界惩罚: 超过 ±3 m/s 的部分施加二次惩罚。

        Args:
            delta_v: (*, 3) 或 (*, N*3) Δv 张量

        Returns:
            标量边界违反惩罚
        """
        limit = self.delta_v_limit
        over = torch.relu(delta_v.abs() - limit)
        return (over ** 2).mean()

    def smoothness_loss(self, states, mask):
        """
        轨迹平滑性损失: 最小化位置的三阶差分 (jerk)。

        Args:
            states: (B, T, max_dim)
            mask:   (B, max_dim)

        Returns:
            标量 jerk 均方值
        """
        B, T, D = states.shape

        total_loss = 0.0
        count = 0

        for b in range(B):
            n_agents = int(mask[b].sum().item()) // 6
            if n_agents == 0:
                continue

            for a in range(n_agents):
                # 仅对位置分量 (前3维) 计算 jerk
                base = a * 6
                pos = states[b, :, base:base + 3]  # (T, 3)
                if T >= 4:
                    jerk = pos[3:] - 3 * pos[2:-1] + 3 * pos[1:-2] - pos[:-3]
                    total_loss += (jerk ** 2).mean()
                    count += 1

        return total_loss / max(count, 1)

    def boundary_loss(self, states, mask, max_distance=200.0):
        """
        边界约束损失: 相对距离不应超过合理范围 (~200 km)。

        Args:
            states:       (B, T, max_dim)
            mask:         (B, max_dim)
            max_distance: 最大允许相对距离 (km)

        Returns:
            标量边界违反惩罚
        """
        B, T, D = states.shape

        total_loss = 0.0
        count = 0

        for b in range(B):
            n_agents = int(mask[b].sum().item()) // 6
            if n_agents == 0:
                continue

            for a in range(n_agents):
                base = a * 6
                pos = states[b, :, base:base + 3]  # (T, 3)
                dist = torch.norm(pos, dim=-1)  # (T,)
                over = torch.relu(dist - max_distance)
                total_loss += (over ** 2).mean()
                count += 1

        return total_loss / max(count, 1)

    def forward(
        self,
        pred_states,
        target_states,
        input_states,
        delta_v_pred,
        mask,
        compute_all=True,
    ):
        """
        计算全部物理损失分量。

        Args:
            pred_states:   (B, 10, max_dim) 预测轨迹
            target_states: (B, 10, max_dim) 真实轨迹
            input_states:  (B, 10, max_dim) 输入序列
            delta_v_pred:  (B, 19, N*3) 模型估计的所有 Δv
                           (前9个来自输入序列, 后10个为预测步)
            mask:          (B, max_dim)
            compute_all:   是否计算所有损失（含平滑和边界）

        Returns:
            dict: {
                "cw_consistency": 标量,
                "dv_consistency": 标量,
                "dv_bound":       标量,
                "smoothness":     标量 (仅 compute_all=True),
                "boundary":       标量 (仅 compute_all=True),
            }
        """
        B = pred_states.shape[0]
        T_in = input_states.shape[1]  # 10
        T_out = pred_states.shape[1]  # 10

        # 拆分 Δv: 前 T_in-1 属于输入序列, 后 T_out 属于预测序列
        dv_input = delta_v_pred[:, :T_in - 1, :]   # (B, 9, N*3)
        dv_output_all = delta_v_pred[:, T_in - 1:, :]  # (B, 10, N*3)
        # dv_output_all[0]: last_input→pred[0], dv_output_all[1:]: pred内部转移
        dv_output = dv_output_all[:, 1:, :]  # (B, 9, N*3) 预测序列内部转移

        losses = {}

        # 1. CW 自洽损失（输入序列）
        losses["cw_consistency_input"] = self.cw_self_consistency_loss(
            input_states, dv_input, mask
        )

        # 2. CW 自洽损失（预测序列）
        losses["cw_consistency_pred"] = self.cw_self_consistency_loss(
            pred_states, dv_output, mask
        )

        # 3. Δv 一致性损失（模型估计 vs CW 逆推）
        losses["dv_consistency"] = self.delta_v_consistency_loss(
            dv_input, input_states, mask
        )

        # 4. Δv 边界惩罚
        losses["dv_bound"] = self.delta_v_bound_loss(delta_v_pred)

        # 5. 平滑性损失
        if compute_all:
            losses["smoothness"] = self.smoothness_loss(pred_states, mask)

        # 6. 边界约束
        if compute_all:
            losses["boundary"] = self.boundary_loss(pred_states, mask)

        return losses


def compute_input_delta_v(states, mask, Phi_h, B_eff_h):
    """
    从输入序列计算 CW 逆推 Δv（用于数据预处理或标签生成）。

    Args:
        states: (B, T, max_dim) 状态序列
        mask:   (B, max_dim)
        Phi_h:  (6, 6) CW 状态转移矩阵
        B_eff_h:(6, 3) Δv 传播矩阵

    Returns:
        delta_v: (B, T-1, max_N*3) CW 逆推的速度增量
    """
    B, T, D = states.shape
    max_N = D // 6
    device = states.device

    delta_v = torch.zeros(B, T - 1, max_N * 3, device=device)

    for b in range(B):
        n_agents = int(mask[b].sum().item()) // 6
        for a in range(n_agents):
            base = a * 6
            s = states[b, :, base:base + 6]  # (T, 6)
            dv = estimate_delta_v_from_states(s[:-1], s[1:], Phi_h, B_eff_h)
            delta_v[b, :, a * 3:(a + 1) * 3] = dv

    return delta_v
