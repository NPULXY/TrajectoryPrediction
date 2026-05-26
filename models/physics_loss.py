"""
物理损失模块：CW 方程残差、Δv 边界约束、平滑正则化。
所有计算均在原始物理量纲（km, km/s）下进行。

使用方式:
    physics_fn = PhysicsLoss(scaler=scaler, delta_v_limit=3.0, device=device)
    losses = physics_fn(pred_norm, target_norm, x_norm, mask)
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
    """
    nt = n * dt
    sin_nt = math.sin(nt)
    cos_nt = math.cos(nt)

    Phi = torch.zeros(6, 6, dtype=dtype)

    # Row 0: x position
    Phi[0, 0] = 4.0 - 3.0 * cos_nt
    Phi[0, 3] = sin_nt / n
    Phi[0, 4] = 2.0 * (1.0 - cos_nt) / n

    # Row 1: y position
    Phi[1, 0] = 6.0 * (sin_nt - nt)
    Phi[1, 1] = 1.0
    Phi[1, 3] = 2.0 * (cos_nt - 1.0) / n
    Phi[1, 4] = (4.0 * sin_nt - 3.0 * nt) / n

    # Row 2: z position
    Phi[2, 2] = cos_nt
    Phi[2, 5] = sin_nt / n

    # Row 3: x velocity
    Phi[3, 0] = 3.0 * n * sin_nt
    Phi[3, 3] = cos_nt
    Phi[3, 4] = 2.0 * sin_nt

    # Row 4: y velocity
    Phi[4, 0] = 6.0 * n * (cos_nt - 1.0)
    Phi[4, 3] = -2.0 * sin_nt
    Phi[4, 4] = 4.0 * cos_nt - 3.0

    # Row 5: z velocity
    Phi[5, 2] = -n * sin_nt
    Phi[5, 5] = cos_nt

    return Phi


def compute_cw_B_eff(Phi):
    """
    从 CW 状态转移矩阵提取 Δv 传播矩阵 B_eff = Φ 的速度列 (3,4,5)。
    """
    return Phi[:, 3:6].clone()


def estimate_delta_v_from_states(x_curr, x_next, Phi, B_eff):
    """
    根据相邻两步状态和 CW 矩阵反推速度增量 Δv（最小二乘解）。

    求解: B_eff · Δv = x_next - Φ · x_curr

    Args:
        x_curr: (*, 6) 当前状态 [x,y,z, vx,vy,vz]
        x_next: (*, 6) 下一状态
        Phi:    (6, 6) CW 状态转移矩阵
        B_eff:  (6, 3) Δv 传播矩阵

    Returns:
        delta_v: (*, 3) 估计的速度增量
    """
    residual = x_next - (x_curr @ Phi.T)  # (*, 6)
    BtB = B_eff.T @ B_eff  # (3, 3)
    BtB_inv = torch.linalg.inv(BtB)
    B_pinv = BtB_inv @ B_eff.T  # (3, 6)
    delta_v = residual @ B_pinv.T  # (*, 3)
    return delta_v


class PhysicsLoss(nn.Module):
    """
    物理信息损失模块（在原始物理量纲下计算）。

    提供:
    - cw_propagation_loss: CW 自由演化残差 |x_{t+1} - Φ_h·x_t|²
    - dv_bound_loss: 速度变化幅值边界惩罚
    - smoothness_loss: 轨迹平滑性正则化
    """

    def __init__(
        self,
        scaler=None,
        n=N_MEAN,
        dt_h=1.0,
        delta_v_limit=3.0,   # m/s
        device="cpu",
    ):
        super().__init__()
        self.n = n
        self.dt_h = dt_h
        self.scaler = scaler  # FeatureScaler, 用于反标准化
        self.delta_v_limit = delta_v_limit / 1000.0  # m/s → km/s

        Phi_h = compute_cw_matrix(n, dt_h)
        self.register_buffer("Phi_h", Phi_h)  # (6, 6)
        self.device = device

    def _to_physical(self, x_norm):
        """将标准化张量转为原始物理量纲。"""
        if self.scaler is None:
            return x_norm
        # x_norm: (B, T, D) torch tensor
        x_np = x_norm.detach().cpu().numpy()
        x_phys = self.scaler.inverse_transform(x_np)
        return torch.from_numpy(x_phys).to(x_norm.device).float()

    def cw_propagation_loss(self, states, mask):
        """
        CW 自由演化残差: MSE(x_{t+1} - Φ_h·x_t)。

        物理含义: 相邻状态应近似满足 CW 动力学。
        在机动存在时该残差反映机动贡献。
        """
        B, T, D = states.shape
        Ph = self.Phi_h.to(states.device)

        total_loss = 0.0
        count = 0

        for b in range(B):
            n_agents = int(mask[b].sum().item()) // 6
            if n_agents == 0:
                continue

            for a in range(n_agents):
                base = a * 6
                s = states[b, :, base:base + 6]  # (T, 6)
                s_free = s[:-1] @ Ph.T  # (T-1, 6)
                residual = s[1:] - s_free  # (T-1, 6)
                total_loss += (residual ** 2).mean()
                count += 1

        return total_loss / max(count, 1)

    def velocity_change_loss(self, states, mask):
        """
        速度变化约束: 惩罚过大的相邻步速度变化。

        Δv ≈ v_{t+1} - v_t (对于 h=1s 的小时间步)，应 ≤ 3 m/s。
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
                vel = states[b, :, base + 3:base + 6]  # (T, 3)
                dv = vel[1:] - vel[:-1]  # (T-1, 3)
                # 超出限制的部分施加惩罚
                dv_abs = torch.norm(dv, dim=-1)  # (T-1,)
                over = torch.relu(dv_abs - self.delta_v_limit)
                total_loss += (over ** 2).mean()
                count += 1

        return total_loss / max(count, 1)

    def smoothness_loss(self, states, mask):
        """位置 jerk 最小化。"""
        B, T, D = states.shape
        if T < 4:
            return torch.tensor(0.0, device=states.device)

        total_loss = 0.0
        count = 0

        for b in range(B):
            n_agents = int(mask[b].sum().item()) // 6
            if n_agents == 0:
                continue

            for a in range(n_agents):
                base = a * 6
                pos = states[b, :, base:base + 3]  # (T, 3)
                jerk = pos[3:] - 3 * pos[2:-1] + 3 * pos[1:-2] - pos[:-3]
                total_loss += (jerk ** 2).mean()
                count += 1

        return total_loss / max(count, 1)

    def forward(self, pred_states, target_states, input_states, mask,
                compute_all=True):
        """
        计算物理损失（在原始物理量纲下）。

        Args:
            pred_states:   (B, 10, max_dim) 预测轨迹 (标准化)
            target_states: (B, 10, max_dim) 真实轨迹 (标准化)
            input_states:  (B, 10, max_dim) 输入序列 (标准化)
            mask:          (B, max_dim) bool

        Returns:
            dict: {"cw_input", "cw_pred", "dv_change", "smoothness"}
        """
        # 转换到物理空间
        pred_phys = self._to_physical(pred_states)
        target_phys = self._to_physical(target_states)
        input_phys = self._to_physical(input_states)

        losses = {}

        # 1. CW 自由演化残差（输入序列）
        losses["cw_input"] = self.cw_propagation_loss(input_phys, mask)

        # 2. CW 自由演化残差（预测序列）
        losses["cw_pred"] = self.cw_propagation_loss(pred_phys, mask)

        # 3. 速度变化约束
        losses["dv_change"] = self.velocity_change_loss(pred_phys, mask)

        # 4. 平滑性（可选）
        if compute_all:
            losses["smoothness"] = self.smoothness_loss(pred_phys, mask)

        return losses
