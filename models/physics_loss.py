"""
物理损失模块：CW 方程残差、Δv 边界约束、Δv alignment。
所有计算在分维度归一化空间（消除位置/速度量纲不一致）+ 原始物理量纲（边界惩罚）。

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
    物理信息损失模块（分维度归一化空间 + 原始物理量纲混合）。

    关键改进（针对 v6 的根本性修复）：
    1. CW 单步残差：按位置/速度分别除以对应 std，做维度归一化
       —— 消除位置（km）与速度（km/s）量纲不一致带来的优化方向扭曲
    2. 新增 dv_alignment_loss：直接对齐 model 预测的 Δv 与 CW 逆推的 Δv
       —— 实现 paper 描述的 L_mode 核心约束
    3. Δv 边界软约束：保留在原始量纲，符合物理直觉

    所有运算通过 reshape 为 (B, T, max_N, 6) 批量完成，消除 Python 循环。
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
        self.scaler = scaler
        self.delta_v_limit = delta_v_limit / 1000.0  # m/s → km/s
        self.device = device

        Phi_h = compute_cw_matrix(n, dt_h)
        self.register_buffer("Phi_h", Phi_h)  # (6, 6)

        # 将 scaler 参数缓存为 tensor，避免每次 _to_physical 都走 CPU
        if scaler is not None and scaler.mean is not None:
            self.register_buffer("_mean", torch.from_numpy(scaler.mean.astype('float32')))
            self.register_buffer("_std", torch.from_numpy(scaler.std.astype('float32')))
            # 分维度归一化系数：前 3 维是位置（km），后 3 维是速度（km/s）
            # 让位置残差除以 std_pos（典型 15-60 km），速度残差除以 std_vel（典型 0.015-0.055 km/s）
            # 残差归一化后与标准化空间的 MSE 量级相当
            self.register_buffer("_norm_pos", torch.from_numpy(scaler.std[:3].astype('float32').mean().reshape(1, 1)))  # 平均位置 std
            self.register_buffer("_norm_vel", torch.from_numpy(scaler.std[3:6].astype('float32').mean().reshape(1, 1)))  # 平均速度 std
        else:
            self._mean = None
            self._std = None
            self._norm_pos = None
            self._norm_vel = None

    def _to_physical(self, x_norm, mask):
        """将标准化张量转为原始物理量纲，纯 GPU 运算。"""
        if self._mean is None or self._std is None:
            return x_norm
        mean = self._mean.to(x_norm.device).unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        std = self._std.to(x_norm.device).unsqueeze(0).unsqueeze(0)    # (1, 1, D)
        eps = 1e-8
        x_phys = x_norm * (std + eps) + mean
        # 将 padding 部分的物理值置零，避免无效数据影响损失
        x_phys = x_phys * mask.unsqueeze(1).float()
        return x_phys

    def _cw_residual_normalized(self, states, mask):
        """
        CW 单步递推残差（分维度归一化）。

        残差计算：r = x_{t+1} - Φ_h · x_t（原始物理空间）
        分维度归一化：r_pos / std_pos, r_vel / std_vel
        这样残差各维度量级一致，loss 值与预测损失可比。

        Args:
            states: (B, T, max_dim) 原始物理量纲状态
            mask:   (B, max_dim) bool
        """
        B, T, D = states.shape
        max_N = D // 6
        Phi = self.Phi_h.to(states.device)

        if self._norm_pos is None or self._norm_vel is None:
            return self._cw_residual_legacy(states, mask)

        norm_pos = self._norm_pos.to(states.device)
        norm_vel = self._norm_vel.to(states.device)

        # (B, T, max_N, 6)
        s = states.reshape(B, T, max_N, 6)
        s_free = s[:, :-1] @ Phi.T  # (B, T-1, max_N, 6)
        residual = s[:, 1:] - s_free  # (B, T-1, max_N, 6)

        # 分维度归一化：位置 0-2 用 norm_pos，速度 3-5 用 norm_vel
        residual_norm = residual.clone()
        residual_norm[..., :3] = residual_norm[..., :3] / norm_pos
        residual_norm[..., 3:] = residual_norm[..., 3:] / norm_vel

        # 每个 agent 的 MSE（对时间步和状态维平均）
        agent_mse = residual_norm.pow(2).mean(dim=(1, 3))  # (B, max_N)

        # 仅有效 agent 参与平均
        valid = mask.reshape(B, max_N, 6).any(dim=-1).float()  # (B, max_N)
        loss = (agent_mse * valid).sum() / valid.sum().clamp(min=1)
        return loss

    def _cw_residual_legacy(self, states, mask):
        """
        原始量纲空间的 CW 单步残差（兼容无 scaler 场景）。
        """
        B, T, D = states.shape
        max_N = D // 6
        Phi = self.Phi_h.to(states.device)

        s = states.reshape(B, T, max_N, 6)
        s_free = s[:, :-1] @ Phi.T
        residual = s[:, 1:] - s_free

        agent_mse = residual.pow(2).mean(dim=(1, 3))
        valid = mask.reshape(B, max_N, 6).any(dim=-1).float()
        loss = (agent_mse * valid).sum() / valid.sum().clamp(min=1)
        return loss

    def _velocity_change_loss(self, states, mask):
        """
        Δv 边界软约束（原始量纲 km/s）。
        当 ‖Δv‖ > 3/1000 km/s 时惩罚。
        """
        B, T, D = states.shape
        max_N = D // 6
        limit = self.delta_v_limit

        s = states.reshape(B, T, max_N, 6)
        vel = s[..., 3:6]  # (B, T, max_N, 3)
        dv = vel[:, 1:] - vel[:, :-1]  # (B, T-1, max_N, 3)
        dv_abs = torch.norm(dv, dim=-1)  # (B, T-1, max_N)
        over = torch.relu(dv_abs - limit)  # (B, T-1, max_N)

        agent_penalty = over.pow(2).mean(dim=1)  # (B, max_N)
        valid = mask.reshape(B, max_N, 6).any(dim=-1).float()
        loss = (agent_penalty * valid).sum() / valid.sum().clamp(min=1)
        return loss

    def dv_alignment_loss(self, dv_model, dv_cw, mask):
        """
        Δv alignment loss：让模型学到的 Δv 与 CW 逆推的 Δv 对齐（PI-LSTM 核心约束）。

        Args:
            dv_model: (B, T, max_N*3) 模型预测的 Δv（km/s）
            dv_cw:    (B, T_cw, max_N*3) CW 逆推的 Δv（km/s）
            mask:     (B, max_dim) bool

        Returns:
            对齐损失标量
        """
        B, Tm, D = dv_model.shape
        Tc = dv_cw.shape[1]
        T = min(Tm, Tc)
        max_N = D // 3

        if T == 0:
            return torch.tensor(0.0, device=dv_model.device)

        # 取前 T 步对齐
        m = dv_model[:, :T].reshape(B, T, max_N, 3)
        c = dv_cw[:, :T].reshape(B, T, max_N, 3)

        # MSE 损失
        diff = (m - c).pow(2)  # (B, T, max_N, 3)
        agent_mse = diff.mean(dim=(1, 3))  # (B, max_N)

        # 仅有效 agent 参与平均
        valid = mask.reshape(B, max_N, 6).any(dim=-1).float()
        loss = (agent_mse * valid).sum() / valid.sum().clamp(min=1)
        return loss

    def forward(self, pred_states, target_states, input_states, mask,
                compute_all=True, dv_model=None, dv_cw=None):
        """
        计算物理损失（分维度归一化空间）。

        Args:
            pred_states:   (B, 10, max_dim) 预测轨迹 (标准化)
            target_states: (B, 10, max_dim) 真实轨迹 (标准化)
            input_states:  (B, 10, max_dim) 输入序列 (标准化)
            mask:          (B, max_dim) bool
            compute_all:   是否计算所有子损失（用于 warmup）
            dv_model:      (B, 19, max_N*3) 模型预测的 Δv 序列（可选）
            dv_cw:         (B, 19, max_N*3) CW 逆推的 Δv 序列（可选）

        Returns:
            dict: {
                "cw_input", "cw_pred",   # CW 残差（分维度归一化）
                "dv_change",              # Δv 边界软约束（原始量纲）
                "dv_align"                # Δv alignment（原始量纲）
            }
        """
        # 转换到物理空间（GPU 向量化）
        pred_phys = self._to_physical(pred_states, mask)
        target_phys = self._to_physical(target_states, mask)
        input_phys = self._to_physical(input_states, mask)

        losses = {}
        losses["cw_input"] = self._cw_residual_normalized(input_phys, mask)
        losses["cw_pred"] = self._cw_residual_normalized(pred_phys, mask)
        losses["dv_change"] = self._velocity_change_loss(pred_phys, mask)

        if compute_all:
            # Δv alignment：模型 Δv 与 CW Δv 对齐
            if dv_model is not None and dv_cw is not None:
                losses["dv_align"] = self.dv_alignment_loss(dv_model, dv_cw, mask)
            else:
                losses["dv_align"] = torch.tensor(0.0, device=pred_states.device)

        return losses