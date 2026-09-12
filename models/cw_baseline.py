"""
CW 演化 baseline（v3 修复版，标准化空间）。

修复要点：
- 接受 scaler.std，在 __init__ 时按维度缩放 Phi 矩阵
- 标准化空间 CW 矩阵：Phi_norm[i, j] = Phi_phys[i, j] * std[j] / std[i]
- 这样 Phi_norm 作用于 (pos/std, vel/std) 输出 (pos/std, vel/std)
- 不再有 8000 km 爆炸
"""
import math
import torch
import torch.nn as nn

from config import CW_N


def compute_cw_matrix_phys(n: float, dt: float, dtype=torch.float32):
    """CW 状态转移矩阵（原始物理空间）"""
    nt = n * dt
    sin_nt = math.sin(nt)
    cos_nt = math.cos(nt)
    Phi = torch.zeros(6, 6, dtype=dtype)
    Phi[0, 0] = 4.0 - 3.0 * cos_nt
    Phi[0, 3] = sin_nt / n
    Phi[0, 4] = 2.0 * (1.0 - cos_nt) / n
    Phi[1, 0] = 6.0 * (sin_nt - nt)
    Phi[1, 1] = 1.0
    Phi[1, 3] = 2.0 * (cos_nt - 1.0) / n
    Phi[1, 4] = (4.0 * sin_nt - 3.0 * nt) / n
    Phi[2, 2] = cos_nt
    Phi[2, 5] = sin_nt / n
    Phi[3, 0] = 3.0 * n * sin_nt
    Phi[3, 3] = cos_nt
    Phi[3, 4] = 2.0 * sin_nt
    Phi[4, 0] = 6.0 * n * (cos_nt - 1.0)
    Phi[4, 3] = -2.0 * sin_nt
    Phi[4, 4] = 4.0 * cos_nt - 3.0
    Phi[5, 2] = -n * sin_nt
    Phi[5, 5] = cos_nt
    return Phi


def normalize_phi(Phi_phys: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    将物理空间 CW 矩阵转换为标准化空间 CW 矩阵。
    Phi_norm[i, j] = Phi_phys[i, j] * std[j] / std[i]
    """
    D_inv = 1.0 / std
    return D_inv.unsqueeze(1) * Phi_phys * std.unsqueeze(0)


class CWBaseline(nn.Module):
    """
    CW 演化 baseline（标准化空间）。
    输入 x_init_norm: (B, max_dim) - 标准化空间
    输出 baseline_norm: (B, num_steps, max_dim) - 标准化空间

    使用方法：
        scaler = FeatureScaler(); scaler.load('output/scaler.pkl')
        cw = CWBaseline(scaler=scaler).to(DEVICE)
        baseline_norm = cw(x_init_norm, mask)  # 标准化的 baseline
    """

    def __init__(self, n=CW_N, dt=60.0, num_steps=10, max_N=4, scaler=None):
        super().__init__()
        self.n = n
        self.dt = dt
        self.num_steps = num_steps
        self.max_N = max_N

        Phi_phys = compute_cw_matrix_phys(n, dt)

        if scaler is not None and scaler.std is not None:
            std = scaler.std
            pos_std = float(std[:3].mean())
            vel_std = float(std[3:6].mean())
            std_vec = torch.tensor(
                [pos_std]*3 + [vel_std]*3,
                dtype=torch.float32
            )
            Phi_norm = normalize_phi(Phi_phys, std_vec)
            self.register_buffer("std_vec", std_vec)
            self.register_buffer("pos_std", torch.tensor([pos_std]))
            self.register_buffer("vel_std", torch.tensor([vel_std]))
        else:
            Phi_norm = Phi_phys
            self.std_vec = None
            self.pos_std = None
            self.vel_std = None

        # 预计算 Phi^t
        powers = [torch.eye(6, dtype=Phi_norm.dtype)]
        for _ in range(num_steps):
            powers.append(powers[-1] @ Phi_norm)
        self.register_buffer("Phi_powers", torch.stack(powers, dim=0))

    def forward(self, x_init, mask):
        """
        Args:
            x_init: (B, max_dim) - 标准化空间
            mask: (B, max_dim) bool
        Returns:
            baseline: (B, num_steps, max_dim) - 标准化空间
        """
        B, D = x_init.shape
        max_N_local = D // 6
        x_init_rs = x_init.reshape(B, max_N_local, 6)

        baseline = []
        current = x_init_rs
        for t in range(self.num_steps):
            current = current @ self.Phi_powers[t + 1].T
            baseline.append(current)
        baseline = torch.stack(baseline, dim=1)
        baseline = baseline.reshape(B, self.num_steps, D)
        baseline = baseline * mask.unsqueeze(1).float()
        return baseline


# 单元测试
if __name__ == "__main__":
    import numpy as np
    from utils.data_loader import FeatureScaler, parse_csv

    # 加载 scaler
    scaler = FeatureScaler(); scaler.load('output/scaler.pkl')
    print(f'std[:3] (pos) = {scaler.std[:3]}, mean={scaler.std[:3].mean():.4f}')
    print(f'std[3:6] (vel) = {scaler.std[3:6]}, mean={scaler.std[3:6].mean():.6f}')

    cw = CWBaseline(n=0.001134, dt=60.0, num_steps=10, max_N=4, scaler=scaler)
    print(f'Phi_norm[0,3] = {cw.Phi_powers[1, 0, 3]:.4f}（标准化空间）')
    print(f'Phi^10_norm[0,3] = {cw.Phi_powers[10, 0, 3]:.4f}')

    # 验证 baseline 物理合理性
    XN, XN_masks = parse_csv(f'{config.DATA_DIR}/X_now.csv')
    XN2, _ = parse_csv(f'{config.DATA_DIR}/X_next.csv')
    x_init_raw = XN[0][-1].astype(np.float32)  # (24,)
    x_init_norm = scaler.transform(x_init_raw.reshape(1, 1, 24))[0, -1]  # (24,)
    mask = torch.from_numpy(XN_masks[0].reshape(1, 24)).bool()
    x_init_t = torch.from_numpy(x_init_norm.reshape(1, 24)).float()
    baseline_norm = cw(x_init_t, mask)
    baseline_raw = scaler.inverse_transform(baseline_norm.detach().numpy())
    print()
    print(f'=== 样本 0 真实 X_next 末步 vs CW baseline ===')
    for a in range(int(XN_masks[0].sum())//6):
        base = a*6
        print(f'  Agent {a+1}: baseline末步 = {baseline_raw[0, -1, base:base+3]}')
        print(f'           真值 = {XN2[0][-1][base:base+3]}')
        print(f'           delta = {XN2[0][-1][base:base+3] - baseline_raw[0, -1, base:base+3]}')