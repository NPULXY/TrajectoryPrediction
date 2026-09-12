# -*- coding: utf-8 -*-
"""
时间步长核验（只读）。

动机：视觉分析 best_predictions/top01 时发现 X_now 末步与 X_next 首步之间存在
20–30 km 的跳变，而 1 s 内追踪星位移应仅约 0.05 km。这暗示数据实际步长远大于
文档所述"h = 1 s"。同时 cw_residual 呈双峰分布（~1.8 与 ~5.5），也提示存在
两种不同的时间间隔。

本脚本用 CW 状态转移矩阵反推：在不同 dt 假设下，X_now / X_next 序列内部
相邻步的 CW 残差哪个最小，从而判定真实步长。

用法：python _tools/verify_timestep.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.physics_loss import compute_cw_matrix, N_MEAN
from utils.data_loader import parse_csv
import config

N_SAMPLES = 3000
DT_CANDIDATES = [1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 540.0, 600.0]


def residual_stats(x, masks, dt):
    """给定 dt，返回 X 序列内部相邻步 CW 残差的逐分量统计（仅有效 agent）。"""
    Phi = compute_cw_matrix(N_MEAN, dt).numpy().astype(np.float64)
    B, T, D = x.shape
    maxN = D // 6
    s = x.reshape(B, T, maxN, 6).astype(np.float64)

    free = s[:, :-1] @ Phi.T              # (B, T-1, maxN, 6)
    r = s[:, 1:] - free                   # (B, T-1, maxN, 6)
    norm = np.linalg.norm(r, axis=-1)     # (B, T-1, maxN)

    valid = masks.reshape(B, maxN, 6).any(-1)          # (B, maxN)
    vmask = np.broadcast_to(valid[:, None, :], norm.shape)
    return float(norm[vmask].mean()), float(np.median(norm[vmask]))


def main():
    print("=" * 84)
    print(f"时间步长核验（前 {N_SAMPLES} 样本，CW 残差范数）")
    print("=" * 84)

    XN, XM = parse_csv(os.path.join(config.DATA_DIR, "X_now.csv"))
    XN2, _ = parse_csv(os.path.join(config.DATA_DIR, "X_next.csv"))
    X = np.stack(XN[:N_SAMPLES])
    Y = np.stack(XN2[:N_SAMPLES])
    M = np.stack(XM[:N_SAMPLES])

    print(f"\nX_now  shape={X.shape}   X_next shape={Y.shape}")
    print(f"N 分布: 2={int((M[:,11] & ~M[:,12]).sum())}, "
          f"3={int((M[:,17] & ~M[:,18]).sum())}, 4={int(M[:,23].sum())}")

    print("\n【A】X_now 内部相邻步残差（9 对/样本）")
    print(f"{'dt (s)':>8} {'残差均值':>12} {'残差中位数':>12}")
    print("-" * 84)
    for dt in DT_CANDIDATES:
        mu, med = residual_stats(X, M, dt)
        print(f"{dt:8.1f} {mu:12.4f} {med:12.4f}")

    print("\n【B】X_next 内部相邻步残差（9 对/样本）")
    print(f"{'dt (s)':>8} {'残差均值':>12} {'残差中位数':>12}")
    print("-" * 84)
    for dt in DT_CANDIDATES:
        mu, med = residual_stats(Y, M, dt)
        print(f"{dt:8.1f} {mu:12.4f} {med:12.4f}")

    # 【C】X_now 末步 → X_next 首步的跳变（位置 / 速度）
    B = X.shape[0]
    maxN = X.shape[2] // 6
    last_n = X[:, -1, :].reshape(B, maxN, 6)
    first_x = Y[:, 0, :].reshape(B, maxN, 6)
    valid = M.reshape(B, maxN, 6).any(-1)

    d_pos = np.linalg.norm(first_x[..., :3] - last_n[..., :3], axis=-1)   # km
    d_vel = np.linalg.norm(first_x[..., 3:] - last_n[..., 3:], axis=-1)   # km/s

    print("\n【C】X_now 末步 → X_next 首步 的跳变（跨序列边界）")
    print(f"  位置跳变: 均值 {d_pos[valid].mean():.3f} km   中位数 {np.median(d_pos[valid]):.3f} km   "
          f"最大 {d_pos[valid].max():.3f} km")
    print(f"  速度跳变: 均值 {d_vel[valid].mean() * 1000:.3f} m/s  中位数 {np.median(d_vel[valid]) * 1000:.3f} m/s  "
          f"最大 {d_vel[valid].max() * 1000:.3f} m/s")

    # 【D】X_now 内部逐步步长位移（对照）
    s = X.reshape(B, X.shape[1], maxN, 6)
    step_pos = np.linalg.norm(np.diff(s[..., :3], axis=1), axis=-1)       # (B, 9, maxN)
    vmask = np.broadcast_to(valid[:, None, :], step_pos.shape)
    print(f"\n【D】X_now 内部相邻步位移:  均值 {step_pos[vmask].mean():.3f} km  "
          f"中位数 {np.median(step_pos[vmask]):.3f} km")

    # 【E】X_next 内部逐步步长位移
    sy = Y.reshape(B, Y.shape[1], maxN, 6)
    step_pos_y = np.linalg.norm(np.diff(sy[..., :3], axis=1), axis=-1)
    vmask_y = np.broadcast_to(valid[:, None, :], step_pos_y.shape)
    print(f"【E】X_next 内部相邻步位移: 均值 {step_pos_y[vmask_y].mean():.3f} km  "
          f"中位数 {np.median(step_pos_y[vmask_y]):.3f} km")

    print("\n" + "=" * 84)
    print("判读提示：残差最小的 dt 即为该序列的真实步长；")
    print("若 X_now 的最优 dt 与 X_next 不同，说明两个序列的时间间隔不一致。")
    print("=" * 84)


if __name__ == "__main__":
    main()
