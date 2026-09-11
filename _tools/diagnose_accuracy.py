# -*- coding: utf-8 -*-
"""
精度瓶颈诊断（只读）。

目的：在调参之前先定位误差来源。回答四个问题：
  1) 当前模型相比两个朴素基线（persistence / CW 外推）到底好多少？
     —— 若 CW 基线已接近模型，说明"锚点选择"是主要瓶颈（v7 残差学习思路的价值）；
  2) 误差如何随预测步长增长？（是否长时程误差累积主导）
  3) 不同 N（目标数）下误差差异有多大？
  4) 误差与哪些样本特征相关？（初距、速度幅值、末端距离分位）

用法：python _tools/diagnose_accuracy.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as C
from models.cw_baseline import CWBaseline
from models.model import create_model
from utils.data_loader import FeatureScaler, load_and_split


def pos_rmse(pred_raw, tgt_raw, masks, steps=None):
    """位置 RMSE（原始量纲 km）。steps=None 表示全部时间步。"""
    pos_idx = C.POS_INDICES
    N = pred_raw.shape[0]
    sl = slice(None) if steps is None else steps
    p = pred_raw[:, sl][:, :, pos_idx]
    t = tgt_raw[:, sl][:, :, pos_idx]
    valid = np.repeat(masks[:, pos_idx].reshape(N, C.MAX_N, 3).any(-1), 3, axis=1)
    m = np.broadcast_to(valid[:, None, :], p.shape)
    return float(np.sqrt(np.mean(((p - t)[m]) ** 2)))


def terminal_dist(pred_raw, tgt_raw, masks):
    pos_idx = C.POS_INDICES
    N = pred_raw.shape[0]
    p = pred_raw[:, -1, pos_idx].reshape(N, C.MAX_N, 3)
    t = tgt_raw[:, -1, pos_idx].reshape(N, C.MAX_N, 3)
    d = np.linalg.norm(p - t, axis=-1)
    valid = masks[:, pos_idx].reshape(N, C.MAX_N, 3).any(-1)
    return np.where(valid, d, -np.inf).max(axis=1)   # 每样本取最差目标（与 train.py 口径一致）


def main():
    print("=" * 92)
    print("精度瓶颈诊断（seed=42 验证集）")
    print("=" * 92)

    (_, val_X, _, _, val_Y, _, _, val_masks, _, _) = load_and_split(C.DATA_DIR)
    scaler = FeatureScaler()
    scaler.load(C.SCALER_SAVE_PATH)

    model = create_model(C.DEVICE, scaler)
    ckpt = torch.load(C.MODEL_SAVE_PATH, map_location=C.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(C.DEVICE).eval()
    print(f"模型: epoch {ckpt['epoch']}, val_td={ckpt.get('val_terminal_dist'):.4f} km")

    cw = CWBaseline(n=C.CW_N, dt=C.CW_DT_H, num_steps=C.OUTPUT_STEPS,
                    max_N=C.MAX_N, scaler=scaler).to(C.DEVICE)

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(val_X).float(),
        torch.from_numpy(val_Y).float(),
        torch.from_numpy(val_masks).bool())
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)

    P, B_cw, B_pers, T, M = [], [], [], [], []
    with torch.no_grad():
        for x, y, m in loader:
            xd, md = x.to(C.DEVICE), m.to(C.DEVICE)
            pred, _ = model(xd, return_dv=True, mask=md)
            P.append(pred.cpu().numpy())
            B_cw.append(cw(xd[:, -1, :], md).cpu().numpy())                      # CW 外推基线
            B_pers.append(xd[:, -1:, :].repeat(1, C.OUTPUT_STEPS, 1).cpu().numpy())  # 持久基线
            T.append(y.numpy())
            M.append(m.numpy())

    P, B_cw, B_pers, T, M = map(np.concatenate, (P, B_cw, B_pers, T, M))

    # 反归一化
    inv = scaler.inverse_transform
    P_r, Bc_r, Bp_r, T_r = inv(P), inv(B_cw), inv(B_pers), inv(T)

    print("\n【1】三种预测源的整体位置 RMSE（原始量纲）")
    print(f"{'方法':<22}{'位置RMSE':>12}{'末端距离均值':>14}")
    print("-" * 92)
    for name, pr in [("持久基线 (persistence)", Bp_r),
                     ("CW 外推基线", Bc_r),
                     ("当前模型", P_r)]:
        print(f"{name:<22}{pos_rmse(pr, T_r, M):>12.4f}{float(np.mean(terminal_dist(pr, T_r, M))):>14.4f}")

    print("\n【2】模型误差随预测步长增长（各步位置 RMSE）")
    step_rmse = [pos_rmse(P_r, T_r, M, steps=slice(i, i + 1)) for i in range(C.OUTPUT_STEPS)]
    cw_step = [pos_rmse(Bc_r, T_r, M, steps=slice(i, i + 1)) for i in range(C.OUTPUT_STEPS)]
    print(f"{'步':>4}{'模型RMSE':>11}{'CW基线RMSE':>13}{'模型/CW':>10}")
    for i, (a, b) in enumerate(zip(step_rmse, cw_step)):
        print(f"{i+1:>4}{a:>11.4f}{b:>13.4f}{a/b if b else float('nan'):>10.2f}")

    print("\n【3】按目标数 N 分组（模型 vs 基线）")
    n_targets = M[:, C.POS_INDICES].reshape(-1, C.MAX_N, 3).any(-1).sum(axis=1)
    print(f"{'N':>3}{'样本数':>9}{'模型RMSE':>11}{'CW基线':>10}{'持久基线':>11}{'模型末端距离':>14}")
    for nv in (2, 3, 4):
        sel = n_targets == nv
        if sel.sum() == 0:
            continue
        print(f"{nv:>3}{int(sel.sum()):>9}"
              f"{pos_rmse(P_r[sel], T_r[sel], M[sel]):>11.4f}"
              f"{pos_rmse(Bc_r[sel], T_r[sel], M[sel]):>10.4f}"
              f"{pos_rmse(Bp_r[sel], T_r[sel], M[sel]):>11.4f}"
              f"{float(np.mean(terminal_dist(P_r[sel], T_r[sel], M[sel]))):>14.4f}")

    print("\n【4】末步误差分布（模型的每样本最大目标偏差）")
    td = terminal_dist(P_r, T_r, M)
    for q in (50, 75, 90, 95, 99):
        print(f"  {q} 分位: {np.percentile(td, q):8.4f} km")
    print(f"  均值   : {td.mean():8.4f} km   最大: {td.max():.4f} km")
    print(f"  <1 km 占比: {(td < 1).mean()*100:.2f}%    <0.5 km 占比: {(td < 0.5).mean()*100:.2f}%")

    # 误差与初始距离的相关性
    x0 = inv(val_X[:, -1, :])                      # 末步观测
    init_r = np.linalg.norm(x0[:, C.POS_INDICES].reshape(-1, C.MAX_N, 3), axis=-1)
    init_r_max = np.where(M[:, C.POS_INDICES].reshape(-1, C.MAX_N, 3).any(-1),
                          init_r, np.nan)
    init_r_max = np.nanmax(init_r_max, axis=1)
    print("\n【5】末步误差 vs 初始距离（分箱）")
    bins = [0, 20, 40, 60, 80, 120, 200]
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (init_r_max >= lo) & (init_r_max < hi)
        if sel.sum() > 50:
            print(f"  初距 [{lo:3d}, {hi:3d}) km: {int(sel.sum()):>6d} 样本  "
                  f"模型RMSE={pos_rmse(P_r[sel], T_r[sel], M[sel]):7.4f}  "
                  f"末步误差均值={td[sel].mean():7.4f} km")

    print("\n" + "=" * 92)
    print("判读提示：若 CW 基线 RMSE 明显小于持久基线，说明当前 persistence 锚点不适合机动目标，")
    print("         改用 CW 锚定（残差学习）应能显著降低误差。")
    print("=" * 92)


if __name__ == "__main__":
    main()
