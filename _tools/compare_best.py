# -*- coding: utf-8 -*-
"""
候选权重横向对比（只读，不修改任何产物）。

目的：回答"现在哪个版本最好"。
仅凭 checkpoint 里记录的 val_terminal_dist 不足以下结论，原因有三：
  1. 记录值可能来自不同口径（v7/v10/v11 是独立脚本，评估代码与 train.py 不完全一致）
  2. 部分权重架构不同（v5 Transformer / v7a hidden=256 / v10 hidden=512），与当前 config 不兼容
  3. 需要确认权重能否被当前 config 构建的模型严格加载

做法：加载数据集一次，在**同一验证集**（seed=42，32,646 样本）上用统一口径评估所有候选，
     指标口径对齐 train.py 的 validate()：
       - td_max  : 每样本取"各有效目标中末步 3D 距离的最大值"，再对样本求均值（= checkpoint 记录值）
       - td_mean : 每样本取"各有效目标末步 3D 距离的均值"，再对样本求均值（= 训练损失口径）
       - 位置/速度 RMSE、MAE（原始物理量纲）

用法：python _tools/compare_best.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as C
from models.model import create_model
from utils.data_loader import FeatureScaler, load_and_split

# 候选权重（相对 ROOT），覆盖 v3 架构各训练分支的代表性结果
CANDIDATES = [
    ("output/best_model.pth",                                    "当前挂载（full80 loss_balanced）"),
    ("_archive/weights/backup_v6/best_model_v6_ep80_td4.294.pth",                 "v6 原始训练（历史记录最优）"),
    ("_archive/weights/backup_v6/best_model_v6_loss_unbalanced_replay_ep79_td4.2957.pth", "v6 replay（loss 未平衡）"),
    ("_archive/weights/backup_v6/best_model_v6_loss_balanced_ep78_td4.2964.pth",   "loss_balanced 续训 22ep"),
    ("_archive/weights/backup_v6/best_model_v6_loss_balanced_full200attempt_ep81_td4.3165.pth", "loss_balanced 长程续训"),
    ("_archive/weights/backup_v3/best_model_v3_ep58_td4.30.pth",                  "v3 起点 ep58"),
    ("_archive/weights/backup_v6/best_model_v6_loss_balanced_full80_ep80_td4.3171.pth", "full80 备份（应与当前一致）"),
    # 架构不同者作对照，预期加载失败
    ("_archive/weights/backup_v6/best_model_v5_transformer_loss_balanced_ep12_td5.4947.pth", "v5 Transformer（架构不同）"),
    ("_archive/weights/backup_v7/best_model_v7a.pth",                             "v7a hidden=256（架构不同）"),
]


def eval_on_val(model, loader, scaler, device):
    """在验证集上按统一口径计算指标（向量化，避免 Python 逐样本循环）。"""
    model.eval()
    preds, targets, masks = [], [], []
    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            pred, _ = model(x, return_dv=True, mask=mask.to(device))
            preds.append(pred.cpu())
            targets.append(y)
            masks.append(mask)

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    masks = torch.cat(masks).numpy()

    preds_raw = scaler.inverse_transform(preds)
    targets_raw = scaler.inverse_transform(targets)

    pos_idx = C.POS_INDICES
    vel_idx = C.VEL_INDICES
    N = preds_raw.shape[0]

    # 末端 3D 距离（原始量纲 km）
    p = preds_raw[:, -1, pos_idx].reshape(N, C.MAX_N, 3)
    t = targets_raw[:, -1, pos_idx].reshape(N, C.MAX_N, 3)
    dist = np.linalg.norm(p - t, axis=-1)                      # (N, max_N)
    valid = masks[:, pos_idx].reshape(N, C.MAX_N, 3).any(-1)   # (N, max_N)

    d_valid = np.where(valid, dist, -np.inf)
    td_max = np.mean(np.max(d_valid, axis=1))                  # 对齐 checkpoint 记录
    td_mean = np.mean(np.sum(np.where(valid, dist, 0.0), axis=1)
                      / np.maximum(valid.sum(axis=1), 1))

    # 位置 / 速度 RMSE（仅有效维度；mask 广播到时间维）
    valid12 = np.repeat(valid, 3, axis=1)                      # (N, 12)

    pe = preds_raw[:, :, pos_idx] - targets_raw[:, :, pos_idx]  # (N, 10, 12)
    m_pos = np.broadcast_to(valid12[:, None, :], pe.shape)
    pos_mse = np.mean(np.abs(pe)[m_pos] ** 2)
    pos_mae = np.mean(np.abs(pe)[m_pos])

    ve = preds_raw[:, :, vel_idx] - targets_raw[:, :, vel_idx]
    m_vel = np.broadcast_to(valid12[:, None, :], ve.shape)
    vel_mse = np.mean(np.abs(ve)[m_vel] ** 2)
    vel_mae = np.mean(np.abs(ve)[m_vel])

    # 每样本"最小末端距离 < 1 km"视为成功
    min_dist = np.where(valid, dist, np.inf).min(axis=1)
    succ_1km = float(np.mean(min_dist < 1.0))

    return {
        "td_max": td_max,
        "td_mean": td_mean,
        "pos_rmse": float(np.sqrt(pos_mse)),
        "pos_mae": float(pos_mae),
        "vel_rmse": float(np.sqrt(vel_mse)),
        "vel_mae": float(vel_mae),
        "succ_1km": succ_1km,
    }


def main():
    print("=" * 96)
    print("候选权重横向对比（统一在 seed=42 验证集 32,646 样本上评估）")
    print("=" * 96)

    (_, val_X, _, _, val_Y, _, _, val_masks, _, _) = load_and_split(C.DATA_DIR)
    scaler = FeatureScaler()
    scaler.load(C.SCALER_SAVE_PATH)

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(val_X).float(),
        torch.from_numpy(val_Y).float(),
        torch.from_numpy(val_masks).bool(),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)

    rows = []
    for rel, desc in CANDIDATES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"[跳过] 不存在: {rel}")
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        model = create_model(C.DEVICE)
        try:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        except Exception as e:
            print(f"[不兼容] {rel}\n          {type(e).__name__}: {str(e).splitlines()[0][:110]}")
            continue

        model = model.to(C.DEVICE)
        m = eval_on_val(model, loader, scaler, C.DEVICE)
        m["rel"] = rel
        m["desc"] = desc
        m["epoch"] = ckpt.get("epoch", "-")
        m["rec_td"] = ckpt.get("val_terminal_dist", float("nan"))
        rows.append(m)
        print(f"[完成] {rel}  td_max={m['td_max']:.4f} km")

    if not rows:
        print("无可用结果。")
        return

    rows.sort(key=lambda r: r["td_max"])

    print("\n" + "=" * 96)
    print("结果（按 td_max 升序，越小越好）")
    print("=" * 96)
    hdr = (f"{'td_max':>9} {'td_mean':>9} {'记录td':>9} {'pos RMSE':>9} "
           f"{'pos MAE':>8} {'vel RMSE':>10} {'<1km':>6} {'ep':>4}  说明")
    print(hdr)
    print("-" * 96)
    for r in rows:
        print(f"{r['td_max']:9.4f} {r['td_mean']:9.4f} {r['rec_td']:9.4f} "
              f"{r['pos_rmse']:9.4f} {r['pos_mae']:8.4f} {r['vel_rmse']:10.2e} "
              f"{r['succ_1km']*100:5.1f}% {str(r['epoch']):>4}  {r['desc']}")

    best = rows[0]
    print("\n" + "-" * 96)
    print(f"最优: {best['rel']}")
    print(f"      {best['desc']}｜epoch {best['epoch']}｜td_max={best['td_max']:.4f} km｜"
          f"位置 RMSE={best['pos_rmse']:.4f} km")
    cur = next((r for r in rows if r["rel"] == "output/best_model.pth"), None)
    if cur and cur is not best:
        gap = cur["td_max"] - best["td_max"]
        print(f"当前挂载 output/best_model.pth 的 td_max={cur['td_max']:.4f} km，"
              f"比最优差 {gap:+.4f} km（{gap / best['td_max'] * 100:+.2f}%）")


if __name__ == "__main__":
    main()
