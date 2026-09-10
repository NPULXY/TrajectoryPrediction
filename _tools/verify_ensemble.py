# -*- coding: utf-8 -*-
"""
集成策略核验（只读，不修改任何产物）。

背景：backup_v7/best_ensemble_v3_v6.pth 记录 val_terminal_dist = 4.0072 km，
      看似显著优于单模型最优值 4.2936 km。但该数字存在两处口径问题：
        1. 由 _tools/ensemble_final.py 生成，评估集是 Dataset_Summary 的**前 5000 个原始样本**
           （未做 train/val/test 划分），与 train.py 的验证集不可比；
        2. 该脚本无论哪个策略最优，都硬编码保存 0.4:0.6 的元数据 →
           4.0072 未必是当时的最优值（同脚本 final_ensemble_v11.py 内硬编码的
           "v6 单独" 参照值为 4.0048，反而更低）。

本脚本在**统一的 seed=42 验证集（32,646 样本）**上，用同一口径实测：
      单模型 vs 各种加权/中位数集成，判断集成是否真有增益。

用法：python _tools/verify_ensemble.py
"""
import os
import sys
import itertools

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as C
from models.model import create_model
from utils.data_loader import FeatureScaler, load_and_split

SINGLE = [
    ("v6", "output/best_model.pth", "v6 full80 loss_balanced（当前挂载）"),
    ("v3", "_archive/weights/backup_v3/best_model_v3_ep58_td4.30.pth", "v3 起点 ep58"),
    ("v6o", "_archive/weights/backup_v6/best_model_v6_ep80_td4.294.pth", "v6 原始训练 ep80"),
    ("v11", "_archive/weights/backup_v11/best_model_seed42.pth", "v11 bagging seed42"),
]


def load(path):
    model = create_model(C.DEVICE)
    ckpt = torch.load(os.path.join(ROOT, path), map_location=C.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.to(C.DEVICE).eval()


def infer(model, loader):
    """返回标准化空间的预测 (N, 10, 24)。"""
    out = []
    with torch.no_grad():
        for x, _, mask in loader:
            pred, _ = model(x.to(C.DEVICE), return_dv=True, mask=mask.to(C.DEVICE))
            out.append(pred.cpu().numpy())
    return np.concatenate(out, axis=0)


def score(pred_norm, scaler, Y_norm, masks):
    """统一口径评分：td 口径对齐 train.py::validate()（每样本取各目标末步距离的最大值）。"""
    preds_raw = scaler.inverse_transform(pred_norm)
    targets_raw = scaler.inverse_transform(Y_norm)
    N = preds_raw.shape[0]
    pos_idx = C.POS_INDICES

    p = preds_raw[:, -1, pos_idx].reshape(N, C.MAX_N, 3)
    t = targets_raw[:, -1, pos_idx].reshape(N, C.MAX_N, 3)
    dist = np.linalg.norm(p - t, axis=-1)
    valid = masks[:, pos_idx].reshape(N, C.MAX_N, 3).any(-1)

    td_max = float(np.mean(np.max(np.where(valid, dist, -np.inf), axis=1)))
    td_mean = float(np.mean(np.sum(np.where(valid, dist, 0.0), axis=1)
                            / np.maximum(valid.sum(axis=1), 1)))

    valid12 = np.repeat(valid, 3, axis=1)
    pe = preds_raw[:, :, pos_idx] - targets_raw[:, :, pos_idx]
    m = np.broadcast_to(valid12[:, None, :], pe.shape)
    pos_rmse = float(np.sqrt(np.mean(np.abs(pe)[m] ** 2)))
    pos_mae = float(np.mean(np.abs(pe)[m]))
    return dict(td_max=td_max, td_mean=td_mean, pos_rmse=pos_rmse, pos_mae=pos_mae)


def main():
    print("=" * 92)
    print("集成策略核验（统一口径：seed=42 验证集 32,646 样本）")
    print("=" * 92)

    (_, val_X, _, _, val_Y, _, _, val_masks, _, _) = load_and_split(C.DATA_DIR)
    scaler = FeatureScaler()
    scaler.load(C.SCALER_SAVE_PATH)

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(val_X).float(),
        torch.from_numpy(val_Y).float(),
        torch.from_numpy(val_masks).bool(),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)

    preds = {}
    print("\n单模型推理：")
    for key, rel, desc in SINGLE:
        try:
            m = load(rel)
        except Exception as e:
            print(f"  [跳过] {key:4s} {rel} → {type(e).__name__}")
            continue
        preds[key] = infer(m, loader)
        s = score(preds[key], scaler, val_Y, val_masks)
        print(f"  [{key:4s}] td_max={s['td_max']:.4f}  td_mean={s['td_mean']:.4f}  "
              f"pos_rmse={s['pos_rmse']:.4f}  ({desc})")

    if len(preds) < 2:
        print("可用模型不足，无法集成。")
        return

    strategies = {}
    for k in preds:
        strategies[f"{k} 单独"] = preds[k]

    # 两两 / 三组加权网格
    keys = list(preds.keys())
    for r in range(2, len(keys) + 1):
        for combo in itertools.combinations(keys, r):
            if r == 2:
                for w in (0.3, 0.4, 0.5, 0.6, 0.7):
                    a, b = combo
                    strategies[f"{w:.1f}{a} + {1-w:.1f}{b}"] = w * preds[a] + (1 - w) * preds[b]
            strategies[" + ".join(combo) + " 平均"] = np.mean([preds[k] for k in combo], axis=0)
            strategies[" + ".join(combo) + " 中位数"] = np.median(
                np.stack([preds[k] for k in combo]), axis=0)

    rows = []
    for name, p in strategies.items():
        s = score(p, scaler, val_Y, val_masks)
        s["name"] = name
        rows.append(s)
    rows.sort(key=lambda r: r["td_max"])

    print("\n" + "=" * 92)
    print(f"{'td_max':>9} {'td_mean':>9} {'pos RMSE':>9} {'pos MAE':>8}  策略")
    print("-" * 92)
    for r in rows[:14]:
        print(f"{r['td_max']:9.4f} {r['td_mean']:9.4f} {r['pos_rmse']:9.4f} "
              f"{r['pos_mae']:8.4f}  {r['name']}")

    best_single = min((r for r in rows if r["name"].endswith("单独")),
                      key=lambda r: r["td_max"])
    best_all = rows[0]
    print("\n" + "-" * 92)
    print(f"最优单模型 : {best_single['name']}  td_max={best_single['td_max']:.4f} km")
    print(f"最优集成   : {best_all['name']}  td_max={best_all['td_max']:.4f} km")
    gain = best_single["td_max"] - best_all["td_max"]
    print(f"集成增益   : {gain:+.4f} km（{gain / best_single['td_max'] * 100:+.2f}%）")


if __name__ == "__main__":
    main()
