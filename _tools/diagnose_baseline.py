# -*- coding: utf-8 -*-
"""
基线一致性诊断（只读，不修改任何产物）。

验证三件事：
  A. output/latest_checkpoint.pth 的架构是否与 config.py 当前配置一致
     → 判断 `python train.py`（RESUME_TRAINING=True）是否会直接抛异常
  B. best_model.pth（v3 PI-LSTM）能否被当前 config 构建的模型严格加载
  C. predict.py 推理时未传 mask（等同于全 True）与训练/验证时传 mask 的输出差异
     → 量化"训练-推理不一致"的严重程度

用法：python _tools/diagnose_baseline.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as C
from models.model import create_model
from utils.data_loader import FeatureScaler, load_and_split, create_dataloaders


def sep(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    sep("A. 配置与主线检查点的一致性")
    print(f"config: USE_TRANSFORMER={C.USE_TRANSFORMER}  PHYSICS_ENABLED={C.PHYSICS_ENABLED}")
    model = create_model(C.DEVICE)
    print(f"config 构建的模型: {type(model).__name__}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    for name in ("latest_checkpoint.pth", "best_model.pth"):
        p = os.path.join(C.OUTPUT_DIR, name)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        has_lstm = any(k.startswith("encoder_lstm.weight_ih") for k in sd)
        arch = "v3 PI-LSTM" if has_lstm else "v5 Transformer-PI"
        print(f"\n[{name}] epoch={ckpt.get('epoch')}  arch={arch}  "
              f"params={sum(v.numel() for v in sd.values()):,}")
        # 真正尝试加载，捕获异常（train.py 第 571 行无 try/except）
        try:
            model.load_state_dict(sd)
            print("  → load_state_dict(strict=True): 成功")
        except Exception as e:
            print(f"  → load_state_dict(strict=True): 失败 → {type(e).__name__}")
            msg = str(e).split("\n")
            print(f"     {msg[0][:200]}")
            if len(msg) > 1:
                print(f"     {msg[1][:200]}")
            print("     ⚠ train.py 第 571 行无 try/except，续训将直接中断")

    sep("B. best_model.pth 严格加载核验（独立实例）")
    model = create_model(C.DEVICE)
    ckpt = torch.load(os.path.join(C.OUTPUT_DIR, "best_model.pth"),
                      map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"成功。epoch={ckpt['epoch']}  td={ckpt.get('val_terminal_dist'):.4f} km")
    except Exception as e:
        print(f"失败: {type(e).__name__}: {str(e)[:300]}")

    sep("C. 推理 mask 缺失造成的一致性偏差（predict.py 第 284 行）")
    model = model.to(C.DEVICE).eval()
    scaler = FeatureScaler()
    scaler.load(C.SCALER_SAVE_PATH)

    (_, val_X, _, _, val_Y, _, _, val_masks, _, _) = load_and_split(C.DATA_DIR)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(val_X).float(),
            torch.from_numpy(val_Y).float(),
            torch.from_numpy(val_masks).bool(),
        ), batch_size=256, shuffle=False)

    x, y, mask = next(iter(loader))
    x, y, mask = x.to(C.DEVICE), y.to(C.DEVICE), mask.to(C.DEVICE)

    with torch.no_grad():
        pred_masked, _ = model(x, return_dv=True, mask=mask)      # 训练/验证/evaluate 用法
        pred_nomask, _ = model(x, return_dv=True)                 # predict.py 用法（mask=None）

    n_targets = mask.reshape(x.shape[0], C.MAX_N, 6).any(-1).sum(-1)
    diff_norm = (pred_masked - pred_nomask).abs()
    # 反归一化到 km
    std = torch.from_numpy(scaler.std.astype("float32")).to(C.DEVICE)
    mean = torch.from_numpy(scaler.mean.astype("float32")).to(C.DEVICE)
    pm = pred_masked * (std + 1e-8) + mean
    pn = pred_nomask * (std + 1e-8) + mean
    d_km = (pm - pn).abs()

    pos_idx = C.POS_INDICES
    print(f"batch 大小: {x.shape[0]}   N 分布: "
          f"2={int((n_targets==2).sum())}, 3={int((n_targets==3).sum())}, "
          f"4={int((n_targets==4).sum())}")
    print(f"标准化空间平均绝对差: {diff_norm.mean().item():.6f}")
    print(f"原始空间位置平均绝对差: {d_km[:, :, pos_idx].mean().item():.6f} km")
    print(f"原始空间位置最大绝对差: {d_km[:, :, pos_idx].max().item():.6f} km")

    for nv in (2, 3, 4):
        sel = n_targets == nv
        if sel.sum() == 0:
            continue
        sub = d_km[sel][:, :, pos_idx]
        print(f"  N={nv} ({int(sel.sum()):3d} 样本): 位置平均差 {sub.mean():.6f} km, "
              f"最大 {sub.max():.6f} km")

    print("\n判读：若 N<4 分组差异显著大于 N=4，即为 dv_global 除以 max_N(4) 而非实际 N 所致。")


if __name__ == "__main__":
    main()
