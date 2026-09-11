# -*- coding: utf-8 -*-
"""
训练/验证指标口径一致性核验（只读）。

动机：train.py 中"训练 pred 损失"与"验证 pred 损失"可能使用了**不同的度量**，
若如此，则二者不可直接比较，据此判断"过拟合"会得出错误结论。

train_epoch 中 l_pred =
    huber_loss_per_sample(...) 经 outlier 加权（td>10km→0.1, 5–10km→0.3）后的加权平均

validate 中 l_pred =
    utils.data_loader.masked_mse_loss(...)  —— 纯 MSE，无加权

本脚本对**同一个模型**分别计算 train/val 上的两种度量，判断：
  1) 是否存在真实过拟合（同度量下 train vs val 的差距）
  2) 历史报告中"2.4× 差距"有多少来自度量不一致

用法：python _tools/verify_metrics_consistency.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as C
from models.model import create_model
from utils.data_loader import FeatureScaler, load_and_split, masked_mse_loss


def huber_weighted(pred, y, mask, scaler, device):
    """复刻 train.py 的 train_epoch 中 l_pred 的定义。"""
    diff = (pred - y).abs()
    delta = 1.0
    quad = torch.minimum(diff, torch.tensor(delta, device=device))
    lin = diff - quad
    per_elem = 0.5 * quad.pow(2) + delta * lin
    me = mask.unsqueeze(1).expand_as(per_elem)
    n_valid = me.float().sum(dim=(1, 2)).clamp(min=1)
    per_sample = (per_elem * me).sum(dim=(1, 2)) / n_valid

    # 末步物理距离（用于 outlier 降权）
    eps = 1e-8
    pos_idx = [i * 6 + j for i in range(C.MAX_N) for j in range(3)]
    mean = torch.from_numpy(scaler.mean.astype("float32")).to(device)
    std = torch.from_numpy(scaler.std.astype("float32")).to(device)
    pp = (pred[:, -1, pos_idx] * (std[pos_idx] + eps) + mean[pos_idx]).reshape(-1, C.MAX_N, 3)
    pt = (y[:, -1, pos_idx] * (std[pos_idx] + eps) + mean[pos_idx]).reshape(-1, C.MAX_N, 3)
    dist = torch.norm(pp - pt, dim=-1)
    vp = mask[:, pos_idx].reshape(-1, C.MAX_N, 3).any(-1).float()
    td = (dist * vp).sum(-1) / vp.sum(-1).clamp(min=1)
    w = torch.ones_like(td)
    w = torch.where(td > 10, torch.tensor(0.1, device=device), w)
    w = torch.where((td > 5) & (td <= 10), torch.tensor(0.3, device=device), w)
    return ((per_sample * w).sum() / w.sum().clamp(min=1)).item()


@torch.no_grad()
def eval_both(model, loader, scaler, device):
    model.eval()
    P, Y, M = [], [], []
    for x, y, mask in loader:
        pred, _ = model(x.to(device), return_dv=True, mask=mask.to(device))
        P.append(pred.cpu()); Y.append(y); M.append(mask)
    P, Y, M = torch.cat(P), torch.cat(Y), torch.cat(M)
    P, Y, M = P.to(device), Y.to(device), M.to(device)
    return {
        "MSE": masked_mse_loss(P, Y, M).item(),
        "Huber加权": huber_weighted(P, Y, M, scaler, device),
    }


def main():
    print("=" * 88)
    print("训练/验证指标口径一致性核验")
    print("=" * 88)

    (tr_X, va_X, _, tr_Y, va_Y, _, tr_M, va_M, _, _) = load_and_split(C.DATA_DIR)
    scaler = FeatureScaler()
    scaler.load(C.SCALER_SAVE_PATH)

    def loader_of(X, Y, M):
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X).float(), torch.from_numpy(Y).float(),
            torch.from_numpy(M).bool())
        return torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)

    tr_loader, va_loader = loader_of(tr_X, tr_Y, tr_M), loader_of(va_X, va_Y, va_M)

    model = create_model(C.DEVICE, scaler)
    ck = torch.load(C.MODEL_SAVE_PATH, map_location=C.DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model = model.to(C.DEVICE)
    print(f"\n当前 output/best_model.pth: epoch {ck['epoch']}, val_td={ck.get('val_terminal_dist'):.4f} km")

    tr = eval_both(model, tr_loader, scaler, C.DEVICE)
    va = eval_both(model, va_loader, scaler, C.DEVICE)

    print("\n【同一模型下的度量对照】")
    print(f"{'度量':<14}{'训练集':>14}{'验证集':>14}{'val/train':>12}")
    print("-" * 88)
    for k in ("MSE", "Huber加权"):
        r = va[k] / tr[k] if tr[k] else float("nan")
        print(f"{k:<14}{tr[k]:>14.6f}{va[k]:>14.6f}{r:>12.3f}")

    print("\n【train.py 历史日志中报告的两个数字如何产生】")
    print(f"  训练端报告值 = Huber加权 = {tr['Huber加权']:.6f}")
    print(f"  验证端报告值 = 纯 MSE    = {va['MSE']:.6f}")
    print(f"  → 二者比 = {va['MSE'] / tr['Huber加权']:.3f}×（这就是历史报告中的'泛化差距'）")

    print("\n【同度量下的真实泛化差距】")
    print(f"  纯 MSE    : val/train = {va['MSE'] / tr['MSE']:.3f}×  ← 真实过拟合程度")
    print(f"  Huber加权 : val/train = {va['Huber加权'] / tr['Huber加权']:.3f}×")
    print("\n判读：若「纯 MSE 的 val/train」明显小于历史报告的 2.4x，")
    print("      则此前'过拟合严重'的判断主要来自度量不一致，而非真实过拟合。")
    print("=" * 88)


if __name__ == "__main__":
    main()
