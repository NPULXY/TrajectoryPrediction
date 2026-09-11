# -*- coding: utf-8 -*-
"""
集成评估（测试集，统一口径）。

用途：方向1（多 seed 集成）的验证工具。对每个成员单独评估，再评估等权集成，
对比主指标（物理空间位置 RMSE）与末端距离。

口径与 evaluate.py 一致：全量测试集（32,647 样本），原始物理量纲，mask 仅取有效维度。
集成为**标准化空间的等权平均**（成员预测同尺度，可直接平均）。

用法：
    python _tools/eval_ensemble.py                       # 用内置默认成员列表
    python _tools/eval_ensemble.py p1 p2 p3              # 指定模型路径
"""
import os
import re
import sys
import itertools

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as C
from models.model import create_model
from models.pinn_lstm import PhysicsInformedTrajectoryLSTM
from utils.data_loader import FeatureScaler, load_and_split

DEFAULT_MEMBERS = [
    ("physfix", "output/best_model.pth"),
    ("e1", "output/best_model_e1.pth"),
    ("e2", "output/best_model_e2.pth"),
]


def build_from_ckpt(ckpt, scaler, device):
    """
    依据检查点权重形状**推断架构**并构建模型。

    必要性：不同容量的模型（如 hidden=512）无法加载进按 config 默认值（384）
    构建的网络 —— 形状不匹配会被 strict=False 静默保留为随机初始化，导致评估结果错误。
    本函数从 `encoder_lstm.weight_ih_l0` 推断 hidden（= 行数/4）与层数。
    """
    sd = ckpt["model_state_dict"]
    if "encoder_lstm.weight_ih_l0" not in sd:
        return create_model(device, scaler)          # 非 PI-LSTM，回退默认
    hidden = sd["encoder_lstm.weight_ih_l0"].shape[0] // 4
    layer_ids = [int(m.group(1)) for k in sd
                 if (m := re.match(r"encoder_lstm\.weight_ih_l(\d+)$", k))]
    layers = max(layer_ids) + 1 if layer_ids else 4
    cond_dim = sd["condition_builder.dv_mlp.0.weight"].shape[0] if \
        "condition_builder.dv_mlp.0.weight" in sd else C.CONDITION_EMBED_DIM

    model = PhysicsInformedTrajectoryLSTM(
        hidden_size=hidden, num_layers=layers,
        condition_embed_dim=cond_dim, scaler=scaler)
    return model.to(device)


def metric(pred_norm, tgt_norm, masks, scaler):
    """位置 RMSE / MAE、末端距离（每样本取最差目标）。"""
    P = scaler.inverse_transform(pred_norm)
    T = scaler.inverse_transform(tgt_norm)
    N = P.shape[0]
    pos = C.POS_INDICES

    valid = masks[:, pos].reshape(N, C.MAX_N, 3).any(-1)           # (N, max_N)
    v12 = np.repeat(valid, 3, axis=1)                              # (N, 12)

    pe = P[:, :, pos] - T[:, :, pos]                               # (N, 10, 12)
    m = np.broadcast_to(v12[:, None, :], pe.shape)
    pos_mse = float(np.mean(pe[m] ** 2))
    pos_mae = float(np.mean(np.abs(pe[m])))

    d = np.linalg.norm(P[:, -1, pos].reshape(N, C.MAX_N, 3)
                       - T[:, -1, pos].reshape(N, C.MAX_N, 3), axis=-1)
    td = float(np.mean(np.max(np.where(valid, d, -np.inf), axis=1)))

    return {"pos_rmse": float(np.sqrt(pos_mse)), "pos_mae": pos_mae, "td": td}


def main():
    args = sys.argv[1:]
    members = ([(os.path.splitext(os.path.basename(p))[0], p) for p in args]
               if args else DEFAULT_MEMBERS)

    print("=" * 96)
    print("集成评估（测试集，全量 32,647 样本）")
    print("=" * 96)

    (_, _, te_X, _, _, te_Y, _, _, te_M, _) = load_and_split(C.DATA_DIR)
    scaler = FeatureScaler()
    scaler.load(C.SCALER_SAVE_PATH)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(te_X).float(),
            torch.from_numpy(te_Y).float(),
            torch.from_numpy(te_M).bool()),
        batch_size=512, shuffle=False)

    # 真值（numpy，供指标计算）
    T_norm = te_Y

    preds = {}
    for name, rel in members:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"[跳过] 不存在: {rel}")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"]
        hidden = sd["encoder_lstm.weight_ih_l0"].shape[0] // 4 if \
            "encoder_lstm.weight_ih_l0" in sd else "-"
        model = build_from_ckpt(ck, scaler, C.DEVICE)
        try:
            missing = sorted(set(model.state_dict()) - set(sd))
            model.load_state_dict(sd, strict=False)
            if missing:
                print(f"  ⚠ {name}: {len(missing)} 个键缺失（可能架构推断有误）")
        except Exception as e:
            print(f"[不兼容] {name}: {type(e).__name__} {str(e)[:80]}")
            continue
        model = model.to(C.DEVICE).eval()

        out = []
        with torch.no_grad():
            for x, _, m in loader:
                p, _ = model(x.to(C.DEVICE), return_dv=True, mask=m.to(C.DEVICE))
                out.append(p.cpu().numpy())
        preds[name] = np.concatenate(out, axis=0)
        print(f"[完成] {name:10s} epoch={ck['epoch']:<4} hidden={hidden:<4} "
              f"td(记录)={ck.get('val_terminal_dist', float('nan')):.4f} km")

    if not preds:
        print("无可用模型。")
        return

    rows = []
    print("\n【单模型】")
    print(f"{'成员':<12}{'位置RMSE':>11}{'位置MAE':>11}{'末端距离':>11}")
    print("-" * 96)
    for name in preds:
        r = metric(preds[name], T_norm, te_M, scaler)
        r["name"] = name
        rows.append(r)
        print(f"{name:<12}{r['pos_rmse']:>11.4f}{r['pos_mae']:>11.4f}{r['td']:>11.4f}")

    # 全成员等权集成
    if len(preds) >= 2:
        names = list(preds)
        ens = np.mean([preds[n] for n in names], axis=0)
        r = metric(ens, T_norm, te_M, scaler)
        r["name"] = f"集成({len(names)}成员)"
        rows.append(r)
        print(f"\n【集成】")
        print(f"{'组合':<12}{'位置RMSE':>11}{'位置MAE':>11}{'末端距离':>11}")
        print("-" * 96)
        print(f"{r['name']:<12}{r['pos_rmse']:>11.4f}{r['pos_mae']:>11.4f}{r['td']:>11.4f}")

        # 两两组合（便于判断增益来源）
        if len(names) >= 3:
            print("\n【两两组合】")
            for a, b in itertools.combinations(names, 2):
                rr = metric((preds[a] + preds[b]) / 2, T_norm, te_M, scaler)
                print(f"{a} + {b:<8}{rr['pos_rmse']:>11.4f}{rr['pos_mae']:>11.4f}{rr['td']:>11.4f}")

    # 汇总对比
    best_single = min([r for r in rows if "集成" not in r["name"]], key=lambda r: r["pos_rmse"])
    print("\n" + "=" * 96)
    print(f"最优单模型 : {best_single['name']}  pos RMSE={best_single['pos_rmse']:.4f} km  "
          f"td={best_single['td']:.4f} km")
    if len(preds) >= 2:
        ens_r = [r for r in rows if "集成" in r["name"]][0]
        g1 = (best_single["pos_rmse"] - ens_r["pos_rmse"]) / best_single["pos_rmse"] * 100
        g2 = (best_single["td"] - ens_r["td"]) / best_single["td"] * 100
        print(f"集成       : {ens_r['name']}  pos RMSE={ens_r['pos_rmse']:.4f} km  "
              f"td={ens_r['td']:.4f} km")
        print(f"集成增益   : 位置 RMSE {g1:+.2f}%   末端距离 {g2:+.2f}%")
    print("=" * 96)


if __name__ == "__main__":
    main()
