# -*- coding: utf-8 -*-
"""
集成子集选择（严谨版：在**验证集**上选组合，在**测试集**上报告）。

为什么不能直接在测试集上挑最好的组合？
  在测试集上枚举 2^N 个子集并选最优，会产生**测试集过拟合**（选择偏差），
  报告的数字会偏乐观。正确做法是在验证集上做选择，再在测试集上无偏评估。

流程：
  1. 对每个候选模型计算 验证集 / 测试集 的标准化预测（各一次，缓存复用）
  2. 枚举全部非空子集，在**验证集**上按位置 RMSE 选最优
  3. 报告该子集在**测试集**上的表现，并与单模型最优对比

用法：python _tools/select_ensemble.py [--max-subset 4]
"""
import os
import sys
import itertools
import json

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as C
from utils.data_loader import FeatureScaler, load_and_split
from utils.ensemble import load_members, infer_arch

# 候选池（覆盖所有已训练配置）
CANDIDATES = [
    "output/best_model.pth",       # 基线 384
    "output/best_model_e1.pth",    # 384  seed101
    "output/best_model_e2.pth",    # 384  seed202
    "output/best_model_h512.pth",  # 512  增容量
    "output/best_model_p1.pth",    # 384 + 位置损失
    "output/best_model_f1.pth",    # 512 + 位置损失  seed301
    "output/best_model_f2.pth",    # 512 + 位置损失  seed302
]


def metrics(pred_norm, tgt_norm, masks, scaler):
    """位置 RMSE / MAE、末端距离。"""
    P = scaler.inverse_transform(pred_norm)
    T = scaler.inverse_transform(tgt_norm)
    N = P.shape[0]
    pos = C.POS_INDICES
    valid = masks[:, pos].reshape(N, C.MAX_N, 3).any(-1)
    v12 = np.repeat(valid, 3, axis=1)

    pe = P[:, :, pos] - T[:, :, pos]
    m = np.broadcast_to(v12[:, None, :], pe.shape)
    rmse = float(np.sqrt(np.mean(pe[m] ** 2)))
    mae = float(np.mean(np.abs(pe[m])))

    d = np.linalg.norm(P[:, -1, pos].reshape(N, C.MAX_N, 3)
                       - T[:, -1, pos].reshape(N, C.MAX_N, 3), axis=-1)
    td = float(np.mean(np.max(np.where(valid, d, -np.inf), axis=1)))
    return {"rmse": rmse, "mae": mae, "td": td}


@torch.no_grad()
def predict_all(path, scaler, loaders):
    """加载单个模型，对其在多个 loader 上推理，返回 {标签: 预测数组}。"""
    from models.model import create_model
    from models.pinn_lstm import PhysicsInformedTrajectoryLSTM

    full = os.path.join(ROOT, path)
    ck = torch.load(full, map_location="cpu", weights_only=False)
    arch = infer_arch(ck["model_state_dict"])
    if arch:
        model = PhysicsInformedTrajectoryLSTM(hidden_size=arch[0], num_layers=arch[1],
                                              condition_embed_dim=arch[2], scaler=scaler)
    else:
        model = create_model(C.DEVICE, scaler)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model = model.to(C.DEVICE).eval()

    out = {}
    for tag, (X, M) in loaders.items():
        buf = []
        for i in range(0, len(X), 512):
            xb = torch.from_numpy(X[i:i + 512]).float().to(C.DEVICE)
            mb = torch.from_numpy(M[i:i + 512]).bool().to(C.DEVICE)
            p, _ = model(xb, return_dv=True, mask=mb)
            buf.append(p.cpu().numpy())
        out[tag] = np.concatenate(buf, axis=0)
    return out, ck.get("epoch"), arch[0] if arch else "-"


def main():
    max_subset = 4
    if "--max-subset" in sys.argv:
        max_subset = int(sys.argv[sys.argv.index("--max-subset") + 1])

    print("=" * 96)
    print(f"集成子集选择（验证集选组合，测试集报告；子集上限 {max_subset}）")
    print("=" * 96)

    (_, va_X, te_X, _, va_Y, te_Y, _, va_M, te_M, _) = load_and_split(C.DATA_DIR)
    scaler = FeatureScaler()
    scaler.load(C.SCALER_SAVE_PATH)

    loaders = {"val": (va_X, va_M), "test": (te_X, te_M)}
    Ys = {"val": va_Y, "test": te_Y}
    Ms = {"val": va_M, "test": te_M}

    preds, names, name_to_path = {}, [], {}
    for path in CANDIDATES:
        if not os.path.exists(os.path.join(ROOT, path)):
            print(f"[跳过] 不存在: {path}")
            continue
        out, ep, h = predict_all(path, scaler, loaders)
        key = os.path.splitext(os.path.basename(path))[0]
        key = key.replace("best_model_", "").replace("best_model", "base") or "base"
        preds[key] = out
        names.append(key)
        name_to_path[key] = path
        print(f"[完成] {key:<10} hidden={h:<4} epoch={ep}")

    if len(names) < 2:
        print("可用模型不足。")
        return

    # 单模型基线（验证集 & 测试集）
    print("\n【单模型】")
    print(f"{'模型':<12}{'val RMSE':>10}{'test RMSE':>11}{'test MAE':>10}{'test td':>10}")
    print("-" * 96)
    singles = {}
    for n in names:
        mv = metrics(preds[n]["val"], Ys["val"], Ms["val"], scaler)
        mt = metrics(preds[n]["test"], Ys["test"], Ms["test"], scaler)
        singles[n] = (mv, mt)
        print(f"{n:<12}{mv['rmse']:>10.4f}{mt['rmse']:>11.4f}{mt['mae']:>10.4f}{mt['td']:>10.4f}")

    # 枚举子集，在**验证集**上选最优
    print(f"\n枚举 {len(names)} 个模型的全部 2~{max_subset} 元子集 ...")
    best = None
    rows = []
    for r in range(2, min(max_subset, len(names)) + 1):
        for combo in itertools.combinations(names, r):
            pv = np.mean([preds[n]["val"] for n in combo], axis=0)
            mv = metrics(pv, Ys["val"], Ms["val"], scaler)
            rows.append((mv["rmse"], combo, mv))
    rows.sort(key=lambda t: t[0])

    print("\n【验证集前 10（用于选择）】")
    for rmse, combo, mv in rows[:10]:
        print(f"  val RMSE={rmse:.4f}  ({'+'.join(combo)})")

    # 用验证集最优子集，报告测试集表现
    best_rmse, best_combo, _ = rows[0]
    pt = np.mean([preds[n]["test"] for n in best_combo], axis=0)
    mt = metrics(pt, Ys["test"], Ms["test"], scaler)

    best_single = min(names, key=lambda n: singles[n][0]["rmse"])
    sv, st = singles[best_single]

    print("\n" + "=" * 96)
    print("最终方案（验证集选出，测试集评估）")
    print("=" * 96)
    print(f"  选中子集        : {' + '.join(best_combo)}")
    print(f"  验证集位置 RMSE : {best_rmse:.4f} km")
    print(f"  测试集位置 RMSE : {mt['rmse']:.4f} km")
    print(f"  测试集位置 MAE  : {mt['mae']:.4f} km")
    print(f"  测试集末端距离  : {mt['td']:.4f} km")
    print("")
    print(f"  对照-最优单模型 : {best_single}")
    print(f"    val RMSE      : {sv['rmse']:.4f} km")
    print(f"    test RMSE     : {st['rmse']:.4f} km   (MAE {st['mae']:.4f}, td {st['td']:.4f})")
    print("")
    print(f"  相对最优单模型增益: 位置 RMSE {(st['rmse'] - mt['rmse']) / st['rmse'] * 100:+.2f}%   "
          f"末端距离 {(st['td'] - mt['td']) / st['td'] * 100:+.2f}%")

    # 落盘配置，供 predict.py / evaluate.py 直接使用
    out_cfg = {
        "members": [name_to_path[n] for n in best_combo],
        "member_keys": list(best_combo),
        "val_pos_rmse": best_rmse,
        "test_pos_rmse": mt["rmse"],
        "test_pos_mae": mt["mae"],
        "test_td": mt["td"],
    }
    cfg_path = os.path.join(ROOT, "output", "ensemble_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, ensure_ascii=False, indent=2)
    print(f"\n配置已写入: {cfg_path}")
    print(f"  TP_ENSEMBLE=\"{','.join(out_cfg['members'])}\"")


if __name__ == "__main__":
    main()
