# -*- coding: utf-8 -*-
"""
检查点元数据盘点：列出所有 .pth 的 epoch / val_loss / val_terminal_dist / model_type 与参数量。

用途：确认"当前最佳模型"究竟是哪一个，避免被 README 与历史日志误导。
用法：python _tools/inspect_checkpoints.py
"""
import os
import sys
import glob

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 关注的目标
TARGETS = [
    "output/best_model.pth",
    "output/latest_checkpoint.pth",
    "output/best_model_v1_oldrun.pth",
]


def describe(path, full=False):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return f"  [读取失败] {e}"

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        return "  [非标准 checkpoint 结构]"

    sd = ckpt["model_state_dict"]
    n_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    epoch = ckpt.get("epoch", "-")
    val_loss = ckpt.get("val_loss", float("nan"))
    td = ckpt.get("val_terminal_dist", float("nan"))
    btd = ckpt.get("best_terminal_dist", float("nan"))
    mtype = ckpt.get("model_type", "?")
    keys = set(sd.keys())
    # 架构判定：以决定性的权重键名为准，而非模糊子串匹配
    has_lstm = any(k.startswith("encoder_lstm.weight_ih") for k in keys)
    has_tf = any(".self_attn." in k for k in keys)
    assert not (has_lstm and has_tf), f"键名同时出现 LSTM 与 Transformer 特征: {path}"
    if has_lstm:
        arch = "v3 PI-LSTM"
    elif has_tf:
        arch = "v5 Transformer-PI"
    else:
        arch = "未知/其他"

    lines = [
        f"epoch={epoch}  model_type={mtype}  params={n_params:,}  arch={arch}",
        f"  val_loss(pred MSE)={val_loss:.6f}  val_terminal_dist={td:.4f} km  best_td={btd:.4f} km",
    ]
    if full:
        extra = {k: ckpt[k] for k in ("best_epoch", "patience_counter") if k in ckpt}
        if extra:
            lines.append(f"  其他: {extra}")
    return "\n".join(lines)


def main():
    print("=" * 74)
    print("当前主线检查点")
    print("=" * 74)
    for rel in TARGETS:
        p = os.path.join(ROOT, rel)
        print(f"\n[{rel}]")
        if not os.path.exists(p):
            print("  不存在")
            continue
        print(describe(p, full=True))

    print("\n" + "=" * 74)
    print("备份检查点（按 val_terminal_dist 排序）")
    print("=" * 74)
    rows = []
    for p in glob.glob(os.path.join(ROOT, "_archive", "weights", "backup_*", "*.pth")):
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
                continue
            rows.append((
                ckpt.get("val_terminal_dist", float("inf")),
                os.path.relpath(p, ROOT),
                ckpt.get("epoch", "-"),
                ckpt.get("val_loss", float("nan")),
                ckpt.get("model_type", "?"),
            ))
        except Exception:
            continue
    rows.sort(key=lambda r: r[0])
    for td, name, ep, vl, mt in rows:
        print(f"  td={td:8.4f} km | ep={ep:>4} | val_loss={vl:.6f} | {mt:<10} | {name}")


if __name__ == "__main__":
    main()
