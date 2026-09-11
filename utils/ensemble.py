"""
集成推理工具（2026-09-11 新增）。

动机：多 seed / 多容量集成在测试集上稳定带来 2%~5% 的位置精度提升
（见 _tools/eval_ensemble.py 的实测），但原 predict.py / evaluate.py 仅支持单模型。
本模块提供统一的「按检查点自动推断架构 → 加载 → 等权集成推理」能力。

设计要点：
- **架构自动推断**：不同容量的权重（如 hidden=384 与 512）无法加载进同一网络。
  这里从 `encoder_lstm.weight_ih_l0` 的行数推断 hidden（= 行数/4），
  从键名索引推断层数，避免形状不匹配被 strict=False 静默忽略（会产生错误的评估结果）。
- **标准化空间平均**：各成员输出同尺度，直接等权平均即可；等价于对预测取均值集成。
"""

import os
import re

import torch

from config import (
    DEVICE, MAX_DIM, CONDITION_EMBED_DIM, PHYSICS_ENABLED,
)


def infer_arch(state_dict):
    """从权重形状推断 PI-LSTM 架构，返回 (hidden_size, num_layers, condition_embed_dim)。"""
    if "encoder_lstm.weight_ih_l0" not in state_dict:
        return None
    hidden = state_dict["encoder_lstm.weight_ih_l0"].shape[0] // 4
    ids = [int(m.group(1)) for k in state_dict
           if (m := re.match(r"encoder_lstm\.weight_ih_l(\d+)$", k))]
    layers = max(ids) + 1 if ids else 4
    cond = (state_dict["condition_builder.dv_mlp.0.weight"].shape[0]
            if "condition_builder.dv_mlp.0.weight" in state_dict else CONDITION_EMBED_DIM)
    return hidden, layers, cond


def load_members(paths, scaler, device=None):
    """
    加载集成成员。返回 [(name, model), ...]；不兼容的成员会被跳过并打印原因。

    Args:
        paths: 权重路径列表（相对项目根目录或绝对路径）
        scaler: FeatureScaler（PI-LSTM 需用它把标准化输入还原到物理空间估计 Δv）
        device: 目标设备，默认 config.DEVICE
    """
    from models.model import create_model
    from models.pinn_lstm import PhysicsInformedTrajectoryLSTM

    device = device or DEVICE
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    members = []

    for rel in paths:
        rel = rel.strip()
        if not rel:
            continue
        path = rel if os.path.isabs(rel) else os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"  [跳过] 权重不存在: {rel}")
            continue

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        arch = infer_arch(sd)

        if arch is None:
            model = create_model(device, scaler)
        else:
            hidden, layers, cond = arch
            model = PhysicsInformedTrajectoryLSTM(
                hidden_size=hidden, num_layers=layers,
                condition_embed_dim=cond, scaler=scaler)

        missing = sorted(set(model.state_dict().keys()) - set(sd.keys()))
        try:
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"  [不兼容] {os.path.basename(rel)}: {type(e).__name__} {str(e)[:70]}")
            continue
        if missing:
            print(f"  [警告] {os.path.basename(rel)}: {len(missing)} 个键缺失，"
                  f"该成员权重可能不完整，已跳过")
            continue

        model = model.to(device).eval()
        name = os.path.splitext(os.path.basename(rel))[0]
        members.append((name, model))
        h = arch[0] if arch else "-"
        print(f"  [集成成员] {name:<22} hidden={h:<4} epoch={ckpt.get('epoch', '-')}")
    return members


@torch.no_grad()
def predict_batch(members, x, mask):
    """
    对单个 batch 做集成推理（标准化空间等权平均）。

    Args:
        members: load_members 的返回值
        x: (B, 10, max_dim) 标准化输入
        mask: (B, max_dim) bool
    Returns:
        (B, 10, max_dim) 标准化空间的集成预测
    """
    acc = None
    for _, model in members:
        if PHYSICS_ENABLED:
            p, _ = model(x, return_dv=True, mask=mask)
        else:
            p = model(x)
        acc = p if acc is None else acc + p
    return acc / len(members)
