"""
推理脚本 —— 加载训练好的模型对新样本进行预测并保存结果。
用法: python predict.py [--input INPUT_CSV] [--output OUTPUT_CSV]
      python predict.py --visualize [--ground-truth GT_CSV] [--top-k 10]
"""

import os
import sys

# ── conda MKL DLL 搜索路径修复 (Windows) ──
if sys.platform == "win32":
    _conda_lib_bin_candidates = [
        os.environ.get("CONDA_PREFIX", ""),
        sys.prefix,
    ]
    for _prefix in _conda_lib_bin_candidates:
        _lib_bin = os.path.join(_prefix, "Library", "bin") if _prefix else ""
        if _lib_bin and os.path.isdir(_lib_bin) and _lib_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _lib_bin + os.pathsep + os.environ.get("PATH", "")

import ast
import argparse
import numpy as np
import pandas as pd
import torch
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import (
    DATA_DIR, OUTPUT_DIR, MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    DEVICE, MAX_DIM, INPUT_STEPS, OUTPUT_STEPS, PHYSICS_ENABLED, CW_DT_H,
    ENSEMBLE_MODELS,
)
from utils.data_loader import FeatureScaler, parse_csv
from models.model import create_model

# ── 字体配置：Times New Roman，变量斜体 ──
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
    "figure.titlesize": 20,
})

# ── 目标颜色映射 ──
TARGET_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]  # 深蓝 深绿 深红 深橙


def visualize_best_predictions(X_raw, preds_raw, gt_raw, masks,
                               dist_array, top_indices, output_dir):
    """
    为末端距离最小的 top-k 样本生成综合轨迹图。

    每张图包含：
      - 左侧：所有目标的 3D 组合轨迹图
      - 右侧：每个目标的位置分量子图（x, y, z vs 时间步）

    Args:
        X_raw:       (N, 10, 24) 已知轨迹（原始量纲）
        preds_raw:   (N, 10, 24) 预测未来轨迹（原始量纲）
        gt_raw:      (N, 10, 24) 真实未来轨迹（原始量纲）
        masks:       (N, 24) 各样本有效特征 bool mask
        dist_array:  (N,) 各样本的平均末端距离 (km)
        top_indices: (k,) 最佳样本索引（已排序）
        output_dir:  图片保存目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 时间轴（步长取实测值 CW_DT_H = 60 s，2026-09-10 修正为引用常量）
    time_known = np.arange(1, INPUT_STEPS + 1) * CW_DT_H
    time_future = np.arange(INPUT_STEPS + 1, INPUT_STEPS + OUTPUT_STEPS + 1) * CW_DT_H

    for rank, sample_idx in enumerate(top_indices, 1):
        dist_val = dist_array[sample_idx]
        mask = masks[sample_idx]
        n_features = int(mask.sum())
        n_targets = n_features // 6

        # ── 创建图形 ──
        fig_height = max(12, 5.0 * n_targets)
        fig_width = fig_height * 4.0 / 3.0
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = GridSpec(n_targets, 4, figure=fig,
                      width_ratios=[2.0, 1.0, 1.0, 1.0],
                      hspace=0.55, wspace=0.40)

        # ── 左侧：3D 组合轨迹图 ──
        ax_3d = fig.add_subplot(gs[:, 0], projection="3d")

        for t in range(n_targets):
            base = t * 6
            color = TARGET_COLORS[t]

            xk = X_raw[sample_idx, :, base + 0]
            yk = X_raw[sample_idx, :, base + 1]
            zk = X_raw[sample_idx, :, base + 2]
            xt = gt_raw[sample_idx, :, base + 0]
            yt = gt_raw[sample_idx, :, base + 1]
            zt = gt_raw[sample_idx, :, base + 2]
            xp = preds_raw[sample_idx, :, base + 0]
            yp = preds_raw[sample_idx, :, base + 1]
            zp = preds_raw[sample_idx, :, base + 2]

            ax_3d.plot(xk, yk, zk, color=color, linestyle="-", alpha=0.4, linewidth=3)
            ax_3d.plot(xt, yt, zt, color=color, linestyle="-", linewidth=3)
            ax_3d.plot(xp, yp, zp, color=color, linestyle="--", linewidth=3)
            # 连接 Known 末点与 True / Pred 首点
            ax_3d.plot([xk[-1], xt[0]], [yk[-1], yt[0]], [zk[-1], zt[0]],
                       color=color, linestyle="-", linewidth=3)
            ax_3d.plot([xk[-1], xp[0]], [yk[-1], yp[0]], [zk[-1], zp[0]],
                       color=color, linestyle="--", linewidth=3)

        # 3D legend
        ax_3d.plot([], [], [], color="gray", linestyle="-", alpha=0.4,
                   linewidth=3, label="Known")
        ax_3d.plot([], [], [], color="gray", linestyle="-", linewidth=3,
                   label="True")
        ax_3d.plot([], [], [], color="gray", linestyle="--", linewidth=3,
                   label="Predicted")
        ax_3d.legend(loc="best")
        ax_3d.set_xlabel("$x$ (km)", labelpad=20)
        ax_3d.set_ylabel("$y$ (km)", labelpad=20)
        ax_3d.set_zlabel("$z$ (km)", labelpad=20) # type: ignore
        ax_3d.tick_params(pad=12)
        ax_3d.set_title("3D Trajectories")

        # ── 右侧：位置分量子图 ──
        for t in range(n_targets):
            base = t * 6
            color = TARGET_COLORS[t]

            for c, (axis_label, offset) in enumerate([("x", 0), ("y", 1), ("z", 2)]):
                ax = fig.add_subplot(gs[t, c + 1])

                ax.plot(time_known, X_raw[sample_idx, :, base + offset],
                        color=color, linestyle="-", alpha=0.5, linewidth=3)
                ax.plot(time_future, gt_raw[sample_idx, :, base + offset],
                        color=color, linestyle="-", linewidth=3)
                ax.plot(time_future, preds_raw[sample_idx, :, base + offset],
                        color=color, linestyle="--", linewidth=3)
                # 连接 Known 末点与 True / Pred 首点
                last_known = X_raw[sample_idx, -1, base + offset]
                first_true = gt_raw[sample_idx, 0, base + offset]
                first_pred = preds_raw[sample_idx, 0, base + offset]
                ax.plot([time_known[-1], time_future[0]], [last_known, first_true],
                        color=color, linestyle="-", linewidth=3)
                ax.plot([time_known[-1], time_future[0]], [last_known, first_pred],
                        color=color, linestyle="--", linewidth=3)

                # Divider between known and future
                ax.axvline(x=CW_DT_H * (INPUT_STEPS + 0.5), color="gray", linestyle=":",
                           alpha=0.5, linewidth=0.8)

                ax.set_ylabel(f"${axis_label}$ (km)")
                if t == n_targets - 1:
                    ax.set_xlabel("$t$ (s)")
                if t == 0:
                    ax.set_title(f"${axis_label}$")
                ax.tick_params()
                ax.grid(True, alpha=0.25)

        # ── 总标题 ──
        fig.suptitle(f"Top-{rank} | Terminal distance: {dist_val:.4f} km | $N$ = {n_targets}",
                     fontweight="bold", y=0.99)

        # ── 保存 SVG ──
        save_name = f"top{rank:02d}_dist{dist_val:.4f}_N{n_targets}.png"
        save_path = os.path.join(output_dir, save_name)
        fig.savefig(save_path, dpi=72)
        plt.close(fig)

        # ── 保存对应 .mat 数据文件 ──
        mat_name = save_name.replace(".png", ".mat")
        mat_path = os.path.join(output_dir, mat_name)
        mat_data = {
            "time_known": time_known,           # (10,) s
            "time_future": time_future,         # (10,) s
            "N": n_targets,                     # scalar
        }
        for t in range(n_targets):
            base = t * 6
            for comp, comp_name in enumerate(["x", "y", "z"]):
                mat_data[f"{comp_name}_known_{t+1}"] = X_raw[sample_idx, :, base + comp]
                mat_data[f"{comp_name}_true_{t+1}"] = gt_raw[sample_idx, :, base + comp]
                mat_data[f"{comp_name}_pred_{t+1}"] = preds_raw[sample_idx, :, base + comp]
        sio.savemat(mat_path, mat_data)
        print(f"  [{rank}/{len(top_indices)}] 已保存: {save_name} 及 {mat_name}")


def compute_terminal_distance(pred_raw, gt_raw, masks):
    """计算每个样本预测末端与真实末端的平均 3D 距离（仅有效目标，单位 km）。

    Args:
        pred_raw: (N, 10, 24) 预测轨迹（原始量纲）
        gt_raw:   (N, 10, 24) 真实轨迹（原始量纲）
        masks:    (N, 24) 有效特征 bool mask

    Returns:
        (N,) 各样本的平均末端距离
    """
    n_samples = pred_raw.shape[0]
    dist_list = []
    for i in range(n_samples):
        n_targets = int(masks[i].sum()) // 6
        # 末步位置索引
        last_step = -1
        max_dist = 0.0
        for tgt in range(n_targets):
            base = tgt * 6
            dx = pred_raw[i, last_step, base + 0] - gt_raw[i, last_step, base + 0]
            dy = pred_raw[i, last_step, base + 1] - gt_raw[i, last_step, base + 1]
            dz = pred_raw[i, last_step, base + 2] - gt_raw[i, last_step, base + 2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            if dist > max_dist:
                max_dist = dist
        dist_list.append(max_dist)
    return np.array(dist_list)


def predict(input_path, output_path, model_path=MODEL_SAVE_PATH,
            scaler_path=SCALER_SAVE_PATH,
            visualize=False, ground_truth_path=None, top_k=10):
    """
    加载模型和 scaler，对输入 CSV 进行推理，输出预测的 X_next。

    Args:
        input_path:       输入 CSV 文件路径（格式同 X_now.csv）
        output_path:      输出 CSV 文件路径
        model_path:       训练好的模型权重路径
        scaler_path:      保存的 scaler 路径
        visualize:        是否启用可视化
        ground_truth_path:真实未来轨迹 CSV 路径（仅 visualize=True 时使用）
        top_k:             可视化样本数量
    """
    # ── 加载 scaler ──
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler 文件不存在: {scaler_path}")
    scaler = FeatureScaler()
    scaler.load(scaler_path)
    print(f"Scaler 已加载: {scaler_path}")

    # ── 加载模型（支持多模型集成，2026-09-11）──
    ensemble_members = None
    if ENSEMBLE_MODELS:
        from utils.ensemble import load_members
        print(f"\n集成模式：加载 {len(ENSEMBLE_MODELS)} 个成员")
        ensemble_members = load_members(ENSEMBLE_MODELS, scaler, DEVICE)
        if not ensemble_members:
            raise RuntimeError("集成成员全部加载失败，请检查 ENSEMBLE_MODELS 配置。")
        print(f"集成成员加载完成：{len(ensemble_members)} 个 "
              f"（预测将在标准化空间等权平均）")
        model_type = "ensemble"
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        model = create_model(DEVICE, scaler)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        ckpt_state = checkpoint["model_state_dict"]
        model_keys = set(model.state_dict().keys())
        missing = sorted(model_keys - set(ckpt_state.keys()))
        if missing:
            raise RuntimeError(
                f"模型权重与当前架构不匹配：缺失 {len(missing)}/{len(model_keys)} 个参数"
                f"（示例 {missing[:5]}）。请确认 config.py 中 PHYSICS_ENABLED / USE_TRANSFORMER "
                f"与该权重来源一致。"
            )
        # strict=False：容忍旧 checkpoint 中多余的键（如已改为非持久化的派生 buffer）
        model.load_state_dict(ckpt_state, strict=False)
        model.eval()
        model_type = checkpoint.get("model_type", "lstm")
        term_dist_info = ""
        if "val_terminal_dist" in checkpoint:
            term_dist = checkpoint["val_terminal_dist"]
            term_dist_info = f", terminal_dist={term_dist:.4f} km"
        print(f"模型已加载: {model_path} (epoch {checkpoint['epoch']}, "
              f"val_loss={checkpoint['val_loss']:.6f}{term_dist_info}, type={model_type})")

    # ── 读取并解析输入数据 ──
    samples, masks = parse_csv(input_path)
    n_samples = len(samples)
    print(f"读取到 {n_samples} 个样本")

    X_raw = np.stack(samples, axis=0)       # (N, 10, 24)
    X_norm = scaler.transform(X_raw)
    X_tensor = torch.from_numpy(X_norm).float().to(DEVICE)
    masks_np = np.stack(masks, axis=0)       # (N, 24)
    masks_tensor = torch.from_numpy(masks_np).bool().to(DEVICE)

    # ── 分批推理 ──
    batch_size = 256
    all_preds = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = X_tensor[i:i + batch_size]
            batch_mask = masks_tensor[i:i + batch_size]

            if ensemble_members is not None:
                from utils.ensemble import predict_batch
                pred = predict_batch(ensemble_members, batch, batch_mask)
            elif PHYSICS_ENABLED:
                # ⚠️ 2026-09-10 修正：原先漏传 mask，导致 forward 内 mask 默认为全 True，
                # dv_global 被除以 max_N(=4) 而非实际 N，条件向量与训练/验证口径不一致
                # （实测 N=2 样本位置偏差平均 0.271 km、最大 2.422 km）。
                pred, _ = model(batch, return_dv=True, mask=batch_mask)
            else:
                pred = model(batch)

            all_preds.append(pred.cpu().numpy())

    preds_norm = np.concatenate(all_preds, axis=0)   # (N, 10, 24)
    preds_raw = scaler.inverse_transform(preds_norm)

    # ── 还原为嵌套列表字符串格式 ──
    rows = []
    for i in range(n_samples):
        mask = masks[i]
        n_features = mask.sum()
        valid_pred = preds_raw[i, :, :n_features]    # (10, N*6)
        inner_lists = []
        for t in range(OUTPUT_STEPS):
            step_vals = valid_pred[t].tolist()
            formatted = [f"{v:.4f}" for v in step_vals]
            inner_lists.append("[" + ", ".join(formatted) + "]")
        row_str = "[" + ", ".join(inner_lists) + "]"
        rows.append(row_str)

    # ── 保存 ──
    df = pd.DataFrame({"X_next": rows})
    df.to_csv(output_path, index=False)
    print(f"预测结果已保存至: {output_path} ({n_samples} 个样本)")

    # ── 可视化最佳样本 ──
    if visualize:
        if ground_truth_path is None:
            ground_truth_path = os.path.join(DATA_DIR, "X_next.csv")

        if not os.path.exists(ground_truth_path):
            print(f"警告: 真实轨迹文件不存在 ({ground_truth_path})，跳过可视化。")
            return

        print(f"\n加载真实轨迹: {ground_truth_path}")
        gt_samples, gt_masks = parse_csv(ground_truth_path)
        if len(gt_samples) != n_samples:
            print(f"警告: 真实样本数 ({len(gt_samples)}) 与输入样本数 "
                  f"({n_samples}) 不一致，跳过可视化。")
            return

        gt_raw = np.stack(gt_samples, axis=0)           # (N, 10, 24)

        # 计算各样本预测末端与真实末端的平均 3D 距离
        dist_array = compute_terminal_distance(preds_raw, gt_raw, masks_np)

        # 按 N（目标数）分组，每组各取 top-k/3 个最佳样本
        n_targets_per_sample = masks_np.sum(axis=1) // 6  # (N,)
        top_indices_list = []
        for n_val in [2, 3, 4]:
            group_mask = n_targets_per_sample == n_val
            group_indices = np.where(group_mask)[0]
            if len(group_indices) == 0:
                continue
            n_select = min(top_k // 3 if top_k >= 3 else max(1, top_k), len(group_indices))
            # 每组内按末端距离升序取前 n_select 个（距离越小越好）
            group_order = np.argsort(dist_array[group_indices])[:n_select]
            selected = group_indices[group_order]
            top_indices_list.append(selected)
            print(f"  N={n_val}: {len(group_indices)} 个样本, "
                  f"末端距离范围 [{dist_array[group_indices].min():.4f}, "
                  f"{dist_array[group_indices].max():.4f}] km, "
                  f"选取 {n_select} 个: {dist_array[selected]}")
        top_indices = np.concatenate(top_indices_list)
        # 最终按末端距离升序统一排序
        final_order = np.argsort(dist_array[top_indices])
        top_indices = top_indices[final_order]

        vis_dir = os.path.join(OUTPUT_DIR, "best_predictions")
        os.makedirs(vis_dir, exist_ok=True)
        # 清空历史图（.png/.mat/.svg）。
        # ⚠️ 2026-09-10 修正：原实现为 endswith(('.png','.mat','.png'))——'.png' 重复、漏掉 .svg，
        # 导致旧 SVG 永久累积；且 os.remove 无容错，在受限（走回收站）环境中会直接崩溃。
        removed, failed = 0, 0
        for fname in os.listdir(vis_dir):
            if fname.lower().endswith((".png", ".mat", ".svg")):
                try:
                    os.remove(os.path.join(vis_dir, fname))
                    removed += 1
                except OSError:
                    failed += 1
        if failed:
            print(f"  注意: {failed} 个历史文件无法删除（可能被占用），将直接覆盖同名文件")
        if removed:
            print(f"  已清理 {removed} 个历史图")

        print(f"\n生成最佳预测可视化图，保存至: {vis_dir}")
        visualize_best_predictions(
            X_raw, preds_raw, gt_raw, masks_np,
            dist_array, top_indices, vis_dir,
        )
        print(f"可视化完成，共 {len(top_indices)} 张图。")


def main():
    parser = argparse.ArgumentParser(description="轨迹预测推理")
    parser.add_argument("--input", type=str, default=None,
                        help="输入 CSV 文件路径（默认使用测试集第一个样本演示）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 CSV 文件路径（默认保存到 Dataset/X_pred.csv，供 IntentRecognition_V2 消费）")
    parser.add_argument("--model", type=str, default=MODEL_SAVE_PATH,
                        help="模型权重路径")
    parser.add_argument("--scaler", type=str, default=SCALER_SAVE_PATH,
                        help="Scaler 路径")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="启用最佳预测样本可视化（默认开启）")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize",
                        help="禁用可视化")
    parser.add_argument("--ground-truth", type=str, default=None,
                        help="真实未来轨迹 CSV 路径（默认 Dataset_new2/X_next.csv）")
    parser.add_argument("--top-k", type=int, default=30,
                        help="可视化最佳样本数量（默认 30，N=2/3/4 各 10 个）")
    args = parser.parse_args()

    # 默认输入输出路径
    input_path = args.input
    if input_path is None:
        input_path = os.path.join(DATA_DIR, "X_now.csv")
        print(f"未指定输入文件，使用默认: {input_path}")

    output_path = args.output
    if output_path is None:
        # X_pred.csv 是下游 IntentRecognition_V2 的输入，统一存放于工作空间顶层 Dataset/，
        # 避免各项目各存一份、版本不同步（2026-09-13 归并）
        os.makedirs(DATA_DIR, exist_ok=True)
        output_path = os.path.join(DATA_DIR, "X_pred.csv")

    ground_truth_path = args.ground_truth
    if ground_truth_path is None:
        ground_truth_path = os.path.join(DATA_DIR, "X_next.csv")

    predict(input_path, output_path, args.model, args.scaler,
            visualize=args.visualize,
            ground_truth_path=ground_truth_path,
            top_k=args.top_k)


if __name__ == "__main__":
    main()
