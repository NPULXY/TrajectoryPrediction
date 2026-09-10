"""
从 output/best_predictions/ 下的 .mat 文件读取数据并重绘最佳预测可视化图。

用法:
    python plot_best_predictions.py                          # 全部重绘
    python plot_best_predictions.py --dir some/other/path    # 指定路径
    python plot_best_predictions.py --top-k 5                # 只重绘前 5 个
    python plot_best_predictions.py --show                   # 屏幕显示（而不是保存 SVG）
"""

import os
import sys

# ── conda MKL DLL 搜索路径修复 (Windows) ──
# conda 构建的 numpy 依赖 MKL，其 DLL 位于 <anaconda3>/Library/bin。
# 如果该目录不在 PATH 中（如 Git Bash 未激活 conda），numpy 会因找不到
# mkl_rt.*.dll 而 ImportError。此处主动检测并补充到 PATH 中。
if sys.platform == "win32":
    _conda_lib_bin_candidates = [
        os.environ.get("CONDA_PREFIX", ""),                          # 已激活的 conda 环境
        os.path.join(os.environ.get("SystemDrive", "C:"), os.sep,
                     "Users", os.environ.get("USERNAME", "Hasee"),
                     "anaconda3"),                                   # 默认 base
    ]
    for _prefix in _conda_lib_bin_candidates:
        _lib_bin = os.path.join(_prefix, "Library", "bin") if _prefix else ""
        if _lib_bin and os.path.isdir(_lib_bin):
            # 只在尚未包含时添加，避免重复
            if _lib_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = _lib_bin + os.pathsep + os.environ.get("PATH", "")

import argparse
import glob
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── 字体配置（与 predict.py 一致） ──
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
TARGET_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]


def load_mat_data(mat_path):
    """加载 .mat 文件，返回数据字典。"""
    data = sio.loadmat(mat_path)
    # scipy.io.savemat 保存时标量变成二维数组，Python 侧 squeeze 掉单维度
    for k, v in data.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape == (1, 1):
            data[k] = v.item()
        elif isinstance(v, np.ndarray) and v.ndim >= 2:
            data[k] = v.squeeze()
    return data


def plot_from_mat(mat_path, output_dir=None, show=False, dpi=150):
    """根据单个 .mat 文件重绘轨迹图。

    Args:
        mat_path:     .mat 文件路径
        output_dir:   图片保存目录（None 时与 mat 同路径）
        show:         是否弹窗显示（否则保存 SVG）
        dpi:          图片分辨率
    """
    data = load_mat_data(mat_path)
    n_targets = int(data["N"])
    time_known = data["time_known"]  # (10,) s
    time_future = data["time_future"]  # (10,) s

    # ── 从文件名提取 top 编号和末端距离 ──
    basename = os.path.splitext(os.path.basename(mat_path))[0]
    parts = basename.split("_")
    rank_str = parts[0].replace("top", "")  # "01"
    dist_val = None
    for p in parts:
        if p.startswith("dist"):
            try:
                dist_val = float(p.replace("dist", ""))
            except ValueError:
                pass
            break

    # ── 创建图形 ──
    fig_height = max(12, 5.0 * n_targets)
    fig_width = fig_height * 4.0 / 3.0
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        n_targets, 4, figure=fig,
        width_ratios=[2.0, 1.0, 1.0, 1.0],
        hspace=0.55, wspace=0.40,
    )

    # ── 左侧：3D 组合轨迹图 ──
    ax_3d = fig.add_subplot(gs[:, 0], projection="3d")

    for t in range(n_targets):
        color = TARGET_COLORS[t]
        suffix = f"_{t+1}"

        xk = data[f"x_known{suffix}"]
        yk = data[f"y_known{suffix}"]
        zk = data[f"z_known{suffix}"]
        xt = data[f"x_true{suffix}"]
        yt = data[f"y_true{suffix}"]
        zt = data[f"z_true{suffix}"]
        xp = data[f"x_pred{suffix}"]
        yp = data[f"y_pred{suffix}"]
        zp = data[f"z_pred{suffix}"]

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
    ax_3d.set_zlabel("$z$ (km)", labelpad=20)
    ax_3d.tick_params(pad=12)
    ax_3d.set_title("3D Trajectories")

    # ── 右侧：位置分量子图 ──
    for t in range(n_targets):
        color = TARGET_COLORS[t]
        suffix = f"_{t+1}"

        for c, (axis_label, offset_key) in enumerate([("x", "x"), ("y", "y"), ("z", "z")]):
            ax = fig.add_subplot(gs[t, c + 1])

            known_arr = data[f"{offset_key}_known{suffix}"]
            true_arr = data[f"{offset_key}_true{suffix}"]
            pred_arr = data[f"{offset_key}_pred{suffix}"]

            ax.plot(time_known, known_arr,
                    color=color, linestyle="-", alpha=0.5, linewidth=3)
            ax.plot(time_future, true_arr,
                    color=color, linestyle="-", linewidth=3)
            ax.plot(time_future, pred_arr,
                    color=color, linestyle="--", linewidth=3)
            # 连接 Known 末点与 True / Pred 首点
            ax.plot([time_known[-1], time_future[0]], [known_arr[-1], true_arr[0]],
                    color=color, linestyle="-", linewidth=3)
            ax.plot([time_known[-1], time_future[0]], [known_arr[-1], pred_arr[0]],
                    color=color, linestyle="--", linewidth=3)

            # Divider between known and future
            ax.axvline(x=630, color="gray", linestyle=":",
                       alpha=0.5, linewidth=0.8)

            ax.set_ylabel(f"${axis_label}$ (km)")
            if t == n_targets - 1:
                ax.set_xlabel("$t$ (s)")
            if t == 0:
                ax.set_title(f"${axis_label}$")
            ax.tick_params()
            ax.grid(True, alpha=0.25)

    # ── 总标题 ──
    title_parts = [f"Top-{int(rank_str)}"]
    if dist_val is not None:
        title_parts.append(f"Terminal distance: {dist_val:.4f} km")
    title_parts.append(f"$N$ = {n_targets}")
    fig.suptitle(" | ".join(title_parts), fontweight="bold", y=0.99)

    # ── 保存或显示 ──
    if show:
        plt.show()
    else:
        out_dir = output_dir or os.path.dirname(mat_path)
        os.makedirs(out_dir, exist_ok=True)
        svg_name = basename + ".png"
        svg_path = os.path.join(out_dir, svg_name)
        fig.savefig(svg_path, dpi=dpi)
        print(f"  已保存: {svg_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="从 .mat 文件重绘最佳预测轨迹图")
    parser.add_argument("--dir", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "output", "best_predictions"),
                        help="包含 .mat 文件的目录（默认 output/best_predictions）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="图片输出目录（默认与 --dir 相同）")
    parser.add_argument("--top-k", type=int, default=0,
                        help="只处理前 K 个 .mat 文件（按文件名排序，0=全部）")
    parser.add_argument("--show", action="store_true",
                        help="弹窗显示而不是保存图片")
    parser.add_argument("--dpi", type=int, default=150,
                        help="图片分辨率（默认 150）")
    args = parser.parse_args()

    # ── 收集 .mat 文件 ──
    mat_pattern = os.path.join(args.dir, "*.mat")
    mat_files = sorted(glob.glob(mat_pattern))
    if not mat_files:
        print(f"未找到 .mat 文件: {mat_pattern}")
        return

    if args.top_k > 0:
        mat_files = mat_files[:args.top_k]

    print(f"找到 {len(mat_files)} 个 .mat 文件，开始重绘...")
    for i, mp in enumerate(mat_files, 1):
        print(f"[{i}/{len(mat_files)}] {os.path.basename(mp)}")
        plot_from_mat(mp, output_dir=args.output_dir, show=args.show, dpi=args.dpi)

    print("全部完成。")


if __name__ == "__main__":
    main()
