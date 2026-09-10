"""
读取 output/training_history.mat 并分别绘制各项训练参数随 epoch 的变化曲线。
每项指标保存为独立的 SVG 文件到 output/training_history/ 目录。

用法:
    python plot_training_history.py                              # 全部绘制
    python plot_training_history.py --path some/path.mat         # 指定文件
    python plot_training_history.py --show                       # 弹窗显示
    python plot_training_history.py --output-dir my/plots        # 指定输出目录
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

import argparse
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 字体配置 ──
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 22,
})

# ── 颜色方案 ──
C_TRAIN = "#1f77b4"
C_VAL = "#d62728"
C_PRED = "#2ca02c"
C_PHYSICS = "#ff7f0e"
C_MODE = "#9467bd"
C_GOLD = "#d4a017"
ALPHA_FILL = 0.15

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "training_history")


def squeeze_mat(data):
    """将 .mat 加载的二维数组 squeeze 为一维，标量转为 Python 标量。"""
    out = {}
    for k, v in data.items():
        if k.startswith("__"):
            continue
        a = np.asarray(v)
        if a.ndim == 2 and a.shape[0] == 1:
            a = a[0]
        if a.ndim == 1 and a.shape[0] == 1:
            a = a.item()
        out[k] = a
    return out


def load_history(mat_path):
    """加载 training_history.mat 并返回解压后的字典。"""
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"文件不存在: {mat_path}")
    raw = sio.loadmat(mat_path)
    data = squeeze_mat(raw)
    print(f"已加载: {mat_path}")
    print(f"  epoch 数: {len(data.get('epoch', []))}")
    if "best_epoch" in data:
        bv = data.get("best_val_loss", None)
        if bv is not None:
            print(f"  最佳 epoch: {data['best_epoch']}, best_val_loss: {bv:.6f}")
        else:
            print(f"  最佳 epoch: {data['best_epoch']}")
    return data


def get_epoch_range(data):
    """确定有效 epoch 范围（去除末尾未训练的 0 值 epoch）。"""
    epoch = data["epoch"]
    val_total = data.get("val_total", epoch * 0)
    nonzero = np.where(val_total > 1e-10)[0]
    last_valid = nonzero[-1] + 1 if len(nonzero) > 0 else len(epoch)
    return epoch[:last_valid]


def mark_best_epoch(ax, data, epoch=None, values=None):
    """在图上标记最佳 epoch 竖线。

    Args:
        ax: matplotlib Axes
        data: 完整数据字典（用于读取 best_epoch）
        epoch: 若提供，则从 values 中 argmin 计算最佳 epoch
        values: 与 epoch 对应的数值序列
    """
    if epoch is not None and values is not None:
        local_best = epoch[np.argmin(values)]
        ax.axvline(x=local_best, color="gray", linestyle=":", alpha=0.7, linewidth=1.2)
    elif "best_epoch" in data:
        be = int(data["best_epoch"])
        ax.axvline(x=be, color="gray", linestyle=":", alpha=0.7, linewidth=1.2)


def save_or_show(fig, name, output_dir, show, dpi):
    """保存 SVG 或弹窗显示。"""
    if show:
        plt.show()
    else:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, name)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  已保存: {path}")
    plt.close(fig)


# ============================================================
#  各子图绘制函数（每个独立 fig）
# ============================================================

def plot_loss(data, epoch, output_dir, show, dpi):
    """图1：总损失 + 预测损失"""
    fig, ax = plt.subplots(figsize=(10, 6))
    last = len(epoch)
    ax.plot(epoch, data["val_total"][:last], color=C_VAL, linewidth=2, label="Val total")
    if "train_total" in data:
        ax.plot(epoch, data["train_total"][:last], color=C_TRAIN,
                linewidth=1.5, alpha=0.7, label="Train total")
    if "val_pred" in data:
        ax.plot(epoch, data["val_pred"][:last], color=C_PRED,
                linewidth=2, linestyle="--", label="Val pred (MSE)")
    if "train_pred" in data:
        ax.plot(epoch, data["train_pred"][:last], color=C_PRED,
                linewidth=1.5, alpha=0.5, linestyle=":", label="Train pred (MSE)")
    if "train_terminal" in data and np.any(data["train_terminal"][:last] > 0):
        ax.plot(epoch, data["train_terminal"][:last], color=C_GOLD,
                linewidth=1.5, linestyle="-.", alpha=0.7, label="Train terminal (last-step pos MSE)")

    if "best_epoch" in data:
        be = int(data["best_epoch"])
        bv = float(data.get("best_val_loss", data["val_total"][be - 1]))
        ax.axvline(x=be, color="gray", linestyle=":", alpha=0.7, linewidth=1.2)
        ax.plot(be, bv, marker="*", color=C_GOLD, markersize=16, zorder=5,
                label=f"Best epoch {be} (loss={bv:.4f})")

    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Total & Prediction Loss")
    ax.legend(fontsize=14); ax.grid(True, alpha=0.25)
    ax.set_yscale("log")
    fig.tight_layout()
    save_or_show(fig, "loss.png", output_dir, show, dpi)


def plot_physics_mode_loss(data, epoch, output_dir, show, dpi):
    """图2：物理 + 模式损失分量 + λ 权重"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    last = len(epoch)

    if "val_physics" in data and np.any(data["val_physics"][:last] > 0):
        ax1.plot(epoch, data["val_physics"][:last], color=C_PHYSICS,
                 linewidth=2, label="Val physics")
    if "train_physics" in data and np.any(data["train_physics"][:last] > 0):
        ax1.plot(epoch, data["train_physics"][:last], color=C_PHYSICS,
                 linewidth=1.5, alpha=0.5, linestyle=":", label="Train physics")
    if "val_mode" in data and np.any(data["val_mode"][:last] > 1e-10):
        ax1.plot(epoch, data["val_mode"][:last], color=C_MODE,
                 linewidth=2, label=r"Val $\Delta v$ consistency")
    if "train_mode" in data and np.any(data["train_mode"][:last] > 1e-10):
        ax1.plot(epoch, data["train_mode"][:last], color=C_MODE,
                 linewidth=1.5, alpha=0.5, linestyle=":", label=r"Train $\Delta v$ consistency")

    mark_best_epoch(ax1, data)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Physics & Mode Loss Components")
    ax1.legend(fontsize=14, loc="upper left"); ax1.grid(True, alpha=0.25)
    ax1.set_yscale("log")

    # λ 权重（右轴）
    ax2 = ax1.twinx()
    has_lambda = False
    if "train_lambda_p" in data and np.any(data["train_lambda_p"][:last] > 0):
        ax2.plot(epoch, data["train_lambda_p"][:last], color=C_PHYSICS,
                 linestyle="--", linewidth=1.5, alpha=0.7, label=r"$\lambda_{physics}$")
        has_lambda = True
    if "train_lambda_m" in data and np.any(data["train_lambda_m"][:last] > 0):
        ax2.plot(epoch, data["train_lambda_m"][:last], color=C_MODE,
                 linestyle="--", linewidth=1.5, alpha=0.7, label=r"$\lambda_{mode}$")
        has_lambda = True
    if "train_lambda_t" in data and np.any(data["train_lambda_t"][:last] > 0):
        ax2.plot(epoch, data["train_lambda_t"][:last], color=C_GOLD,
                 linestyle="--", linewidth=1.5, alpha=0.7, label=r"$\lambda_{terminal}$")
        has_lambda = True
    if has_lambda:
        ax2.set_ylabel(r"Weight $\lambda$")
        ax2.legend(fontsize=14, loc="upper right")

    fig.tight_layout()
    save_or_show(fig, "physics_mode_loss.png", output_dir, show, dpi)


def plot_terminal_distance(data, epoch, output_dir, show, dpi):
    """图3：验证末端距离"""
    fig, ax = plt.subplots(figsize=(10, 6))
    last = len(epoch)

    dist_metrics = {
        "val_terminal_dist_mean": ("Mean", C_TRAIN),
        "val_terminal_dist_median": ("Median", C_PRED),
        "val_terminal_dist_min": ("Min", C_PHYSICS),
        "val_terminal_dist_max": ("Max", C_VAL),
    }
    for key, (label, color) in dist_metrics.items():
        if key in data:
            ax.plot(epoch, data[key][:last], color=color, linewidth=2, label=label)

    if "val_terminal_dist_mean" in data and "val_terminal_dist_std" in data:
        mean_v = data["val_terminal_dist_mean"][:last]
        std_v = data["val_terminal_dist_std"][:last]
        ax.fill_between(epoch, mean_v - std_v, mean_v + std_v,
                         color=C_TRAIN, alpha=ALPHA_FILL, label=r"Mean $\pm$ Std")

    mark_best_epoch(ax, data, epoch=epoch, values=data.get("val_terminal_dist_mean", None)[:last])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Distance (km)")
    ax.set_title("Validation Terminal Distance")
    ax.legend(ncol=2, fontsize=14); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_or_show(fig, "terminal_distance.png", output_dir, show, dpi)


def plot_success_rate(data, epoch, output_dir, show, dpi):
    """图4：验证成功率"""
    fig, ax = plt.subplots(figsize=(10, 6))
    last = len(epoch)

    if "val_success_rate_1km" in data:
        ax.plot(epoch, data["val_success_rate_1km"][:last] * 100,
                color=C_PRED, linewidth=2, label=r"$<1$ km")
    if "val_success_rate_100m" in data:
        ax.plot(epoch, data["val_success_rate_100m"][:last] * 100,
                color=C_TRAIN, linewidth=2, label=r"$<100$ m")

    mark_best_epoch(ax, data, epoch=epoch, values=data.get("val_terminal_dist_mean", None)[:last])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Success rate (%)")
    ax.set_title("Validation Success Rate")
    ax.legend(fontsize=14); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_or_show(fig, "success_rate.png", output_dir, show, dpi)


def plot_dv_limit_rate(data, epoch, output_dir, show, dpi):
    """图5：Δv 超限率"""
    fig, ax = plt.subplots(figsize=(10, 6))
    last = len(epoch)

    if "val_dv_over_limit_rate" in data:
        ax.plot(epoch, data["val_dv_over_limit_rate"][:last] * 100,
                color=C_VAL, linewidth=2, label=r"$\Delta v$ over limit")

    mark_best_epoch(ax, data, epoch=epoch, values=data.get("val_terminal_dist_mean", None)[:last])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Rate (%)")
    ax.set_title(r"$\Delta v$ Over Limit Rate")
    ax.legend(fontsize=14); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_or_show(fig, "dv_over_limit_rate.png", output_dir, show, dpi)


def plot_lr_and_tf(data, epoch, output_dir, show, dpi):
    """图6：学习率 + Teacher Forcing 比例（两个子图并列）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    last = len(epoch)

    # 左：学习率
    if "train_lr" in data:
        ax1.plot(epoch, data["train_lr"][:last], color=C_MODE, linewidth=2, label="Learning rate")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Learning rate")
    ax1.set_title("Learning Rate Schedule")
    ax1.legend(fontsize=14); ax1.grid(True, alpha=0.25)
    ax1.set_yscale("log")

    # 右：TF 比例
    if "train_tf_ratio" in data:
        ax2.plot(epoch, data["train_tf_ratio"][:last], color=C_PHYSICS,
                 linewidth=2, label="Teacher forcing ratio")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("TF ratio")
    ax2.set_title("Teacher Forcing Schedule")
    ax2.legend(fontsize=14); ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    save_or_show(fig, "lr_and_tf.png", output_dir, show, dpi)


def plot_grad_norm_and_time(data, epoch, output_dir, show, dpi):
    """图7：梯度范数 + 每轮训练时间（两个子图并列）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    last = len(epoch)

    # 左：梯度范数
    if "train_grad_norm" in data:
        ax1.plot(epoch, data["train_grad_norm"][:last], color=C_TRAIN,
                 linewidth=2, label="Gradient norm")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Gradient norm")
    ax1.set_title("Gradient Norm")
    ax1.legend(fontsize=14); ax1.grid(True, alpha=0.25)

    # 右：训练时间
    if "train_time" in data:
        ax2.plot(epoch, data["train_time"][:last], color=C_PRED,
                 linewidth=2, label="Time per epoch (s)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Time (s)")
    ax2.set_title("Training Time per Epoch")
    ax2.legend(fontsize=14); ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    save_or_show(fig, "grad_norm_and_time.png", output_dir, show, dpi)


def plot_dv_magnitude(data, epoch, output_dir, show, dpi):
    """图8：Δv 幅值 + 超限率（双 y 轴）"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    last = len(epoch)

    if "val_dv_mag_mean" in data:
        ax1.plot(epoch, data["val_dv_mag_mean"][:last], color=C_TRAIN,
                 linewidth=2, label=r"$\Delta v$ mean (m/s)")
    if "val_dv_mag_max" in data:
        ax1.plot(epoch, data["val_dv_mag_max"][:last], color=C_VAL,
                 linewidth=2, label=r"$\Delta v$ max (m/s)")

    mark_best_epoch(ax1, data)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel(r"$\Delta v$ (m/s)")
    ax1.set_title(r"$\Delta v$ Magnitude Estimation")
    ax1.legend(fontsize=14, loc="upper left"); ax1.grid(True, alpha=0.25)

    # 右轴：超限率
    if "val_dv_over_limit_rate" in data:
        ax2 = ax1.twinx()
        ax2.plot(epoch, data["val_dv_over_limit_rate"][:last] * 100,
                 color=C_PHYSICS, linewidth=1.5, linestyle="--", label="Over limit rate")
        ax2.set_ylabel("Over limit rate (%)")
        ax2.legend(fontsize=14, loc="upper right")

    fig.tight_layout()
    save_or_show(fig, "dv_magnitude.png", output_dir, show, dpi)


def plot_cw_residual(data, epoch, output_dir, show, dpi):
    """图9：CW 残差"""
    fig, ax = plt.subplots(figsize=(10, 6))
    last = len(epoch)

    has_data = False
    if "val_cw_input" in data and np.any(data["val_cw_input"][:last] > 0):
        ax.plot(epoch, data["val_cw_input"][:last], color=C_TRAIN,
                linewidth=2, label="CW input residual")
        has_data = True
    if "val_cw_pred" in data and np.any(data["val_cw_pred"][:last] > 0):
        ax.plot(epoch, data["val_cw_pred"][:last], color=C_VAL,
                linewidth=2, label="CW pred residual")
        has_data = True
    if "val_dv_change" in data and np.any(data["val_dv_change"][:last] > 1e-10):
        ax.plot(epoch, data["val_dv_change"][:last], color=C_PRED,
                linewidth=2, label=r"$\Delta v$ change rate")
        has_data = True

    if has_data:
        mark_best_epoch(ax, data, epoch=epoch, values=data.get("val_terminal_dist_mean", None)[:last])
        ax.set_xlabel("Epoch"); ax.set_ylabel("Residual")
        ax.set_title("CW Consistency & $\Delta v$ Stability")
        ax.legend(fontsize=14); ax.grid(True, alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No CW data available", transform=ax.transAxes,
                ha="center", va="center", fontsize=18, color="gray")

    fig.tight_layout()
    save_or_show(fig, "cw_residual.png", output_dir, show, dpi)


# ============================================================
#  主调度
# ============================================================

PLOT_FUNCTIONS = [
    ("Loss Curves", plot_loss),
    ("Physics & Mode Loss", plot_physics_mode_loss),
    ("Terminal Distance", plot_terminal_distance),
    ("Success Rate", plot_success_rate),
    ("Δv Over Limit Rate", plot_dv_limit_rate),
    ("LR & Teacher Forcing", plot_lr_and_tf),
    ("Gradient Norm & Time", plot_grad_norm_and_time),
    ("Δv Magnitude", plot_dv_magnitude),
    ("CW Residual", plot_cw_residual),
]


def main():
    parser = argparse.ArgumentParser(description="分别绘制训练历史各项曲线")
    parser.add_argument("--path", type=str, default=None,
                        help="training_history.mat 路径")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="SVG 输出目录（默认 output/training_history/）")
    parser.add_argument("--show", action="store_true",
                        help="弹窗显示而非保存")
    parser.add_argument("--dpi", type=int, default=150,
                        help="图片分辨率（默认 150）")
    parser.add_argument("--list", nargs="+", type=int, default=None,
                        help="只绘制指定图号，如 --list 1 3 5")
    args = parser.parse_args()

    # 默认路径
    mat_path = args.path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "output", "training_history.mat")

    out_dir = args.output_dir or OUTPUT_DIR

    data = load_history(mat_path)
    epoch = get_epoch_range(data)

    print(f"有效 epoch 数: {len(epoch)}")
    print(f"输出目录: {out_dir}")
    print(f"共 {len(PLOT_FUNCTIONS)} 张图，开始绘制...\n")

    selected = args.list if args.list is not None else range(1, len(PLOT_FUNCTIONS) + 1)

    for idx in selected:
        if idx < 1 or idx > len(PLOT_FUNCTIONS):
            print(f"  [警告] 跳过无效图号 {idx}")
            continue
        name, func = PLOT_FUNCTIONS[idx - 1]
        print(f"[{idx}/{len(PLOT_FUNCTIONS)}] {name}")
        func(data, epoch, out_dir, args.show, args.dpi)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
