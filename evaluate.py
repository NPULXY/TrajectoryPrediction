"""
评估脚本 —— 在测试集上计算指标并生成可视化图表。
支持原版 LSTM 和物理信息条件 LSTM 两种模式的评估。
用法: python evaluate.py
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

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from config import (
    DATA_DIR, OUTPUT_DIR, MODEL_SAVE_PATH, SCALER_SAVE_PATH, DEVICE, BATCH_SIZE,
    POS_INDICES, VEL_INDICES, MAX_N, MAX_DIM,
    PHYSICS_ENABLED, CW_N, CW_DT_H, DELTAV_LIMIT,
)
from utils.data_loader import (
    load_and_split, create_dataloaders, masked_mse_loss,
    FeatureScaler, compute_metrics,
)
from models.model import create_model


# 设置中文字体与全局字号
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
    "figure.titlesize": 20,
})


def compute_cw_metrics(preds_raw, masks, dv_all=None):
    """
    计算物理一致性指标：CW 单步递推残差（分离位置/速度）与**脉冲 Δv** 统计。

    ⚠️ 2026-09-10 修正（三项）：
      1. 步长改用 CW_DT_H（实测 60 s），原硬编码 1 s；
      2. CW 残差**分离位置 (km) 与速度 (km/s) 分量**，不再混成单一范数
         —— 原实现把 km 与 km/s 混在一个欧氏范数里，单位标注无意义；
      3. Δv 改为**脉冲 Δv**（扣除 CW 自由演化），而非相邻步速度差分
         —— 后者含 60 s 自由演化贡献（约 4 m/s），会系统性高估。

    Args:
        preds_raw: (N, 10, 24) 预测轨迹（原始量纲）
        masks:     (N, 24) 有效特征掩码
        dv_all:    保留兼容；本版直接由轨迹反解，不使用该参数

    Returns:
        result(dict), cw_pos_res(list, km), cw_vel_res(list, km/s), dv_pulses(list, km/s)
    """
    from models.physics_loss import compute_cw_matrix, N_MEAN

    Phi_np = compute_cw_matrix(N_MEAN, CW_DT_H).numpy()

    cw_pos_res, cw_vel_res, dv_pulses = [], [], []

    for i in range(preds_raw.shape[0]):
        n_agents = int(masks[i].sum()) // 6
        if n_agents == 0:
            continue
        for a in range(n_agents):
            base = a * 6
            states = preds_raw[i, :, base:base + 6]      # (10, 6)
            for t in range(9):
                s_curr, s_next = states[t], states[t + 1]
                s_free = Phi_np @ s_curr
                r = s_next - s_free
                cw_pos_res.append(float(np.linalg.norm(r[:3])))          # km
                cw_vel_res.append(float(np.linalg.norm(r[3:])))          # km/s
                dv_pulses.append(float(np.linalg.norm(s_next[3:] - s_free[3:])))  # km/s

    def _stat(lst):
        if not lst:
            return 0.0, 0.0, 0.0
        return float(np.mean(lst)), float(np.std(lst)), float(np.max(lst))

    p_mu, p_sd, p_mx = _stat(cw_pos_res)
    v_mu, v_sd, v_mx = _stat(cw_vel_res)
    d_mu, d_sd, d_mx = _stat(dv_pulses)

    result = {
        "cw_pos_residual_mean": p_mu, "cw_pos_residual_std": p_sd, "cw_pos_residual_max": p_mx,
        "cw_vel_residual_mean": v_mu, "cw_vel_residual_std": v_sd, "cw_vel_residual_max": v_mx,
        "dv_pulse_mean": d_mu, "dv_pulse_std": d_sd, "dv_pulse_max": d_mx,
    }
    if dv_pulses:
        arr = np.array(dv_pulses)
        result["dv_pulse_over_limit_rate"] = float(np.mean(arr > (DELTAV_LIMIT / 1000.0)))

    return result, cw_pos_res, cw_vel_res, dv_pulses


def evaluate_model(model, loader, scaler, device):
    """在测试集上计算指标（还原到原始量纲后计算）。"""
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []
    all_dv = []

    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            if PHYSICS_ENABLED:
                pred, dv_all = model(x, return_dv=True, mask=mask)
                all_dv.append(dv_all.cpu())
            else:
                pred = model(x)

            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
            all_masks.append(mask.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)
    dv_all = torch.cat(all_dv, dim=0) if all_dv else None

    # 还原到原始量纲
    preds_np = preds.numpy()
    targets_np = targets.numpy()
    preds_raw = scaler.inverse_transform(preds_np)
    targets_raw = scaler.inverse_transform(targets_np)

    preds_t = torch.from_numpy(preds_raw)
    targets_t = torch.from_numpy(targets_raw)

    metrics = compute_metrics(preds_t, targets_t, masks)

    print("\n" + "=" * 50)
    print("测试集评估结果（原始量纲）")
    print("=" * 50)
    print(f"整体 MSE:     {metrics['MSE']:.6f}")
    print(f"整体 RMSE:    {metrics['RMSE']:.6f}")
    print(f"整体 MAE:     {metrics['MAE']:.6f}")
    print(f"位置 MSE:     {metrics['MSE_pos']:.6f}")
    print(f"位置 RMSE:    {metrics['RMSE_pos']:.6f}")
    print(f"位置 MAE:     {metrics['MAE_pos']:.6f}")
    print(f"速度 MSE:     {metrics['MSE_vel']:.8f}")
    print(f"速度 RMSE:    {metrics['RMSE_vel']:.8f}")
    print(f"速度 MAE:     {metrics['MAE_vel']:.8f}")

    cw_data = None
    # 物理一致性指标
    if PHYSICS_ENABLED:
        cw_metrics, cw_pos, cw_vel, dv_pulses = compute_cw_metrics(preds_raw, masks.numpy())
        metrics.update(cw_metrics)
        cw_data = (cw_pos, cw_vel, dv_pulses)

        print(f"\n--- 物理一致性指标（CW 步长 {CW_DT_H:.0f} s，脉冲 Δv 口径）---")
        print(f"CW 位置残差均值:   {cw_metrics['cw_pos_residual_mean']:.6f} km")
        print(f"CW 速度残差均值:   {cw_metrics['cw_vel_residual_mean']:.6f} km/s")
        print(f"脉冲 Δv 幅值均值:  {cw_metrics['dv_pulse_mean'] * 1000:.4f} m/s")
        print(f"脉冲 Δv 幅值最大:  {cw_metrics['dv_pulse_max'] * 1000:.4f} m/s")
        if 'dv_pulse_over_limit_rate' in cw_metrics:
            print(f"脉冲 Δv 超 {DELTAV_LIMIT} m/s 占比: "
                  f"{cw_metrics['dv_pulse_over_limit_rate'] * 100:.2f}%")

    return preds_raw, targets_raw, masks.numpy(), metrics, cw_data


def plot_predictions(preds, targets, masks, scaler, num_samples=5, save_dir=OUTPUT_DIR):
    """
    随机选取样本绘制真实轨迹与预测轨迹对比图。
    - 3D 位置轨迹
    - 各分量随时间的 2D 变化图
    """
    n_total = preds.shape[0]
    rng = np.random.RandomState(42)
    indices = rng.choice(n_total, size=min(num_samples, n_total), replace=False)

    for idx, sample_idx in enumerate(indices):
        pred = preds[sample_idx]       # (10, 24)
        true = targets[sample_idx]     # (10, 24)
        mask = masks[sample_idx]       # (24,)
        n_valid = mask.sum().item()    # 有效特征数
        n_agents = n_valid // 6

        fig = plt.figure(figsize=(8 * n_agents, 6 * n_agents))
        fig.suptitle(f"样本 #{sample_idx} (N={n_agents})")

        time_steps = np.arange(10)

        for agent in range(n_agents):
            base = agent * 6

            # ── 3D 位置图 ──
            ax3d = fig.add_subplot(n_agents, 4, agent * 4 + 1, projection="3d")
            ax3d.plot(true[:, base + 0], true[:, base + 1], true[:, base + 2],
                      "b-o", linewidth=3, markersize=5, label="真实")
            ax3d.plot(pred[:, base + 0], pred[:, base + 1], pred[:, base + 2],
                      "r--s", linewidth=3, markersize=5, label="预测")
            ax3d.scatter(true[0, base + 0], true[0, base + 1], true[0, base + 2],
                         c="blue", s=50, marker="o") # type: ignore
            ax3d.scatter(pred[0, base + 0], pred[0, base + 1], pred[0, base + 2],
                         c="red", s=50, marker="o") # type: ignore
            ax3d.set_xlabel("X (km)")
            ax3d.set_ylabel("Y (km)")
            ax3d.set_zlabel("Z (km)") # type: ignore
            ax3d.set_title(f"目标 {agent+1} 3D 轨迹")
            ax3d.legend()

            # ── X 分量 ──
            ax_x = fig.add_subplot(n_agents, 4, agent * 4 + 2)
            ax_x.plot(time_steps, true[:, base + 0], "b-o", linewidth=3, markersize=6, label="真实")
            ax_x.plot(time_steps, pred[:, base + 0], "r--s", linewidth=3, markersize=6, label="预测")
            ax_x.set_xlabel("时间步")
            ax_x.set_ylabel("X (km)")
            ax_x.set_title(f"目标 {agent+1} X 分量")
            ax_x.legend()
            ax_x.grid(True, alpha=0.3)

            # ── Y 分量 ──
            ax_y = fig.add_subplot(n_agents, 4, agent * 4 + 3)
            ax_y.plot(time_steps, true[:, base + 1], "b-o", linewidth=3, markersize=6, label="真实")
            ax_y.plot(time_steps, pred[:, base + 1], "r--s", linewidth=3, markersize=6, label="预测")
            ax_y.set_xlabel("时间步")
            ax_y.set_ylabel("Y (km)")
            ax_y.set_title(f"目标 {agent+1} Y 分量")
            ax_y.legend()
            ax_y.grid(True, alpha=0.3)

            # ── Z 分量 ──
            ax_z = fig.add_subplot(n_agents, 4, agent * 4 + 4)
            ax_z.plot(time_steps, true[:, base + 2], "b-o", linewidth=3, markersize=6, label="真实")
            ax_z.plot(time_steps, pred[:, base + 2], "r--s", linewidth=3, markersize=6, label="预测")
            ax_z.set_xlabel("时间步")
            ax_z.set_ylabel("Z (km)")
            ax_z.set_title(f"目标 {agent+1} Z 分量")
            ax_z.legend()
            ax_z.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sample_{sample_idx:05d}.png")
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"图表已保存: {save_path}")

        # ── 保存对应 .mat 数据文件 ──
        mat_path = save_path.replace(".png", ".mat")
        mat_data = {
            "time_steps": time_steps,           # (10,)
            "true_trajectory": true,             # (10, 24)
            "pred_trajectory": pred,             # (10, 24)
            "mask": mask,                        # (24,)
            "n_agents": n_agents,                # scalar
            "sample_idx": np.int32(sample_idx),  # scalar
        }
        sio.savemat(mat_path, mat_data)
        print(f"  └─ 数据已保存: {mat_path}")


def plot_loss_curve(log_path=os.path.join(os.path.dirname(__file__), "output", "train_log.txt"),
                    save_dir=OUTPUT_DIR):
    """
    从训练日志绘制曲线。

    ⚠️ 2026-09-10 修正：原实现以"训练总损失"为主曲线，但总损失包含 λ_t warmup
    引入的物理量纲末端损失（权重 0.01→2.0），其"上升-峰值-缓降"形状会被误读为
    训练发散。现改为三个子图：
      (1) 训练/验证**预测损失**（真正的优化目标，标准化空间，对数轴）
      (2) 验证**平均末端距离 td (km)** —— 实际的模型选择判据
      (3) 物理损失分量
    """
    if not os.path.exists(log_path):
        print(f"日志文件不存在，跳过 loss 曲线绘制: {log_path}")
        return

    data = {k: [] for k in ["train_total", "train_pred", "train_phy", "train_mode",
                            "val_pred", "val_phy", "val_td"]}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "Train:" not in line or "pred=" not in line:
                continue
            try:
                data["train_total"].append(float(line.split("Train:")[1].split("(")[0].strip()))
                data["train_pred"].append(float(line.split("pred=")[1].split(" ")[0].strip()))
                data["train_phy"].append(float(line.split("phy=")[1].split(" ")[0].strip()))
                # ⚠️ mode= 之后还紧跟 bound=，必须按空格截断；
                #    用 split(")") 会把 "mode 值 + bound 值" 一起取出导致 float() 失败
                data["train_mode"].append(float(line.split("mode=")[1].split(" ")[0].strip()))
            except (IndexError, ValueError):
                pass
            try:
                data["val_pred"].append(
                    float(line.split("Val:")[1].split("(pred=")[1].split(" ")[0].strip()))
                # 同理，Val 段中 phy= 之后还有 dvalign=
                data["val_phy"].append(
                    float(line.split("Val:")[1].split("phy=")[1].split(" ")[0].strip()))
            except (IndexError, ValueError):
                pass
            try:
                data["val_td"].append(float(line.split("td=")[1].split("km")[0].strip()))
            except (IndexError, ValueError):
                pass

    if not data["train_pred"]:
        print("无法从日志中解析 loss 数据")
        return

    n = len(data["train_pred"])
    epochs = list(range(1, n + 1))

    def align(lst):
        """长度不足时以 NaN 补齐，避免 matplotlib 维度不匹配导致绘图中断。"""
        out = list(lst)[:n]
        return out + [float("nan")] * (n - len(out))

    tr_pred, va_pred = align(data["train_pred"]), align(data["val_pred"])
    tr_phy, tr_mode = align(data["train_phy"]), align(data["train_mode"])
    va_phy, va_td = align(data["val_phy"]), align(data["val_td"])

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # ── (1) 预测损失（真正的优化目标）──
    ax = axes[0]
    ax.plot(epochs, tr_pred, "b-", linewidth=2.5, label="训练预测损失")
    ax.plot(epochs, va_pred, "r--", linewidth=2.5, label="验证预测损失")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE（标准化空间）")
    ax.set_title("预测损失（真正的优化目标）")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── (2) 验证末端距离（模型选择判据）──
    ax = axes[1]
    if not all(np.isnan(va_td)):
        ax.plot(epochs, va_td, "g-", linewidth=2.5, label="验证平均末端距离")
        bi = int(np.nanargmin(va_td))
        ax.scatter([epochs[bi]], [va_td[bi]], c="red", s=70, zorder=5,
                   label=f"最优 {va_td[bi]:.4f} km @epoch {epochs[bi]}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("末端距离 (km)")
    ax.set_title("验证末端距离（模型选择判据）")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── (3) 物理损失分量 ──
    ax = axes[2]
    ax.plot(epochs, tr_phy, "g-", linewidth=2.5, label="训练物理损失（CW 残差）")
    ax.plot(epochs, tr_mode, "m-", linewidth=2.5, label="训练模式损失（Δv alignment）")
    ax.plot(epochs, va_phy, "c--", linewidth=2.5, label="验证物理损失")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss（已归一化）")
    ax.set_title("物理损失分量")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Loss 曲线已保存: {save_path}")

    # ── 保存对应 .mat 数据文件 ──
    mat_path = save_path.replace(".png", ".mat")
    mat_data = {
        "epoch": np.arange(1, n + 1, dtype=np.int32),
        "train_total": np.array(data["train_total"]),
        "train_pred": np.array(data["train_pred"]),
        "train_phy": np.array(data["train_phy"]),
        "train_mode": np.array(data["train_mode"]),
    }
    if data["val_pred"]:
        mat_data["val_pred"] = np.array(data["val_pred"])
    if data["val_phy"]:
        mat_data["val_phy"] = np.array(data["val_phy"])
    if data["val_td"]:
        mat_data["val_terminal_dist"] = np.array(data["val_td"])
    sio.savemat(mat_path, mat_data)
    print(f"  └─ 数据已保存: {mat_path}")


def plot_dv_distribution(dv_pulses, save_dir=OUTPUT_DIR):
    """
    绘制**脉冲 Δv** 幅值分布（已扣除 CW 自由演化）。

    ⚠️ 2026-09-10 修正：旧实现接收模型输出 dv_all 并绘制"相邻步速度差分"，
    该量包含 60 s 自由演化贡献（约 4 m/s），会被误读为模型违反 Δv 约束
    （实测超限占比看似接近 100%）。现直接接收由预测轨迹反解的脉冲 Δv（km/s）。
    """
    if dv_pulses is None or len(dv_pulses) == 0:
        return

    mags = np.array(dv_pulses) * 1000.0  # m/s
    over = float(np.mean(mags > DELTAV_LIMIT)) * 100

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    ax1 = axes[0]
    ax1.hist(mags, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax1.axvline(x=DELTAV_LIMIT, color="r", linestyle="--", linewidth=2,
                label=f"机动上限 {DELTAV_LIMIT} m/s")
    ax1.set_xlabel("脉冲 $\\Delta v$ 幅值 (m/s)")
    ax1.set_ylabel("频次")
    ax1.set_title(f"脉冲 $\\Delta v$ 幅值分布（超限占比 {over:.1f}%）")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(mags, bins=60, cumulative=-1, density=True, color="coral",
             edgecolor="white", alpha=0.85)
    ax2.set_xlabel("脉冲 $\\Delta v$ 幅值 (m/s)")
    ax2.set_ylabel("超越概率 $P(\\Delta v > x)$")
    ax2.set_title("脉冲 $\\Delta v$ 超越概率曲线")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "dv_distribution.png")
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"脉冲 Δv 分布图已保存: {save_path}")

    # ── 保存对应 .mat 数据文件 ──
    mat_path = save_path.replace(".png", ".mat")
    sio.savemat(mat_path, {"dv_pulse_magnitudes": mags})  # m/s
    print(f"  └─ 数据已保存: {mat_path}")


def plot_cw_residual_curve(cw_pos_res, cw_vel_res=None, save_dir=OUTPUT_DIR):
    """
    绘制 CW 单步递推残差分布（位置与速度分量**分别**统计）。

    ⚠️ 2026-09-10 修正：旧实现把位置 (km) 与速度 (km/s) 混为单一欧氏范数，
    却标注单位 "km/s" —— 该标注无物理意义。现拆为两个子图分别统计。
    """
    if not cw_pos_res:
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    ax1 = axes[0]
    ax1.hist(cw_pos_res, bins=60, color="coral", edgecolor="white", alpha=0.85)
    ax1.set_xlabel("CW 单步位置残差范数 (km)")
    ax1.set_ylabel("频次")
    ax1.set_title("CW 位置残差分布")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    if cw_vel_res:
        ax2.hist(cw_vel_res, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax2.set_xlabel("CW 单步速度残差范数 (km/s)")
    ax2.set_ylabel("频次")
    ax2.set_title("CW 速度残差分布")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "cw_residual.png")
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"CW 残差图已保存: {save_path}")

    # ── 保存对应 .mat 数据文件 ──
    mat_path = save_path.replace(".png", ".mat")
    mat_data = {"cw_pos_residuals": np.array(cw_pos_res)}  # km
    if cw_vel_res:
        mat_data["cw_vel_residuals"] = np.array(cw_vel_res)  # km/s
    sio.savemat(mat_path, mat_data)
    print(f"  └─ 数据已保存: {mat_path}")


def main():
    # ── 加载数据 ──
    print("加载数据...")
    (train_X, val_X, test_X,
     train_Y, val_Y, test_Y,
     train_masks, val_masks, test_masks,
     scaler) = load_and_split(DATA_DIR)

    _, _, test_loader = create_dataloaders(
        train_X, val_X, test_X,
        train_Y, val_Y, test_Y,
        train_masks, val_masks, test_masks,
    )

    # ── 加载模型 ──
    print(f"\n加载模型: {MODEL_SAVE_PATH}")
    model = create_model(DEVICE, scaler)
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model_type = checkpoint.get("model_type", "lstm")

    # 尝试严格加载，失败则使用部分加载（兼容旧版 checkpoint）
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"模型权重严格加载成功")
    except RuntimeError as e:
        print(f"严格加载失败，尝试部分加载...")
        model_state = model.state_dict()
        pretrained = checkpoint["model_state_dict"]
        loaded = 0
        skipped = 0
        for key in model_state:
            if key in pretrained and model_state[key].shape == pretrained[key].shape:
                model_state[key] = pretrained[key]
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(model_state)
        print(f"部分加载: {loaded} 层匹配, {skipped} 层跳过（随机初始化）")

    term_dist_info = ""
    if "val_terminal_dist" in checkpoint:
        term_dist = checkpoint["val_terminal_dist"]
        term_dist_info = f", terminal_dist={term_dist:.4f} km"
    print(f"模型类型: {model_type}, epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.6f}{term_dist_info}")

    # ── 评估 ──
    preds, targets, masks, metrics, cw_data = evaluate_model(model, test_loader, scaler, DEVICE)

    # ── 可视化 ──
    print("\n生成可视化图表...")
    plot_predictions(preds, targets, masks, scaler, num_samples=5)
    plot_loss_curve()

    # 物理信息相关可视化（复用评估阶段已算好的 CW 指标，避免重复推理）
    if PHYSICS_ENABLED and cw_data is not None:
        cw_pos_res, cw_vel_res, dv_pulses = cw_data
        plot_dv_distribution(dv_pulses)
        plot_cw_residual_curve(cw_pos_res, cw_vel_res)

    print("\n评估完成。")


if __name__ == "__main__":
    main()
