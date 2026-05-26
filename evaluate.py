"""
评估脚本 —— 在测试集上计算指标并生成可视化图表。
支持原版 LSTM 和物理信息条件 LSTM 两种模式的评估。
用法: python evaluate.py
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


# 设置中文字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def compute_cw_metrics(preds_raw, masks, dv_all=None):
    """
    计算物理一致性指标: CW 单步递推残差和速度变化统计。

    Args:
        preds_raw: (N, 10, 24) 预测轨迹（原始量纲）
        masks:     (N, 24) 有效特征掩码
        dv_all:    (N, 19, max_N*3) Δv 估计（可选，仅用于记录）

    Returns:
        dict: CW 残差和速度变化统计量
    """
    from models.physics_loss import compute_cw_matrix, N_MEAN

    Phi_h = compute_cw_matrix(N_MEAN, CW_DT_H)
    Phi_np = Phi_h.numpy()

    n_total = preds_raw.shape[0]
    cw_residuals = []
    dv_magnitudes = []  # 速度变化幅值

    for i in range(n_total):
        mask = masks[i]
        n_agents = int(mask.sum()) // 6
        if n_agents == 0:
            continue

        for a in range(n_agents):
            base = a * 6
            states = preds_raw[i, :, base:base + 6]  # (10, 6)

            # CW 自由演化残差
            for t in range(9):
                s_curr = states[t]
                s_next = states[t + 1]
                s_free = Phi_np @ s_curr
                residual = s_next - s_free
                cw_residuals.append(float(np.linalg.norm(residual)))

            # 速度变化（Δv 近似）
            vel = states[:, 3:6]  # (10, 3)
            dv = vel[1:] - vel[:-1]  # (9, 3)
            mags = np.linalg.norm(dv, axis=-1)  # (9,)
            dv_magnitudes.extend(mags.tolist())

    result = {
        "cw_residual_mean": float(np.mean(cw_residuals)) if cw_residuals else 0.0,
        "cw_residual_std": float(np.std(cw_residuals)) if cw_residuals else 0.0,
        "cw_residual_max": float(np.max(cw_residuals)) if cw_residuals else 0.0,
    }

    if dv_magnitudes:
        result["dv_mag_mean"] = float(np.mean(dv_magnitudes))
        result["dv_mag_std"] = float(np.std(dv_magnitudes))
        result["dv_mag_max"] = float(np.max(dv_magnitudes))

    return result, cw_residuals, dv_magnitudes


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

    # 物理一致性指标
    if PHYSICS_ENABLED:
        dv_np = dv_all.numpy() if dv_all is not None else None
        cw_metrics, cw_res, dv_mags = compute_cw_metrics(preds_raw, masks.numpy(), dv_np)
        metrics.update(cw_metrics)

        print("\n--- 物理一致性指标 ---")
        print(f"CW 残差均值:  {cw_metrics['cw_residual_mean']:.6f} km/s")
        print(f"CW 残差标准差: {cw_metrics['cw_residual_std']:.6f} km/s")
        if 'dv_mag_mean' in cw_metrics:
            print(f"Δv 幅值均值:  {cw_metrics['dv_mag_mean']*1000:.4f} m/s")
            print(f"Δv 幅值最大值: {cw_metrics['dv_mag_max']*1000:.4f} m/s")

    return preds_raw, targets_raw, masks.numpy(), metrics


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

        fig = plt.figure(figsize=(16, 5 * n_agents))
        fig.suptitle(f"样本 #{sample_idx} (N={n_agents})", fontsize=14)

        time_steps = np.arange(10)

        for agent in range(n_agents):
            base = agent * 6

            # ── 3D 位置图 ──
            ax3d = fig.add_subplot(n_agents, 4, agent * 4 + 1, projection="3d")
            ax3d.plot(true[:, base + 0], true[:, base + 1], true[:, base + 2],
                      "b-", linewidth=2, label="真实")
            ax3d.plot(pred[:, base + 0], pred[:, base + 1], pred[:, base + 2],
                      "r--", linewidth=2, label="预测")
            ax3d.scatter(true[0, base + 0], true[0, base + 1], true[0, base + 2],
                         c="blue", s=50, marker="o")
            ax3d.scatter(pred[0, base + 0], pred[0, base + 1], pred[0, base + 2],
                         c="red", s=50, marker="o")
            ax3d.set_xlabel("X (km)")
            ax3d.set_ylabel("Y (km)")
            ax3d.set_zlabel("Z (km)")
            ax3d.set_title(f"目标 {agent+1} 3D 轨迹")
            ax3d.legend(fontsize=7)

            # ── X 分量 ──
            ax_x = fig.add_subplot(n_agents, 4, agent * 4 + 2)
            ax_x.plot(time_steps, true[:, base + 0], "b-o", markersize=4, label="真实")
            ax_x.plot(time_steps, pred[:, base + 0], "r--s", markersize=4, label="预测")
            ax_x.set_xlabel("时间步")
            ax_x.set_ylabel("X (km)")
            ax_x.set_title(f"目标 {agent+1} X 分量")
            ax_x.legend(fontsize=7)
            ax_x.grid(True, alpha=0.3)

            # ── Y 分量 ──
            ax_y = fig.add_subplot(n_agents, 4, agent * 4 + 3)
            ax_y.plot(time_steps, true[:, base + 1], "b-o", markersize=4, label="真实")
            ax_y.plot(time_steps, pred[:, base + 1], "r--s", markersize=4, label="预测")
            ax_y.set_xlabel("时间步")
            ax_y.set_ylabel("Y (km)")
            ax_y.set_title(f"目标 {agent+1} Y 分量")
            ax_y.legend(fontsize=7)
            ax_y.grid(True, alpha=0.3)

            # ── Z 分量 ──
            ax_z = fig.add_subplot(n_agents, 4, agent * 4 + 4)
            ax_z.plot(time_steps, true[:, base + 2], "b-o", markersize=4, label="真实")
            ax_z.plot(time_steps, pred[:, base + 2], "r--s", markersize=4, label="预测")
            ax_z.set_xlabel("时间步")
            ax_z.set_ylabel("Z (km)")
            ax_z.set_title(f"目标 {agent+1} Z 分量")
            ax_z.legend(fontsize=7)
            ax_z.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sample_{sample_idx:05d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"图表已保存: {save_path}")


def plot_loss_curve(log_path=os.path.join(os.path.dirname(__file__), "output", "train_log.txt"),
                    save_dir=OUTPUT_DIR):
    """从训练日志中提取并绘制 loss 曲线（含各分量）。"""
    if not os.path.exists(log_path):
        print(f"日志文件不存在，跳过 loss 曲线绘制: {log_path}")
        return

    train_total = []
    val_total = []
    train_pred = []
    train_phy = []
    train_mode = []
    val_pred = []
    val_phy = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            # 新格式: "Train: 0.123 (pred=0.100 phy=0.020 mode=0.003)"
            if "Train:" in line and "pred=" in line:
                try:
                    # 提取 train total
                    t_part = line.split("Train:")[1].split("(")[0].strip()
                    train_total.append(float(t_part))
                    # 提取 pred
                    pred_part = line.split("pred=")[1].split(" ")[0].strip()
                    train_pred.append(float(pred_part))
                    # 提取 phy
                    phy_part = line.split("phy=")[1].split(" ")[0].strip()
                    train_phy.append(float(phy_part))
                    # 提取 mode
                    mode_part = line.split("mode=")[1].split(")")[0].strip()
                    train_mode.append(float(mode_part))
                except (IndexError, ValueError):
                    pass
                # 提取 val pred
                try:
                    v_pred_part = line.split("Val:")[1].split("(pred=")[1].split(" ")[0].strip()
                    val_pred.append(float(v_pred_part))
                except (IndexError, ValueError):
                    pass
                try:
                    if "phy=" in line.split("Val:")[1]:
                        v_phy_part = line.split("Val:")[1].split("phy=")[1].split(")")[0].strip()
                        val_phy.append(float(v_phy_part))
                except (IndexError, ValueError):
                    pass
            # 旧格式: "Train: 0.123 | Val: 0.234"
            elif "Train:" in line and "Val:" in line and "pred=" not in line:
                parts = line.split("|")
                try:
                    tl_part = [p for p in parts if "Train" in p][0]
                    vl_part = [p for p in parts if "Val" in p][0]
                    tl = float(tl_part.split(":")[1].strip())
                    vl = float(vl_part.split(":")[1].strip())
                    train_total.append(tl)
                    val_total.append(vl)
                except (IndexError, ValueError):
                    continue

    if not train_total:
        print("无法从日志中解析 loss 数据")
        return

    # 主 loss 图
    fig, axes = plt.subplots(1, 2 if train_pred else 1,
                              figsize=(14 if train_pred else 7, 5))
    if train_pred:
        ax1, ax2 = axes[0], axes[1]
    else:
        ax1 = axes
    epochs = range(1, len(train_total) + 1)

    ax1.plot(epochs, train_total, "b-", linewidth=1.5, label="训练总损失")
    if val_total:
        ax1.plot(epochs[:len(val_total)], val_total, "r-", linewidth=1.5, label="验证总损失")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("训练 / 验证损失曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if train_pred:
        ax2.plot(epochs[:len(train_pred)], train_pred, "b-", linewidth=1, label="训练预测损失")
        ax2.plot(epochs[:len(train_phy)], train_phy, "g-", linewidth=1, label="训练物理损失")
        ax2.plot(epochs[:len(train_mode)], train_mode, "m-", linewidth=1, label="训练模式损失")
        if val_pred:
            ax2.plot(epochs[:len(val_pred)], val_pred, "r--", linewidth=1, label="验证预测损失")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("损失各分量")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss 曲线已保存: {save_path}")


def plot_dv_distribution(dv_all, masks, save_dir=OUTPUT_DIR):
    """绘制 Δv 幅值分布直方图。"""
    if dv_all is None:
        return

    dv_mags_all = []
    for i in range(dv_all.shape[0]):
        mask = masks[i]
        n_agents = int(mask.sum()) // 6
        for a in range(n_agents):
            dv = dv_all[i, :, a * 3:(a + 1) * 3]
            mags = np.linalg.norm(dv, axis=-1)
            dv_mags_all.extend(mags.tolist())

    if not dv_mags_all:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 幅值分布直方图
    ax1 = axes[0]
    ax1.hist(np.array(dv_mags_all) * 1000, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax1.axvline(x=DELTAV_LIMIT, color="r", linestyle="--", linewidth=1.5, label=f"上限 {DELTAV_LIMIT} m/s")
    ax1.set_xlabel("Δv 幅值 (m/s)")
    ax1.set_ylabel("频次")
    ax1.set_title("Δv 幅值分布")
    ax1.legend()

    # 各分量分布
    ax2 = axes[1]
    dv_components = []
    labels = ["Δvx", "Δvy", "Δvz"]
    for i in range(dv_all.shape[0]):
        mask = masks[i]
        n_agents = int(mask.sum()) // 6
        for a in range(n_agents):
            dv = dv_all[i, :, a * 3:(a + 1) * 3]
            dv_components.append(dv)
    if dv_components:
        dv_cat = np.concatenate(dv_components, axis=0) * 1000  # m/s
        for j, label in enumerate(labels):
            ax2.hist(dv_cat[:, j], bins=40, alpha=0.5, label=label)
        ax2.set_xlabel("Δv 分量 (m/s)")
        ax2.set_ylabel("频次")
        ax2.set_title("Δv 各分量分布")
        ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, "dv_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Δv 分布图已保存: {save_path}")


def plot_cw_residual_curve(cw_residuals, save_dir=OUTPUT_DIR):
    """绘制 CW 残差分布图。"""
    if not cw_residuals:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax1 = axes[0]
    ax1.hist(cw_residuals, bins=50, color="coral", edgecolor="white", alpha=0.8)
    ax1.set_xlabel("CW 单步残差范数 (km/s)")
    ax1.set_ylabel("频次")
    ax1.set_title("CW 残差分布")

    ax2 = axes[1]
    ax2.boxplot(cw_residuals, vert=True)
    ax2.set_ylabel("CW 单步残差范数 (km/s)")
    ax2.set_title("CW 残差箱线图")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "cw_residual.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"CW 残差图已保存: {save_path}")


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
    model = create_model(DEVICE)
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

    print(f"模型类型: {model_type}, epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6f}")

    # ── 评估 ──
    preds, targets, masks, metrics = evaluate_model(model, test_loader, scaler, DEVICE)

    # ── 可视化 ──
    print("\n生成可视化图表...")
    plot_predictions(preds, targets, masks, scaler, num_samples=5)
    plot_loss_curve()

    # 物理信息相关可视化
    if PHYSICS_ENABLED and model_type == "pinn_lstm":
        # 收集 Δv 数据
        model.eval()
        all_dv = []
        with torch.no_grad():
            for x, y, mask in test_loader:
                x, mask = x.to(DEVICE), mask.to(DEVICE)
                _, dv_all = model(x, return_dv=True, mask=mask)
                all_dv.append(dv_all.cpu())
        dv_all = torch.cat(all_dv, dim=0).numpy()

        plot_dv_distribution(dv_all, masks)
        _, cw_res, _ = compute_cw_metrics(preds, masks, dv_all)
        plot_cw_residual_curve(cw_res)

    print("\n评估完成。")


if __name__ == "__main__":
    main()
