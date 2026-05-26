"""
评估脚本 —— 在测试集上计算指标并生成可视化图表。
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
)
from utils.data_loader import (
    load_and_split, create_dataloaders, masked_mse_loss,
    FeatureScaler, compute_metrics,
)
from models.model import create_model


# 设置中文字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def evaluate_model(model, loader, scaler, device):
    """在测试集上计算指标（还原到原始量纲后计算）。"""
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = model(x)
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
            all_masks.append(mask.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)

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
    """从训练日志中提取并绘制 loss 曲线。"""
    if not os.path.exists(log_path):
        print(f"日志文件不存在，跳过 loss 曲线绘制: {log_path}")
        return

    train_losses = []
    val_losses = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            # 兼容新旧日志格式
            if "Train:" in line and "Val:" in line:
                parts = line.split("|")
                try:
                    # "Train: 0.123" 或 "Train Loss: 0.123"
                    tl_part = [p for p in parts if "Train" in p][0]
                    vl_part = [p for p in parts if "Val" in p][0]
                    tl = float(tl_part.split(":")[1].strip())
                    vl = float(vl_part.split(":")[1].strip())
                    train_losses.append(tl)
                    val_losses.append(vl)
                except (IndexError, ValueError):
                    continue

    if not train_losses:
        print("无法从日志中解析 loss 数据")
        return

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", linewidth=1.5, label="训练损失")
    plt.plot(epochs, val_losses, "r-", linewidth=1.5, label="验证损失")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("训练 / 验证损失曲线")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss 曲线已保存: {save_path}")


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
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"模型来自 epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6f}")

    # ── 评估 ──
    preds, targets, masks, metrics = evaluate_model(model, test_loader, scaler, DEVICE)

    # ── 可视化 ──
    print("\n生成可视化图表...")
    plot_predictions(preds, targets, masks, scaler, num_samples=5)
    plot_loss_curve()

    print("\n评估完成。")


if __name__ == "__main__":
    main()
