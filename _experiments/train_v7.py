"""
v7: CW 演化 baseline + 残差学习

核心改进（vs v3 PI-LSTM）：
1. **CW baseline**：从 X_now 末步用 CW 矩阵外推 10 步作为 baseline（物理可解析）
2. **残差学习**：模型只学 delta = X_next - CW_baseline（更小、更易学）
3. **保留 v3 PI-LSTM 架构**：encoder + 条件门控 + 位置/速度解耦 head

新架构：
  输入 (B, 10, 24) + mask
    → CW baseline (B, 10, 24) from X_now[-1]
    → delta_target = X_next - CW_baseline (训练时算)
    → PI-LSTM 输入 (state + 条件)
    → delta_pred (B, 10, 24)
    → final_pred = CW_baseline + delta_pred
"""

import os
import sys
import math
import time
import numpy as np
import torch
import torch.optim as optim
import scipy.io as sio
from tqdm import tqdm

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

from config import (
    DATA_DIR, OUTPUT_DIR, MODEL_SAVE_PATH, CHECKPOINT_SAVE_PATH,
    SCALER_SAVE_PATH, LOG_PATH,
    DEVICE, BATCH_SIZE, LEARNING_RATE, MIN_LR, WEIGHT_DECAY, EPOCHS,
    EARLY_STOP_PATIENCE, WARMUP_EPOCHS,
    PHYSICS_ENABLED, USE_TRANSFORMER, PHYSICS_LOSS_WEIGHT, PHYSICS_LOSS_WEIGHT_FINAL,
    PHYSICS_WARMUP_EPOCHS, MODE_LOSS_WEIGHT, MODE_LOSS_WEIGHT_FINAL,
    PRED_WARMUP_EPOCHS, DELTAV_LIMIT, CW_N, CW_DT_H, CONDITION_EMBED_DIM,
    RESUME_TRAINING,
    TERMINAL_LOSS_WEIGHT, TERMINAL_LOSS_WEIGHT_FINAL,
    TERMINAL_WARMUP_EPOCHS, MAX_N,
    DELTAV_BOUND_WEIGHT, DELTAV_BOUND_WEIGHT_FINAL, DELTAV_BOUND_WARMUP_EPOCHS,
    TERMINAL_PHYSICAL,
)
from utils.data_loader import (
    load_and_split, create_dataloaders, masked_mse_loss,
)
from models.model import create_model
from models.cw_baseline import CWBaseline


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=MIN_LR):
    """Cosine annealing + warmup 学习率调度器。"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / LEARNING_RATE, 0.5 * (1 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_physics_weight(epoch, warmup_epochs, initial, final):
    if epoch < PRED_WARMUP_EPOCHS:
        return 0.0
    progress = min(1.0, (epoch - PRED_WARMUP_EPOCHS) / max(1, warmup_epochs))
    return initial + (final - initial) * progress


def huber_loss_per_sample(pred, target, mask, delta=1.0):
    diff = (pred - target).abs()
    quadratic = torch.minimum(diff, torch.tensor(delta, device=diff.device))
    linear = diff - quadratic
    loss_per_elem = 0.5 * quadratic.pow(2) + delta * linear
    mask_expanded = mask.unsqueeze(1).expand_as(loss_per_elem)
    n_valid_per_sample = mask_expanded.float().sum(dim=(1, 2)).clamp(min=1)
    loss_per_sample = (loss_per_elem * mask_expanded).sum(dim=(1, 2)) / n_valid_per_sample
    return loss_per_sample


def train_epoch(model, cw_baseline, loader, optimizer, device, epoch, total_epochs, scaler):
    """训练一个 epoch：CW baseline + 残差学习"""
    model.train()
    cw_baseline.eval()  # baseline 是固定的

    total_loss = 0.0
    total_l_pred = 0.0
    total_l_terminal = 0.0
    total_l_residual = 0.0
    total_grad_norm = 0.0
    n_batches = 0

    # 损失权重
    lambda_terminal = get_physics_weight(
        epoch, TERMINAL_WARMUP_EPOCHS,
        TERMINAL_LOSS_WEIGHT, TERMINAL_LOSS_WEIGHT_FINAL
    )

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()

        # 计算 CW baseline：先反归一化到原始物理量纲
        # x_init_norm = x[:, -1, :] (B, 24) - X_now 末步状态（标准化）
        # x_init_raw = x_init_norm * std + mean
        x_init_norm = x[:, -1, :]  # (B, max_dim)
        _mean = torch.from_numpy(scaler.mean.astype('float32')).to(device)
        _std = torch.from_numpy(scaler.std.astype('float32')).to(device)
        x_init_raw = x_init_norm * _std + _mean  # (B, max_dim) 原始物理空间
        with torch.no_grad():
            baseline_raw = cw_baseline(x_init_raw, mask)  # (B, num_steps, max_dim) 原始空间

        # y 是标准化空间，转换为原始空间
        y_raw = y * _std + _mean  # (B, 10, max_dim)

        # delta_target = y_raw - baseline_raw（原始空间）
        delta_target_raw = y_raw - baseline_raw  # (B, 10, max_dim)

        # delta_target 标准化（用于损失）
        delta_target_norm = delta_target_raw / (_std + 1e-8)  # (B, 10, max_dim)

        # 模型预测
        pred, dv_all = model(x, return_dv=True, mask=mask)
        delta_pred = pred - baseline  # 残差预测

        # 损失
        # 1. 残差预测损失 (Huber per-sample)
        loss_per_sample = huber_loss_per_sample(delta_pred, delta_target, mask, delta=1.0)
        l_residual = loss_per_sample.mean()
        loss = l_residual
        total_l_residual += l_residual.item()

        # 2. 末端位置损失（物理空间）
        if lambda_terminal > 0:
            pos_indices = [i * 6 + j for i in range(MAX_N) for j in range(3)]
            # 反归一化最后一步位置
            _term_mean = torch.from_numpy(scaler.mean.astype('float32')).to(device)
            _term_std = torch.from_numpy(scaler.std.astype('float32')).to(device)
            eps = 1e-8
            last_pred_pos = pred[:, -1:, pos_indices].squeeze(1)
            last_true_pos = y[:, -1:, pos_indices].squeeze(1)
            pos_mean_l = _term_mean[pos_indices].to(device)
            pos_std_l = _term_std[pos_indices].to(device)
            last_pred_phys = last_pred_pos * (pos_std_l + eps) + pos_mean_l
            last_true_phys = last_true_pos * (pos_std_l + eps) + pos_mean_l
            B_local = pred.shape[0]
            pp = last_pred_phys.reshape(B_local, MAX_N, 3)
            pt = last_true_phys.reshape(B_local, MAX_N, 3)
            dist = torch.norm(pp - pt, dim=-1)
            pos_mask = mask[:, pos_indices]
            valid_pos = pos_mask.reshape(B_local, MAX_N, 3).any(dim=-1).float()
            l_terminal = (dist * valid_pos).sum() / valid_pos.sum().clamp(min=1)
            loss = loss + lambda_terminal * l_terminal
            total_l_terminal += l_terminal.item()

        # 反向传播
        loss.backward()
        # 梯度范数
        batch_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2).item()
                batch_grad_norm += param_norm ** 2
        total_grad_norm += math.sqrt(batch_grad_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {
        "total": total_loss / max(1, n_batches),
        "pred": total_l_residual / max(1, n_batches),
        "terminal": total_l_terminal / max(1, n_batches),
        "residual": total_l_residual / max(1, n_batches),
        "grad_norm": total_grad_norm / max(1, n_batches),
        "lambda_t": lambda_terminal,
    }


@torch.no_grad()
def validate(model, cw_baseline, loader, device, scaler):
    model.eval()
    cw_baseline.eval()

    total_loss = 0.0
    total_l_pred = 0.0
    total_l_terminal = 0.0
    n_batches = 0

    all_preds = []
    all_targets = []
    all_masks = []

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        x_init = x[:, -1, :]
        baseline = cw_baseline(x_init, mask)

        pred, _ = model(x, return_dv=True, mask=mask)
        delta_pred = pred - baseline

        # 残差损失
        loss_per_sample = huber_loss_per_sample(delta_pred, y - baseline, mask, delta=1.0)
        l_residual = loss_per_sample.mean()
        l_total = l_residual
        total_l_pred += l_residual.item()

        # 末端距离
        pos_indices = [i * 6 + j for i in range(MAX_N) for j in range(3)]
        _term_mean = torch.from_numpy(scaler.mean.astype('float32')).to(device)
        _term_std = torch.from_numpy(scaler.std.astype('float32')).to(device)
        eps = 1e-8
        last_pred_pos = pred[:, -1:, pos_indices].squeeze(1)
        last_true_pos = y[:, -1:, pos_indices].squeeze(1)
        pos_mean_l = _term_mean[pos_indices].to(device)
        pos_std_l = _term_std[pos_indices].to(device)
        last_pred_phys = last_pred_pos * (pos_std_l + eps) + pos_mean_l
        last_true_phys = last_true_pos * (pos_std_l + eps) + pos_mean_l
        B_local = pred.shape[0]
        pp = last_pred_phys.reshape(B_local, MAX_N, 3)
        pt = last_true_phys.reshape(B_local, MAX_N, 3)
        dist = torch.norm(pp - pt, dim=-1)
        pos_mask = mask[:, pos_indices]
        valid_pos = pos_mask.reshape(B_local, MAX_N, 3).any(dim=-1).float()
        l_terminal = (dist * valid_pos).sum() / valid_pos.sum().clamp(min=1)
        l_total = l_total + 2.0 * l_terminal  # 验证时用完整权重
        total_l_terminal += l_terminal.item()

        total_loss += l_total.item()
        n_batches += 1

        # 收集原始量纲预测用于 td 计算
        preds_raw = scaler.inverse_transform(pred.cpu().numpy())
        all_preds.append(preds_raw)
        all_targets.append(y.cpu().numpy())
        all_masks.append(mask.cpu().numpy())

    # 计算 td
    preds_cat = np.concatenate(all_preds, axis=0)
    targets_cat = np.concatenate(all_targets, axis=0)
    masks_cat = np.concatenate(all_masks, axis=0)
    targets_raw = scaler.inverse_transform(targets_cat)

    dists = []
    for i in range(len(preds_cat)):
        n = int(masks_cat[i].sum()) // 6
        max_d = 0
        for a in range(n):
            base = a * 6
            d = np.linalg.norm(preds_cat[i, -1, base:base+3] - targets_raw[i, -1, base:base+3])
            if d > max_d:
                max_d = d
        dists.append(max_d)
    dists = np.array(dists)

    return {
        "total": total_loss / max(1, n_batches),
        "pred": total_l_pred / max(1, n_batches),
        "terminal": total_l_terminal / max(1, n_batches),
        "terminal_dist_mean": float(dists.mean()),
        "terminal_dist_min": float(dists.min()),
        "terminal_dist_max": float(dists.max()),
        "terminal_dist_median": float(np.median(dists)),
    }


def train():
    print("=" * 60)
    print("v7: CW Baseline + 残差学习")
    print("=" * 60)

    print("\n加载数据...")
    (train_X, val_X, test_X,
     train_Y, val_Y, test_Y,
     train_masks, val_masks, test_masks,
     scaler) = load_and_split(DATA_DIR)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_X, val_X, test_X,
        train_Y, val_Y, test_Y,
        train_masks, val_masks, test_masks,
    )

    scaler.save(SCALER_SAVE_PATH)
    print(f"Scaler 已保存至: {SCALER_SAVE_PATH}")

    print(f"\n使用设备: {DEVICE}")
    model = create_model(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型类型: {type(model).__name__}, 参数量: {total_params:,}")

    # CW baseline (固定，不参与训练)
    cw_baseline = CWBaseline(n=CW_N, dt=60.0, num_steps=10, max_N=MAX_N).to(DEVICE)
    print(f"CW Baseline: n={CW_N}, dt=60s, num_steps=10")

    # 优化器 + 调度器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS)

    # 恢复
    start_epoch = 1
    best_val_loss = float("inf")
    best_terminal_dist = float("inf")
    best_epoch = 0
    patience_counter = 0

    if RESUME_TRAINING:
        if os.path.exists(CHECKPOINT_SAVE_PATH):
            ckpt_path = CHECKPOINT_SAVE_PATH
        elif os.path.exists(MODEL_SAVE_PATH):
            ckpt_path = MODEL_SAVE_PATH
        else:
            ckpt_path = None

        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["val_loss"]
            best_terminal_dist = checkpoint.get("best_terminal_dist", float("inf"))
            best_epoch = checkpoint.get("best_epoch", checkpoint["epoch"])
            patience_counter = checkpoint.get("patience_counter", 0)
            source = "latest" if ckpt_path == CHECKPOINT_SAVE_PATH else "best"
            print(f"从检查点恢复 [{source} 存档]: epoch {checkpoint['epoch']}, "
                  f"td={best_terminal_dist:.4f} km")

    # 日志
    log_file = open(LOG_PATH, "a", encoding="utf-8") if (RESUME_TRAINING and start_epoch > 1) else open(LOG_PATH, "w", encoding="utf-8")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"\nv7 训练开始: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"模型类型: {type(model).__name__}")
    log(f"训练样本: {len(train_X)}, 验证样本: {len(val_X)}, 测试样本: {len(test_X)}")
    log(f"Batch: {BATCH_SIZE}, Peak LR: {LEARNING_RATE}")
    log(f"末距损失权重: {TERMINAL_LOSS_WEIGHT} → {TERMINAL_LOSS_WEIGHT_FINAL}")
    log(f"恢复模式: start_epoch={start_epoch}, best_epoch={best_epoch}, "
        f"best_terminal_dist={best_terminal_dist:.4f} km")
    log("-" * 60)

    # 训练循环
    pbar = tqdm(range(start_epoch, EPOCHS + 1), desc="训练", unit="epoch")

    for epoch in pbar:
        t0 = time.time()

        train_info = train_epoch(model, cw_baseline, train_loader, optimizer, DEVICE, epoch, EPOCHS, scaler)
        val_info = validate(model, cw_baseline, val_loader, DEVICE, scaler)

        scheduler.step()
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        # 日志
        td = val_info['terminal_dist_mean']
        log(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train: {train_info['total']:.6f} (residual={train_info['residual']:.6f} term={train_info['terminal']:.6f}) | "
            f"Val: {val_info['total']:.6f} | "
            f"td={td:.3f}km | "
            f"λ_t={train_info['lambda_t']:.3f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # 保存最佳
        if td < best_terminal_dist:
            best_terminal_dist = td
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_info["pred"],
                "val_terminal_dist": td,
                "best_terminal_dist": best_terminal_dist,
                "model_type": "cw_residual_v7",
            }, MODEL_SAVE_PATH)
            log(f"  >> 最佳模型已保存（td={best_terminal_dist:.4f} km）")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                log(f"\n早停触发，最佳 epoch: {best_epoch}, best_td={best_terminal_dist:.4f} km")
                break

        # 保存 latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "best_terminal_dist": best_terminal_dist,
            "patience_counter": patience_counter,
            "model_type": "cw_residual_v7",
        }, CHECKPOINT_SAVE_PATH)

    log_file.close()

    # 保存 training history
    print(f"\n训练完成，最佳: epoch {best_epoch}, td={best_terminal_dist:.4f} km")


if __name__ == "__main__":
    train()