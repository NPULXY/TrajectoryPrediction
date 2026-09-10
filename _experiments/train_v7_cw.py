"""
v7 CW 残差学习（修复版 - 真正架构级残差学习）

v3 修复要点：
- model.forward 现在接受 baseline 参数
- 传 baseline 时：model 输出 = baseline + delta_residual（不是 persistence + delta）
- 训练目标：model_output 接近 y（即 delta_residual 接近 y - baseline）
- loss 直接在 final_pred vs y 上（不分离算残差 loss）
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

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


def train_epoch(model, cw_baseline, loader, optimizer, device, epoch, total_epochs, scaler):
    """训练一个 epoch：CW baseline + 残差学习（架构级）"""
    model.train()
    cw_baseline.eval()

    total_loss = 0.0
    total_l_pred = 0.0
    total_l_terminal = 0.0
    total_grad_norm = 0.0
    n_batches = 0

    lambda_terminal = get_physics_weight(
        epoch, TERMINAL_WARMUP_EPOCHS,
        TERMINAL_LOSS_WEIGHT, TERMINAL_LOSS_WEIGHT_FINAL
    )

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()

        # CW baseline
        x_init = x[:, -1, :]
        with torch.no_grad():
            baseline = cw_baseline(x_init, mask)  # (B, 10, max_dim)

        # Model 学残差：传 baseline，model 输出 baseline + delta_residual
        pred, dv_all = model(x, return_dv=True, mask=mask, baseline=baseline)
        # pred 现在 = baseline + delta_residual（v7 模式）

        # 预测损失：直接比较 pred 与 y
        l_pred = masked_mse_loss(pred, y, mask)
        loss = l_pred
        total_l_pred += l_pred.item()

        # 末端位置损失（物理空间 3D 距离）
        if lambda_terminal > 0:
            pos_indices = [i * 6 + j for i in range(MAX_N) for j in range(3)]
            _term_mean = torch.from_numpy(scaler.mean.astype('float32')).to(device)
            _term_std = torch.from_numpy(scaler.std.astype('float32')).to(device)
            eps = 1e-8
            last_pred_pos_n = pred[:, -1:, pos_indices].squeeze(1)
            last_true_pos_n = y[:, -1:, pos_indices].squeeze(1)
            pos_mean_l = _term_mean[pos_indices].to(device)
            pos_std_l = _term_std[pos_indices].to(device)
            last_pred_phys = last_pred_pos_n * (pos_std_l + eps) + pos_mean_l
            last_true_phys = last_true_pos_n * (pos_std_l + eps) + pos_mean_l
            B = pred.shape[0]
            pp = last_pred_phys.reshape(B, MAX_N, 3)
            pt = last_true_phys.reshape(B, MAX_N, 3)
            dist = torch.norm(pp - pt, dim=-1)
            pos_mask = mask[:, pos_indices]
            valid_pos = pos_mask.reshape(B, MAX_N, 3).any(dim=-1).float()
            l_terminal = (dist * valid_pos).sum() / valid_pos.sum().clamp(min=1)
            loss = loss + lambda_terminal * l_terminal
            total_l_terminal += l_terminal.item()

        loss.backward()
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
        "pred": total_l_pred / max(1, n_batches),
        "terminal": total_l_terminal / max(1, n_batches),
        "grad_norm": total_grad_norm / max(1, n_batches),
        "lambda_t": lambda_terminal,
    }


@torch.no_grad()
def validate(model, cw_baseline, loader, device, scaler):
    model.eval()
    cw_baseline.eval()
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

        pred, _ = model(x, return_dv=True, mask=mask, baseline=baseline)

        l_pred = masked_mse_loss(pred, y, mask)
        total_l_pred += l_pred.item()

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
        B = pred.shape[0]
        pp = last_pred_phys.reshape(B, MAX_N, 3)
        pt = last_true_phys.reshape(B, MAX_N, 3)
        dist = torch.norm(pp - pt, dim=-1)
        pos_mask = mask[:, pos_indices]
        valid_pos = pos_mask.reshape(B, MAX_N, 3).any(dim=-1).float()
        l_terminal = (dist * valid_pos).sum() / valid_pos.sum().clamp(min=1)
        total_l_terminal += l_terminal.item()
        n_batches += 1

        preds_raw = scaler.inverse_transform(pred.cpu().numpy())
        all_preds.append(preds_raw)
        all_targets.append(y.cpu().numpy())
        all_masks.append(mask.cpu().numpy())

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
        "pred": total_l_pred / n_batches,
        "terminal": total_l_terminal / n_batches,
        "terminal_dist_mean": float(dists.mean()),
        "terminal_dist_min": float(dists.min()),
        "terminal_dist_max": float(dists.max()),
        "terminal_dist_median": float(np.median(dists)),
    }


def train():
    print("=" * 60)
    print("v7 CW Baseline + 架构级残差学习（v3 修复后）")
    print("=" * 60)

    print("\n加载数据...")
    (train_X, val_X, test_X,
     train_Y, val_Y, test_Y,
     train_masks, val_masks, test_masks,
     scaler) = load_and_split(DATA_DIR)
    train_loader, val_loader, _ = create_dataloaders(
        train_X, val_X, test_X,
        train_Y, val_Y, test_Y,
        train_masks, val_masks, test_masks,
    )
    print(f"Scaler 已加载")

    print(f"\n使用设备: {DEVICE}")
    model = create_model(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型: {type(model).__name__}, {total_params:,} 参数")

    # CW baseline
    cw_baseline = CWBaseline(n=CW_N, dt=60.0, num_steps=10, max_N=MAX_N, scaler=scaler).to(DEVICE)
    print(f"CW Baseline 已初始化（标准化空间）")

    # 加载 v3 best 初始化
    v3_ckpt_path = "backup_v3/best_model_v3_ep58_td4.30.pth"
    if os.path.exists(v3_ckpt_path):
        v3_ckpt = torch.load(v3_ckpt_path, map_location=DEVICE)
        v3_state = v3_ckpt["model_state_dict"]
        model_state = model.state_dict()
        loaded = 0
        for k, v in v3_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
        model.load_state_dict(model_state)
        print(f"v3 best 权重部分加载（loaded={loaded}，作为 v7 残差学习的起点）")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS)

    best_val_loss = float("inf")
    best_terminal_dist = float("inf")
    best_epoch = 0
    patience_counter = 0
    start_epoch = 1

    log_file = open(LOG_PATH, "w", encoding="utf-8")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"\nv7 CW 残差学习训练开始: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Batch: {BATCH_SIZE}, Peak LR: {LEARNING_RATE}, Epochs: {EPOCHS}")
    log("架构：model 接收 baseline，输出 baseline + delta_residual")
    log("-" * 60)

    pbar = tqdm(range(start_epoch, EPOCHS + 1), desc="v7 CW 残差", unit="epoch")

    for epoch in pbar:
        t0 = time.time()
        train_info = train_epoch(model, cw_baseline, train_loader, optimizer, DEVICE, epoch, EPOCHS, scaler)
        val_info = validate(model, cw_baseline, val_loader, DEVICE, scaler)
        scheduler.step()
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        td = val_info['terminal_dist_mean']
        log(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train: {train_info['total']:.6f} (pred={train_info['pred']:.6f} term={train_info['terminal']:.6f}) | "
            f"Val td={td:.3f}km | "
            f"λ_t={train_info['lambda_t']:.3f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        if td < best_terminal_dist:
            best_terminal_dist = td
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_info["pred"],
                "val_terminal_dist": td,
                "best_terminal_dist": best_terminal_dist,
                "model_type": "pinn_lstm_v7_cw_architectural_residual",
            }, MODEL_SAVE_PATH)
            log(f"  >> 最佳已保存 td={best_terminal_dist:.4f} km")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                log(f"\n早停触发，最佳 td={best_terminal_dist:.4f} km @ epoch {best_epoch}")
                break

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "best_terminal_dist": best_terminal_dist,
            "patience_counter": patience_counter,
            "model_type": "pinn_lstm_v7_cw_architectural_residual",
        }, CHECKPOINT_SAVE_PATH)

    log_file.close()
    print(f"\n训练完成，最佳 td={best_terminal_dist:.4f} km @ epoch {best_epoch}")


if __name__ == "__main__":
    train()