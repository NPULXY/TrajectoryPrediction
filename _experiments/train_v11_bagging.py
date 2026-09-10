"""
v11 训练：3 个 v6 架构不同 seed 模型（用于 bagging 集成）

设计：
- 3 个模型，每个都用 v3 best 初始化但不同随机种子
- 用 v6 loss 组合
- 训练 60 epoch
- 集成推理（3 个模型平均）

时间：3 × ~13 分钟 = 40 分钟
"""

import os
import sys
import time
import math
import random
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
    PHYSICS_ENABLED,
    PRED_WARMUP_EPOCHS, MAX_N,
    TERMINAL_LOSS_WEIGHT, TERMINAL_LOSS_WEIGHT_FINAL,
    TERMINAL_WARMUP_EPOCHS, TERMINAL_PHYSICAL,
    MODE_LOSS_WEIGHT, MODE_LOSS_WEIGHT_FINAL,
)
from utils.data_loader import (
    load_and_split, create_dataloaders, masked_mse_loss,
)
from models.model import create_model


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, device, epoch, total_epochs, scaler):
    model.train()
    total_loss = 0.0
    total_l_pred = 0.0
    total_l_mode = 0.0
    total_l_terminal = 0.0
    total_grad_norm = 0.0
    n_batches = 0

    lambda_terminal = get_physics_weight(
        epoch, TERMINAL_WARMUP_EPOCHS,
        TERMINAL_LOSS_WEIGHT, TERMINAL_LOSS_WEIGHT_FINAL
    )
    lambda_mode = get_physics_weight(
        epoch, 15,
        MODE_LOSS_WEIGHT, MODE_LOSS_WEIGHT_FINAL
    )

    _term_mean = torch.from_numpy(scaler.mean.astype('float32')).to(device)
    _term_std = torch.from_numpy(scaler.std.astype('float32')).to(device)

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()

        pred, dv_all = model(x, target=y, teacher_forcing_ratio=0.0,
                             return_dv=True, mask=mask)

        l_pred = masked_mse_loss(pred, y, mask)
        loss = l_pred
        total_l_pred += l_pred.item()

        if lambda_mode > 0 and dv_all is not None:
            dv_cw_input = dv_all[:, :9]
            dv_model_seq = dv_all[:, 9:18]
            Bv, Tv, Dv = dv_model_seq.shape
            max_N_local = Dv // 3
            dm = dv_model_seq.reshape(Bv, Tv, max_N_local, 3)
            dc = dv_cw_input.reshape(Bv, Tv, max_N_local, 3)
            diff = (dm - dc).pow(2)
            agent_mse = diff.mean(dim=(1, 3))
            valid = mask.reshape(Bv, max_N_local, 6).any(dim=-1).float()
            l_mode = (agent_mse * valid).sum() / valid.sum().clamp(min=1)
            loss = loss + lambda_mode * l_mode
            total_l_mode += l_mode.item()

        if lambda_terminal > 0:
            pos_indices = [i * 6 + j for i in range(MAX_N) for j in range(3)]
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
        "mode": total_l_mode / max(1, n_batches),
        "terminal": total_l_terminal / max(1, n_batches),
        "grad_norm": total_grad_norm / max(1, n_batches),
        "lambda_t": lambda_terminal,
    }


@torch.no_grad()
def validate(model, loader, device, scaler):
    model.eval()
    n_batches = 0
    all_preds = []
    all_targets = []
    all_masks = []

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        pred, _ = model(x, return_dv=True, mask=mask)
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
        "terminal_dist_mean": float(dists.mean()),
        "terminal_dist_min": float(dists.min()),
        "terminal_dist_max": float(dists.max()),
        "terminal_dist_median": float(np.median(dists)),
    }


def train_one_model(seed, save_path, v3_init_path):
    print(f"\n=== 训练 v6 不同 seed={seed} ===")
    set_seed(seed)

    print("加载数据...")
    (train_X, val_X, test_X,
     train_Y, val_Y, test_Y,
     train_masks, val_masks, test_masks,
     scaler) = load_and_split(DATA_DIR)
    train_loader, val_loader, _ = create_dataloaders(
        train_X, val_X, test_X,
        train_Y, val_Y, test_Y,
        train_masks, val_masks, test_masks,
    )

    model = create_model(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型: {total_params:,} 参数")

    # 加载 v3 best 初始化
    v3_ckpt = torch.load(v3_init_path, map_location=DEVICE)
    v3_state = v3_ckpt['model_state_dict']
    model_state = model.state_dict()
    loaded = 0
    for k, v in v3_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
    model.load_state_dict(model_state)
    print(f"v3 best 权重部分加载（loaded={loaded}）")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS)

    best_val_loss = float("inf")
    best_terminal_dist = float("inf")
    best_epoch = 0
    patience_counter = 0

    log_file = open(LOG_PATH, "a", encoding="utf-8")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"\n=== v11 bagging seed={seed} 训练开始 ===")
    log(f"Batch: {BATCH_SIZE}, Peak LR: {LEARNING_RATE}, Epochs: {EPOCHS}")

    pbar = tqdm(range(1, EPOCHS + 1), desc=f"seed{seed}", unit="epoch")

    for epoch in pbar:
        t0 = time.time()
        train_info = train_epoch(model, train_loader, optimizer, DEVICE, epoch, EPOCHS, scaler)
        val_info = validate(model, val_loader, DEVICE, scaler)
        scheduler.step()
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        td = val_info['terminal_dist_mean']
        log(
            f"seed{seed} Epoch {epoch:3d}/{EPOCHS} | "
            f"Train: {train_info['total']:.6f} | "
            f"Val td={td:.3f}km | "
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
                "val_terminal_dist": td,
                "best_terminal_dist": best_terminal_dist,
                "model_type": f"pinn_lstm_v11_seed{seed}",
                "seed": seed,
            }, save_path)
            log(f"  >> 最佳已保存 td={best_terminal_dist:.4f} km")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                log(f"早停触发，最佳 td={best_terminal_dist:.4f} km @ epoch {best_epoch}")
                break

    log_file.close()
    print(f"seed{seed} 训练完成，最佳 td={best_terminal_dist:.4f} km")
    return best_terminal_dist


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    train_one_model(
        seed=args.seed,
        save_path=f"backup_v11/best_model_seed{args.seed}.pth",
        v3_init_path="backup_v3/best_model_v3_ep58_td4.30.pth",
    )