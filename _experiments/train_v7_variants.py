"""
v7 训练：两个变体 v7a (hidden 256) 和 v7b (hidden 320)，从 v3 best 权重初始化
"""
import os
import sys
import time
import math
import numpy as np
import torch
import torch.optim as optim
import scipy.io as sio
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
from models.pinn_lstm_v7 import create_pinn_lstm_v7


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


def train_epoch(model, loader, optimizer, device, epoch, total_epochs, scaler):
    model.train()
    total_loss = 0.0
    total_l_pred = 0.0
    total_l_physics = 0.0
    total_l_mode = 0.0
    total_l_terminal = 0.0
    total_l_bound = 0.0
    total_cw_input = 0.0
    total_cw_pred = 0.0
    total_dv_change = 0.0
    total_dv_align = 0.0
    total_grad_norm = 0.0

    progress = min(1.0, epoch / (total_epochs * 0.25))
    tf_ratio = max(0.0, 1.0 - progress)

    lambda_physics = get_physics_weight(
        epoch, PHYSICS_WARMUP_EPOCHS,
        PHYSICS_LOSS_WEIGHT, PHYSICS_LOSS_WEIGHT_FINAL
    )
    lambda_mode = get_physics_weight(
        epoch, PHYSICS_WARMUP_EPOCHS,
        MODE_LOSS_WEIGHT, MODE_LOSS_WEIGHT_FINAL
    )
    lambda_bound = get_physics_weight(
        epoch, DELTAV_BOUND_WARMUP_EPOCHS,
        DELTAV_BOUND_WEIGHT, DELTAV_BOUND_WEIGHT_FINAL
    )

    use_physics = PHYSICS_ENABLED and lambda_physics > 0

    lambda_terminal = get_physics_weight(
        epoch, TERMINAL_WARMUP_EPOCHS,
        TERMINAL_LOSS_WEIGHT, TERMINAL_LOSS_WEIGHT_FINAL
    )
    use_terminal_loss = lambda_terminal > 0

    _terminal_mean = None
    _terminal_std = None
    if use_terminal_loss and scaler is not None:
        _terminal_mean = torch.from_numpy(scaler.mean.astype('float32')).to(device)
        _terminal_std = torch.from_numpy(scaler.std.astype('float32')).to(device)

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()

        pred, dv_all = model(x, target=y, teacher_forcing_ratio=tf_ratio,
                             return_dv=True, mask=mask)

        l_pred = masked_mse_loss(pred, y, mask)
        loss = l_pred
        total_l_pred += l_pred.item()

        l_dv_align = torch.tensor(0.0, device=pred.device)
        if use_physics and dv_all is not None and lambda_mode > 0:
            dv_cw_input = dv_all[:, :9]
            dv_model_seq = dv_all[:, 9:18]
            Bv, Tv, Dv = dv_model_seq.shape
            max_N_local = Dv // 3
            dm = dv_model_seq.reshape(Bv, Tv, max_N_local, 3)
            dc = dv_cw_input.reshape(Bv, Tv, max_N_local, 3)
            diff = (dm - dc).pow(2)
            agent_mse = diff.mean(dim=(1, 3))
            valid = mask.reshape(Bv, max_N_local, 6).any(dim=-1).float()
            l_dv_align = (agent_mse * valid).sum() / valid.sum().clamp(min=1)
            loss = loss + lambda_mode * l_dv_align
            total_l_mode += l_dv_align.item()

        if use_terminal_loss and scaler is not None:
            pos_indices = [i * 6 + j for i in range(MAX_N) for j in range(3)]
            last_pred_pos = pred[:, -1:, pos_indices].squeeze(1)
            last_true_pos = y[:, -1:, pos_indices].squeeze(1)
            pos_mask = mask[:, pos_indices]

            if TERMINAL_PHYSICAL and scaler is not None:
                _term_mean_local = torch.from_numpy(scaler.mean.astype('float32')).to(device)
                _term_std_local = torch.from_numpy(scaler.std.astype('float32')).to(device)
                eps = 1e-8
                last_pred_pos_n = last_pred_pos.squeeze(1)
                last_true_pos_n = last_true_pos.squeeze(1)
                pos_mean_l = _term_mean_local[pos_indices].to(device)
                pos_std_l = _term_std_local[pos_indices].to(device)
                last_pred_phys = last_pred_pos_n * (pos_std_l + eps) + pos_mean_l
                last_true_phys = last_true_pos_n * (pos_std_l + eps) + pos_mean_l
                B = pred.shape[0]
                max_N_local = MAX_N
                pp = last_pred_phys.reshape(B, max_N_local, 3)
                pt = last_true_phys.reshape(B, max_N_local, 3)
                dist = torch.norm(pp - pt, dim=-1)
                valid_pos = pos_mask.reshape(B, max_N_local, 3).any(dim=-1).float()
                l_terminal = (dist * valid_pos).sum() / valid_pos.sum().clamp(min=1)
            else:
                l_terminal = masked_mse_loss(last_pred_pos, last_true_pos, pos_mask)

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

    n_batches = len(loader)
    return {
        "total": total_loss / n_batches,
        "pred": total_l_pred / n_batches,
        "mode": total_l_mode / n_batches,
        "terminal": total_l_terminal / n_batches,
        "grad_norm": total_grad_norm / n_batches,
        "lambda_t": lambda_terminal,
    }


@torch.no_grad()
def validate(model, loader, device, scaler):
    model.eval()
    total_loss = 0.0
    total_l_pred = 0.0
    total_l_terminal = 0.0
    n_batches = 0

    all_preds = []
    all_targets = []
    all_masks = []

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        pred, _ = model(x, return_dv=True, mask=mask)

        l_pred = masked_mse_loss(pred, y, mask)
        l_total = l_pred
        total_l_pred += l_pred.item()

        pos_indices = [i * 6 + j for i in range(MAX_N) for j in range(3)]
        _term_mean_local = torch.from_numpy(scaler.mean.astype('float32')).to(device)
        _term_std_local = torch.from_numpy(scaler.std.astype('float32')).to(device)
        eps = 1e-8
        last_pred_pos = pred[:, -1:, pos_indices].squeeze(1)
        last_true_pos = y[:, -1:, pos_indices].squeeze(1)
        pos_mask = mask[:, pos_indices]
        pos_mean_l = _term_mean_local[pos_indices].to(device)
        pos_std_l = _term_std_local[pos_indices].to(device)
        last_pred_phys = last_pred_pos * (pos_std_l + eps) + pos_mean_l
        last_true_phys = last_true_pos * (pos_std_l + eps) + pos_mean_l
        B = pred.shape[0]
        max_N_local = MAX_N
        pp = last_pred_phys.reshape(B, max_N_local, 3)
        pt = last_true_phys.reshape(B, max_N_local, 3)
        dist = torch.norm(pp - pt, dim=-1)
        valid_pos = pos_mask.reshape(B, max_N_local, 3).any(dim=-1).float()
        l_terminal = (dist * valid_pos).sum() / valid_pos.sum().clamp(min=1)
        l_total = l_total + 2.0 * l_terminal
        total_l_terminal += l_terminal.item()
        total_loss += l_total.item()
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
        "total": total_loss / n_batches,
        "pred": total_l_pred / n_batches,
        "terminal": total_l_terminal / n_batches,
        "terminal_dist_mean": float(dists.mean()),
        "terminal_dist_min": float(dists.min()),
        "terminal_dist_max": float(dists.max()),
        "terminal_dist_median": float(np.median(dists)),
    }


def train_variant(variant_name, hidden_size, num_layers, init_from_v3_path,
                  save_model_path, save_log_path, epochs=EPOCHS):
    """训练一个 v7 变体"""
    print("=" * 60)
    print(f"v7 {variant_name} 训练: hidden={hidden_size}, layers={num_layers}")
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

    print(f"\n使用设备: {DEVICE}")
    model = create_pinn_lstm_v7(hidden_size=hidden_size, num_layers=num_layers, device=DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")

    # 加载 v3 best 作为初始化（手动跳过 size mismatch 的 key）
    if os.path.exists(init_from_v3_path):
        v3_ckpt = torch.load(init_from_v3_path, map_location=DEVICE)
        v3_state = v3_ckpt["model_state_dict"]
        model_state = model.state_dict()
        # 仅加载形状匹配的 key
        loaded = 0
        skipped = 0
        for key, v_param in v3_state.items():
            if key in model_state and model_state[key].shape == v_param.shape:
                model_state[key] = v_param
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(model_state)
        print(f"v3 best 权重部分加载（loaded={loaded}, skipped={skipped}，如 shape 不匹配跳过）")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_EPOCHS, epochs)

    start_epoch = 1
    best_val_loss = float("inf")
    best_terminal_dist = float("inf")
    best_epoch = 0
    patience_counter = 0

    log_file = open(save_log_path, "w", encoding="utf-8")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"\n{variant_name} 训练开始: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Batch: {BATCH_SIZE}, Peak LR: {LEARNING_RATE}")
    log("-" * 60)

    pbar = tqdm(range(start_epoch, epochs + 1), desc=f"{variant_name}", unit="epoch")

    for epoch in pbar:
        t0 = time.time()
        train_info = train_epoch(model, train_loader, optimizer, DEVICE, epoch, epochs, scaler)
        val_info = validate(model, val_loader, DEVICE, scaler)
        scheduler.step()
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        td = val_info['terminal_dist_mean']
        log(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_info['total']:.6f} (pred={train_info['pred']:.6f} term={train_info['terminal']:.6f}) | "
            f"Val: {val_info['total']:.6f} | "
            f"td={td:.3f}km | "
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
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_info["pred"],
                "val_terminal_dist": td,
                "best_terminal_dist": best_terminal_dist,
                "model_type": f"pinn_lstm_v7_{variant_name}",
                "hidden_size": hidden_size,
                "num_layers": num_layers,
            }, save_model_path)
            log(f"  >> 最佳模型已保存（td={best_terminal_dist:.4f} km）")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                log(f"\n早停触发，最佳: td={best_terminal_dist:.4f} km @ epoch {best_epoch}")
                break

    log_file.close()
    print(f"\n{variant_name} 训练完成，最佳 td={best_terminal_dist:.4f} km")
    return best_terminal_dist


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="a", choices=["a", "b", "both"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    if args.variant in ["a", "both"]:
        train_variant(
            variant_name="a (hidden=256, 3 layers)",
            hidden_size=256,
            num_layers=3,
            init_from_v3_path="backup_v3/best_model_v3_ep58_td4.30.pth",
            save_model_path="backup_v7/best_model_v7a.pth",
            save_log_path="backup_v7/train_v7a_log.txt",
            epochs=args.epochs,
        )

    if args.variant in ["b", "both"]:
        train_variant(
            variant_name="b (hidden=320, 4 layers)",
            hidden_size=320,
            num_layers=4,
            init_from_v3_path="backup_v3/best_model_v3_ep58_td4.30.pth",
            save_model_path="backup_v7/best_model_v7b.pth",
            save_log_path="backup_v7/train_v7b_log.txt",
            epochs=args.epochs,
        )