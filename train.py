"""
训练脚本 —— 训练轨迹预测模型。
支持原版 LSTM 和物理信息条件 LSTM 两种模式。
用法: python train.py
"""

import os
import time
import math
import torch
import torch.optim as optim
from tqdm import tqdm

from config import (
    DATA_DIR, OUTPUT_DIR, MODEL_SAVE_PATH, SCALER_SAVE_PATH, LOG_PATH,
    DEVICE, BATCH_SIZE, LEARNING_RATE, MIN_LR, WEIGHT_DECAY, EPOCHS,
    EARLY_STOP_PATIENCE, WARMUP_EPOCHS,
    PHYSICS_ENABLED, PHYSICS_LOSS_WEIGHT, PHYSICS_LOSS_WEIGHT_FINAL,
    PHYSICS_WARMUP_EPOCHS, MODE_LOSS_WEIGHT, MODE_LOSS_WEIGHT_FINAL,
    PRED_WARMUP_EPOCHS, DELTAV_LIMIT, CW_N, CW_DT_H, CONDITION_EMBED_DIM,
    RESUME_TRAINING,
)
from utils.data_loader import (
    load_and_split, create_dataloaders, masked_mse_loss,
)
from models.model import create_model


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=MIN_LR):
    """Cosine annealing + warmup 学习率调度器。"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / LEARNING_RATE, 0.5 * (1 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_physics_weight(epoch, warmup_epochs, initial, final):
    """物理损失权重 warmup 调度: 从 initial 线性增长到 final。"""
    if epoch < PRED_WARMUP_EPOCHS:
        return 0.0
    progress = min(1.0, (epoch - PRED_WARMUP_EPOCHS) / max(1, warmup_epochs))
    return initial + (final - initial) * progress


def train_epoch(model, loader, optimizer, device, epoch, total_epochs,
                physics_loss_fn=None):
    """训练一个 epoch，teacher forcing 比例逐渐降低。"""
    model.train()
    total_loss = 0.0
    total_l_pred = 0.0
    total_l_physics = 0.0
    total_l_mode = 0.0

    # Teacher forcing: 前 25% 训练用全 TF，之后线性降至 0
    progress = min(1.0, epoch / (total_epochs * 0.25))
    tf_ratio = max(0.0, 1.0 - progress)

    # 物理损失权重（warmup）
    lambda_physics = get_physics_weight(
        epoch, PHYSICS_WARMUP_EPOCHS,
        PHYSICS_LOSS_WEIGHT, PHYSICS_LOSS_WEIGHT_FINAL
    )
    lambda_mode = get_physics_weight(
        epoch, PHYSICS_WARMUP_EPOCHS,
        MODE_LOSS_WEIGHT, MODE_LOSS_WEIGHT_FINAL
    )

    use_physics = PHYSICS_ENABLED and physics_loss_fn is not None and lambda_physics > 0

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()

        if PHYSICS_ENABLED:
            pred, dv_all = model(x, target=y, teacher_forcing_ratio=tf_ratio,
                                 return_dv=True, mask=mask)
        else:
            pred = model(x, target=y, teacher_forcing_ratio=tf_ratio)
            dv_all = None

        # 预测损失
        l_pred = masked_mse_loss(pred, y, mask)
        loss = l_pred
        total_l_pred += l_pred.item()

        # 物理损失（在原始量纲空间计算，通过 scaler 反标准化）
        if use_physics and dv_all is not None:
            phys_losses = physics_loss_fn(
                pred_states=pred,
                target_states=y,
                input_states=x,
                mask=mask,
                compute_all=(epoch > PHYSICS_WARMUP_EPOCHS),
            ) # type: ignore
            l_cw = phys_losses["cw_input"] + phys_losses["cw_pred"]
            l_dv = phys_losses["dv_change"]

            l_physics = l_cw
            l_bound = l_dv

            loss = loss + lambda_physics * l_physics + lambda_mode * l_bound
            total_l_physics += l_physics.item()
            total_l_mode += l_bound.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # type: ignore
        optimizer.step()
        total_loss += loss.item()

    n_batches = len(loader)
    loss_info = {
        "total": total_loss / n_batches,
        "pred": total_l_pred / n_batches,
        "physics": total_l_physics / n_batches,
        "mode": total_l_mode / n_batches,
        "tf_ratio": tf_ratio,
        "lambda_p": lambda_physics,
        "lambda_m": lambda_mode,
    }
    return loss_info


@torch.no_grad()
def validate(model, loader, device, physics_loss_fn=None):
    """验证，不使用 teacher forcing。"""
    model.eval()
    total_loss = 0.0
    total_l_pred = 0.0
    total_l_physics = 0.0
    total_l_mode = 0.0

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        if PHYSICS_ENABLED:
            pred, dv_all = model(x, return_dv=True, mask=mask)
        else:
            pred = model(x)
            dv_all = None

        # 预测损失
        l_pred = masked_mse_loss(pred, y, mask)
        l_total = l_pred
        total_l_pred += l_pred.item()

        # 物理损失（验证时始终计算，但不影响早停判断）
        if PHYSICS_ENABLED and physics_loss_fn is not None and dv_all is not None:
            phys_losses = physics_loss_fn(
                pred_states=pred,
                target_states=y,
                input_states=x,
                mask=mask,
                compute_all=True,
            )
            total_l_physics += (phys_losses["cw_input"].item() + phys_losses["cw_pred"].item())
            total_l_mode += phys_losses["dv_change"].item()

        total_loss += l_total.item()

    n_batches = len(loader)
    return {
        "total": total_loss / n_batches,
        "pred": total_l_pred / n_batches,
        "physics": total_l_physics / n_batches,
        "mode": total_l_mode / n_batches,
    }


def load_pretrained_encoder(model, checkpoint_path, device):
    """
    尝试从预训练权重加载 Encoder LSTM 参数。
    由于新模型的输入维度不同（扩展了条件维度），仅加载兼容的参数。
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        pretrained_state = checkpoint["model_state_dict"]

        # 筛选可加载的参数
        model_state = model.state_dict()
        loaded_keys = []
        skipped_keys = []

        for key in model_state:
            if key in pretrained_state:
                if model_state[key].shape == pretrained_state[key].shape:
                    model_state[key] = pretrained_state[key]
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(key)

        model.load_state_dict(model_state)
        print(f"预训练权重加载: {len(loaded_keys)} 层匹配, {len(skipped_keys)} 层形状不兼容")
        if skipped_keys:
            print(f"  跳过的层: {skipped_keys}")
        return True
    except FileNotFoundError:
        print(f"预训练权重不存在: {checkpoint_path}，使用随机初始化")
        return False
    except Exception as e:
        print(f"预训练权重加载失败: {e}，使用随机初始化")
        return False


def train():
    # ── 加载数据 ──
    print("=" * 60)
    print("加载数据...")
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

    # ── 创建模型 ──
    print(f"\n使用设备: {DEVICE}")
    model = create_model(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型类型: {'物理信息条件 LSTM' if PHYSICS_ENABLED else '标准 LSTM'}")
    print(f"可训练参数量: {total_params:,}")

    # ── 物理损失模块（传入 scaler 以在物理空间计算损失）──
    physics_loss_fn = None
    if PHYSICS_ENABLED:
        from models.physics_loss import PhysicsLoss
        physics_loss_fn = PhysicsLoss(
            scaler=scaler, n=CW_N, dt_h=CW_DT_H,
            delta_v_limit=DELTAV_LIMIT, device=DEVICE,
        ).to(DEVICE)
        print(f"物理损失模块已初始化 (CW n={CW_N:.6f}, dt_h={CW_DT_H}s, Δv_limit={DELTAV_LIMIT}m/s)")

    # ── 优化器和调度器 ──
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS)

    # ── 从检查点恢复 ──
    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    if RESUME_TRAINING and os.path.exists(MODEL_SAVE_PATH):
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]
        best_epoch = checkpoint["epoch"]
        # 恢复学习率调度器状态
        for _ in range(checkpoint["epoch"]):
            scheduler.step()
        print(f"从检查点恢复: epoch {checkpoint['epoch']}, val_loss={best_val_loss:.6f}, "
              f"当前 LR={scheduler.get_last_lr()[0]:.2e}")
    elif PHYSICS_ENABLED and os.path.exists(MODEL_SAVE_PATH):
        load_pretrained_encoder(model, MODEL_SAVE_PATH, DEVICE)

    # ── 日志 ──
    log_file = open(LOG_PATH, "a", encoding="utf-8") if (RESUME_TRAINING and start_epoch > 1) else open(LOG_PATH, "w", encoding="utf-8")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"\n训练开始/恢复: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"模型类型: {'PINN-LSTM' if PHYSICS_ENABLED else '标准LSTM'}")
    log(f"训练样本: {len(train_X)}, 验证样本: {len(val_X)}, 测试样本: {len(test_X)}")
    log(f"Hidden size: {model.hidden_size}, LSTM layers: {model.num_layers}, Batch: {BATCH_SIZE}, Peak LR: {LEARNING_RATE}")
    log(f"Warmup: {WARMUP_EPOCHS} epochs, Weight decay: {WEIGHT_DECAY}")
    if PHYSICS_ENABLED:
        log(f"物理损失权重: {PHYSICS_LOSS_WEIGHT} → {PHYSICS_LOSS_WEIGHT_FINAL} (warmup {PHYSICS_WARMUP_EPOCHS} ep)")
        log(f"模式损失权重: {MODE_LOSS_WEIGHT} → {MODE_LOSS_WEIGHT_FINAL}")
        embed_dim = getattr(model, 'condition_embed_dim', CONDITION_EMBED_DIM)
        log(f"Δv 上限: {DELTAV_LIMIT} m/s, 条件嵌入维度: {embed_dim}")
        log(f"预测预热: 前 {PRED_WARMUP_EPOCHS} epoch 仅使用 L_pred")
    log(f"恢复模式: start_epoch={start_epoch}, best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}")
    log("-" * 60)

    # ── 训练循环 ──
    pbar = tqdm(range(start_epoch, EPOCHS + 1), desc="训练", unit="epoch")

    for epoch in pbar:
        t0 = time.time()

        train_info = train_epoch(
            model, train_loader, optimizer, DEVICE, epoch, EPOCHS,
            physics_loss_fn=physics_loss_fn,
        )
        val_info = validate(model, val_loader, DEVICE, physics_loss_fn=physics_loss_fn)
        scheduler.step()
        elapsed = time.time() - t0

        current_lr = scheduler.get_last_lr()[0]

        # 进度条更新
        pbar.set_postfix({
            "train": f"{train_info['total']:.6f}",
            "val": f"{val_info['total']:.6f}",
            "tf": f"{train_info['tf_ratio']:.2f}",
            "lr": f"{current_lr:.2e}",
        })

        # 日志输出
        if PHYSICS_ENABLED:
            log(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"Train: {train_info['total']:.6f} (pred={train_info['pred']:.6f} "
                f"phy={train_info['physics']:.6f} mode={train_info['mode']:.6f}) | "
                f"Val: {val_info['total']:.6f} (pred={val_info['pred']:.6f} "
                f"phy={val_info['physics']:.6f}) | "
                f"TF: {train_info['tf_ratio']:.2f} | "
                f"λ_p={train_info['lambda_p']:.3f} λ_m={train_info['lambda_m']:.3f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )
        else:
            log(f"Epoch {epoch:3d}/{EPOCHS} | "
                f"Train: {train_info['total']:.6f} | "
                f"Val: {val_info['total']:.6f} | "
                f"TF: {train_info['tf_ratio']:.2f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s")

        # 早停与模型保存（基于验证预测损失）
        val_pred_loss = val_info["pred"]
        if val_pred_loss < best_val_loss:
            best_val_loss = val_pred_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_pred_loss,
                "model_type": "pinn_lstm" if PHYSICS_ENABLED else "lstm",
            }, MODEL_SAVE_PATH)
            log(f"  >> 最佳模型已保存")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                log(f"\n早停触发，最佳 epoch: {best_epoch}, 最佳 val_loss: {best_val_loss:.6f}")
                break

    log_file.close()
    print(f"\n训练完成，最佳模型: epoch {best_epoch}, val_loss={best_val_loss:.6f}")
    print(f"模型已保存至: {MODEL_SAVE_PATH}")
    return model, scaler, test_loader


if __name__ == "__main__":
    train()
