"""
训练脚本 —— 训练轨迹预测模型。
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


def train_epoch(model, loader, optimizer, device, epoch, total_epochs):
    """训练一个 epoch，teacher forcing 比例逐渐降低。"""
    model.train()
    total_loss = 0.0
    # Teacher forcing: 前 25% 训练用全 TF，之后线性降至 0
    progress = min(1.0, epoch / (total_epochs * 0.25))
    tf_ratio = max(0.0, 1.0 - progress)
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()
        pred = model(x, target=y, teacher_forcing_ratio=tf_ratio)
        loss = masked_mse_loss(pred, y, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader), tf_ratio


@torch.no_grad()
def validate(model, loader, device):
    """验证，不使用 teacher forcing。"""
    model.eval()
    total_loss = 0.0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        pred = model(x)
        loss = masked_mse_loss(pred, y, mask)
        total_loss += loss.item()
    return total_loss / len(loader)


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
    print(f"可训练参数量: {total_params:,}")

    # ── 优化器和调度器 ──
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS)

    # ── 日志 ──
    log_file = open(LOG_PATH, "w", encoding="utf-8")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"训练开始: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"训练样本: {len(train_X)}, 验证样本: {len(val_X)}, 测试样本: {len(test_X)}")
    log(f"Hidden size: {model.hidden_size}, LSTM layers: {model.num_layers}, Batch: {BATCH_SIZE}, Peak LR: {LEARNING_RATE}")
    log(f"Warmup: {WARMUP_EPOCHS} epochs, Weight decay: {WEIGHT_DECAY}")
    log("-" * 60)

    # ── 训练循环 ──
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    pbar = tqdm(range(1, EPOCHS + 1), desc="训练", unit="epoch")

    for epoch in pbar:
        t0 = time.time()

        train_loss, tf_ratio = train_epoch(model, train_loader, optimizer, DEVICE, epoch, EPOCHS)
        val_loss = validate(model, val_loader, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({"train": f"{train_loss:.6f}", "val": f"{val_loss:.6f}", "tf": f"{tf_ratio:.2f}", "lr": f"{current_lr:.2e}"})

        log(f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"TF: {tf_ratio:.2f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
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
