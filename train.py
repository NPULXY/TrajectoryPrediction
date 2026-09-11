"""
训练脚本 —— 训练轨迹预测模型。
支持原版 LSTM 和物理信息条件 LSTM 两种模式。
用法: python train.py
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

import time
import math
import numpy as np
import torch
import torch.optim as optim
import scipy.io as sio
from tqdm import tqdm

from config import (
    DATA_DIR, OUTPUT_DIR, MODEL_SAVE_PATH, CHECKPOINT_SAVE_PATH,
    SCALER_SAVE_PATH, LOG_PATH,
    DEVICE, BATCH_SIZE, LEARNING_RATE, MIN_LR, WEIGHT_DECAY, EPOCHS,
    EARLY_STOP_PATIENCE, WARMUP_EPOCHS,
    PHYSICS_ENABLED, USE_TRANSFORMER, PHYSICS_LOSS_WEIGHT, PHYSICS_LOSS_WEIGHT_FINAL,
    PHYSICS_WARMUP_EPOCHS, MODE_LOSS_WEIGHT, MODE_LOSS_WEIGHT_FINAL,
    PRED_WARMUP_EPOCHS, DELTAV_LIMIT, CW_N, CW_DT_H, CONDITION_EMBED_DIM,
    RESUME_TRAINING, RESUME_FIXED_LR,
    TERMINAL_LOSS_WEIGHT, TERMINAL_LOSS_WEIGHT_FINAL,
    TERMINAL_WARMUP_EPOCHS, MAX_N, TERMINAL_REF_DIST,
    DELTAV_BOUND_WEIGHT, DELTAV_BOUND_WEIGHT_FINAL, DELTAV_BOUND_WARMUP_EPOCHS,
    TERMINAL_PHYSICAL, OUTPUT_STEPS,
    AUGMENT_ENABLED, AUGMENT_NOISE_STD, AUGMENT_DECAY_EPOCHS,
    SEED, RUN_TAG, PI_HIDDEN_SIZE, PI_NUM_LAYERS,
    POS_LOSS_WEIGHT, POS_LOSS_WEIGHT_FINAL, POS_LOSS_WARMUP_EPOCHS, POS_LOSS_REF_DIST,
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


def huber_loss_per_sample(pred, target, mask, delta=1.0):
    """
    Huber loss 按样本平均（每个样本的 loss = 平均 Huber 损失）。
    用于 outlier-aware 训练：outlier 样本 loss 不会过大。
    """
    diff = (pred - target).abs()  # (B, T, D)
    quadratic = torch.minimum(diff, torch.tensor(delta, device=diff.device))
    linear = diff - quadratic
    loss_per_elem = 0.5 * quadratic.pow(2) + delta * linear  # (B, T, D)
    # 每个样本的 loss
    mask_expanded = mask.unsqueeze(1).expand_as(loss_per_elem)
    # 避免除零
    n_valid_per_sample = mask_expanded.float().sum(dim=(1, 2)).clamp(min=1)  # (B,)
    loss_per_sample = (loss_per_elem * mask_expanded).sum(dim=(1, 2)) / n_valid_per_sample  # (B,)
    return loss_per_sample  # (B,)


def physical_position_loss(pred, target, scaler_mean, scaler_std, pos_indices, mask, ref_dist):
    """
    全步物理空间位置误差损失（方向3：损失口径对齐，2026-09-11 新增）。

    计算预测与真值在**原始物理空间**的位置 3D 距离（km），按参考距离归一化。
    与 `multi_step_terminal_loss` 的区别：本项覆盖**全部 10 步**（后者只覆盖 t=3/6/9 与末步）。

    动机：评估指标为物理空间位置 RMSE，而 `l_pred` 是标准化空间 Huber 损失；
    标准化按各维 std 缩放（位置 std≈50 km、速度 std≈0.05 km/s），使速度误差在损失中
    被显著放大，与评估口径不一致。本项使优化目标与评估指标对齐。

    步长权重从 0.2 线性升至 1.0 —— 误差随预测步长增长（实测 0.76 km → 2.04 km），
    后期步更需监督。

    Args:
        pred, target: (B, 10, max_dim) 标准化空间
        scaler_mean/std: (max_dim,) numpy → tensor
        pos_indices: 位置维度索引（12 个）
        mask: (B, max_dim) bool
        ref_dist: 归一化参考距离 (km)
    Returns:
        标量损失（无量纲）
    """
    B = pred.shape[0]
    eps = 1e-8
    pm = scaler_mean[pos_indices].to(pred.device)
    ps = scaler_std[pos_indices].to(pred.device)

    # 反归一化到 km，并重组为 (B, 10, max_N, 3)
    pp = (pred[:, :, pos_indices] * (ps + eps) + pm).reshape(B, OUTPUT_STEPS, MAX_N, 3)
    pt = (target[:, :, pos_indices] * (ps + eps) + pm).reshape(B, OUTPUT_STEPS, MAX_N, 3)

    d = torch.norm(pp - pt, dim=-1)                     # (B, 10, max_N)  km

    # 步长权重 0.2 → 1.0（后期步权重更高）
    w = torch.linspace(0.2, 1.0, OUTPUT_STEPS, device=pred.device).view(1, -1, 1)

    valid = mask[:, pos_indices].reshape(B, MAX_N, 3).any(dim=-1).float()  # (B, max_N)
    vm = valid.unsqueeze(1)                             # (B, 1, max_N)

    num = (d * w * vm).sum()
    den = (w * vm).sum().clamp(min=1)
    return num / den / ref_dist


def multi_step_terminal_loss(pred, target, scaler_mean, scaler_std, pos_indices, mask):
    """
    阶梯式末距损失：t=3, 6, 9 各贡献 3D 距离损失（归一化到参考距离）。
    让模型不仅关注末步，中间步也对齐。

    改进 C（2026-09-10）：3D 距离除以 TERMINAL_REF_DIST（km），让 l_terminal 量级从
    ~4.0 km 降到 ~1.0（无量纲），与 l_pred（~0.008）量级差距从 ~500× 降到 ~100×。
    """
    B = pred.shape[0]
    max_N_local = MAX_N
    eps = 1e-8
    pos_mean = scaler_mean[pos_indices].to(pred.device)
    pos_std = scaler_std[pos_indices].to(pred.device)

    total_loss = 0.0
    weight_sum = 0.0
    for t_idx, t_step in enumerate([2, 5, 8]):  # t=3, 6, 9 (0-indexed)
        pred_t = pred[:, t_step:t_step+1, pos_indices].squeeze(1)  # (B, 12)
        true_t = target[:, t_step:t_step+1, pos_indices].squeeze(1)  # (B, 12)
        # 反归一化到物理空间 (km)
        pred_phys = pred_t * (pos_std + eps) + pos_mean  # (B, 12)
        true_phys = true_t * (pos_std + eps) + pos_mean  # (B, 12)
        # reshape (B, max_N, 3)
        pp = pred_phys.reshape(B, max_N_local, 3)
        pt = true_phys.reshape(B, max_N_local, 3)
        dist = torch.norm(pp - pt, dim=-1)  # (B, max_N) 物理 km
        # 【改进 C】归一化到参考距离 → 量级从 ~4.0 降到 ~1.0
        dist_normalized = dist / TERMINAL_REF_DIST
        # mask
        pos_mask = mask[:, pos_indices]
        valid_pos = pos_mask.reshape(B, max_N_local, 3).any(dim=-1).float()
        loss_t = (dist_normalized * valid_pos).sum() / valid_pos.sum().clamp(min=1)
        # 阶梯权重：t=9 权重最高，t=3 最低
        weight = [0.3, 0.5, 1.0][t_idx]
        total_loss = total_loss + weight * loss_t
        weight_sum = weight_sum + weight
    return total_loss / weight_sum


def train_epoch(model, loader, optimizer, device, epoch, total_epochs,
                physics_loss_fn=None, scaler=None):
    """训练一个 epoch，teacher forcing 比例逐渐降低。"""
    model.train()
    total_loss = 0.0
    total_l_pred = 0.0
    total_l_physics = 0.0
    total_l_mode = 0.0
    total_l_terminal = 0.0
    total_l_pos = 0.0
    total_l_bound = 0.0
    total_cw_input = 0.0
    total_cw_pred = 0.0
    total_dv_change = 0.0
    total_dv_align = 0.0
    total_grad_norm = 0.0

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
    lambda_bound = get_physics_weight(
        epoch, DELTAV_BOUND_WARMUP_EPOCHS,
        DELTAV_BOUND_WEIGHT, DELTAV_BOUND_WEIGHT_FINAL
    )

    use_physics = PHYSICS_ENABLED and physics_loss_fn is not None and lambda_physics > 0

    # 末端距离损失权重（与物理损失共享 PRED_WARMUP，单独 warmup）
    lambda_terminal = get_physics_weight(
        epoch, TERMINAL_WARMUP_EPOCHS,
        TERMINAL_LOSS_WEIGHT, TERMINAL_LOSS_WEIGHT_FINAL
    )
    use_terminal_loss = lambda_terminal > 0

    # 全步位置损失权重（方向3：损失口径对齐）
    lambda_pos = get_physics_weight(
        epoch, POS_LOSS_WARMUP_EPOCHS,
        POS_LOSS_WEIGHT, POS_LOSS_WEIGHT_FINAL
    )

    # 将 scaler 参数缓存为 tensor（若尚未缓存且需要末端损失）
    _terminal_mean = None
    _terminal_std = None
    if use_terminal_loss and scaler is not None:
        _terminal_mean = torch.from_numpy(scaler.mean.astype('float32')).to(device)
        _terminal_std = torch.from_numpy(scaler.std.astype('float32')).to(device)

    # 数据增强噪声强度（随 epoch 线性衰减到 0：前期抗过拟合，后期纯数据精调）
    # 注：噪声在 model.forward 内部施加于 LSTM 输入分支，Δv 估计始终用干净输入
    noise_std = (AUGMENT_NOISE_STD * max(0.0, 1.0 - epoch / max(1, AUGMENT_DECAY_EPOCHS))
                 if AUGMENT_ENABLED else 0.0)

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()

        if PHYSICS_ENABLED:
            pred, dv_all = model(x, target=y, teacher_forcing_ratio=tf_ratio,
                                 return_dv=True, mask=mask, augment_std=noise_std)
        else:
            pred = model(x, target=y, teacher_forcing_ratio=tf_ratio)
            dv_all = None

        # 预测损失（v6: Huber loss per-sample + outlier-aware 加权）
        # 计算每个样本的 Huber loss（标准化空间）
        loss_per_sample = huber_loss_per_sample(pred, y, mask, delta=1.0)  # (B,)
        pos_indices = [i * 6 + j for i in range(MAX_N) for j in range(3)]
        _term_mean_local = torch.from_numpy(scaler.mean.astype('float32')).to(device) if scaler is not None else None
        _term_std_local = torch.from_numpy(scaler.std.astype('float32')).to(device) if scaler is not None else None
        if scaler is not None:
            eps = 1e-8
            last_pred_pos_n = pred[:, -1:, pos_indices].squeeze(1)  # (B, 12)
            last_true_pos_n = y[:, -1:, pos_indices].squeeze(1)
            pos_mean_l = _term_mean_local[pos_indices].to(device)
            pos_std_l = _term_std_local[pos_indices].to(device)
            last_pred_phys = last_pred_pos_n * (pos_std_l + eps) + pos_mean_l
            last_true_phys = last_true_pos_n * (pos_std_l + eps) + pos_mean_l
            B_local = pred.shape[0]
            pp = last_pred_phys.reshape(B_local, MAX_N, 3)
            pt = last_true_phys.reshape(B_local, MAX_N, 3)
            dist = torch.norm(pp - pt, dim=-1)  # (B, MAX_N)
            pos_mask = mask[:, pos_indices]
            valid_pos = pos_mask.reshape(B_local, MAX_N, 3).any(dim=-1).float()
            td_per_sample = (dist * valid_pos).sum(dim=-1) / valid_pos.sum(dim=-1).clamp(min=1)  # (B,)
            # outlier 降权：td > 5 km 的样本权重 0.3，td > 10 km 权重 0.1
            sample_weights = torch.ones_like(td_per_sample)
            sample_weights = torch.where(td_per_sample > 10, torch.tensor(0.1, device=device), sample_weights)
            sample_weights = torch.where((td_per_sample > 5) & (td_per_sample <= 10), torch.tensor(0.3, device=device), sample_weights)
        else:
            sample_weights = torch.ones(pred.shape[0], device=device)
        # 加权平均
        l_pred = (loss_per_sample * sample_weights).sum() / sample_weights.sum().clamp(min=1)
        loss = l_pred
        total_l_pred += l_pred.item()

        # 阶梯式末距损失（t=3, 6, 9）
        if use_terminal_loss and scaler is not None and _term_mean_local is not None:
            l_terminal_multi = multi_step_terminal_loss(pred, y, _term_mean_local, _term_std_local, pos_indices, mask)
            loss = loss + 0.5 * lambda_terminal * l_terminal_multi
            total_l_terminal += l_terminal_multi.item()

        # 全步物理空间位置损失（方向3：使优化目标与"物理空间位置 RMSE"评估口径对齐）
        if lambda_pos > 0 and _term_mean_local is not None:
            l_pos = physical_position_loss(pred, y, _term_mean_local, _term_std_local,
                                           pos_indices, mask, POS_LOSS_REF_DIST)
            loss = loss + lambda_pos * l_pos
            total_l_pos += l_pos.item()

        # dv_alignment loss (内联，避免在 physics_loss_fn 中重复 forward)
        l_dv_align = torch.tensor(0.0, device=pred.device)
        if use_physics and dv_all is not None and lambda_mode > 0:
            dv_cw_input = dv_all[:, :9]   # (B, 9, max_N*3)
            dv_model_seq = dv_all[:, 9:18]  # (B, 9, max_N*3) 取模型前 9 步
            # 仅在有效 agent 上计算 MSE
            Bv, Tv, Dv = dv_model_seq.shape
            max_N = Dv // 3
            dm = dv_model_seq.reshape(Bv, Tv, max_N, 3)
            dc = dv_cw_input.reshape(Bv, Tv, max_N, 3)
            # 按 Δv 上限归一化（无量纲），与 physics_loss.dv_alignment_loss 口径一致
            _dv_ref = DELTAV_LIMIT / 1000.0
            diff = ((dm - dc) / _dv_ref).pow(2)  # (B, T, max_N, 3)
            agent_mse = diff.mean(dim=(1, 3))  # (B, max_N)
            valid = mask.reshape(Bv, max_N, 6).any(dim=-1).float()
            l_dv_align = (agent_mse * valid).sum() / valid.sum().clamp(min=1)
            loss = loss + lambda_mode * l_dv_align
            total_l_mode += l_dv_align.item()

        # 物理损失（CW 残差 + Δv 边界）
        if use_physics and dv_all is not None and (lambda_physics > 0 or lambda_bound > 0):
            # dv_all 结构: [9 步 CW 逆推 Δv, 9 步模型速度差分] -> (B, 18, max_N*3)
            #   注: dv_all[:, :9] = CW 逆推；dv_all[:, 9:18] = 模型预测轨迹的速度差分
            phys_losses = physics_loss_fn(
                pred_states=pred,
                target_states=y,
                input_states=x,
                mask=mask,
                compute_all=(epoch > PHYSICS_WARMUP_EPOCHS),
            ) # type: ignore
            l_cw = phys_losses["cw_input"] + phys_losses["cw_pred"]
            l_dv_change = phys_losses["dv_change"]

            total_cw_input += phys_losses["cw_input"].item()
            total_cw_pred += phys_losses["cw_pred"].item()
            total_dv_change += l_dv_change.item()
            total_dv_align += l_dv_align.item()  # 累加（已经在前面算过）

            # 损失组合：
            # - lambda_physics * L_cw_normalized（分维度归一化的 CW 残差）
            # - lambda_mode * L_dv_align（model Δv 与 CW Δv 对齐，PI-LSTM 核心，前面已加）
            # - lambda_bound * L_dv_change（Δv 边界软约束，原始量纲）
            loss = loss + lambda_physics * l_cw + lambda_bound * l_dv_change
            total_l_physics += l_cw.item()
            total_l_bound += l_dv_change.item()

        # 末端距离损失：强化最后一步位置预测精度（直接对应 terminal_dist）
        if use_terminal_loss:
            # 位置特征索引：每个目标的 x,y,z（共 4 个目标 × 3 = 12 维）
            pos_indices = [i * 6 + j for i in range(MAX_N) for j in range(3)]
            # 取最后一步的预测和真值
            last_pred_pos = pred[:, -1:, pos_indices]   # (B, 1, 12)
            last_true_pos = y[:, -1:, pos_indices]      # (B, 1, 12)
            pos_mask = mask[:, pos_indices]              # (B, 12)

            if TERMINAL_PHYSICAL and scaler is not None:
                # 在原始物理量纲空间计算末步 3D 距离（直接对齐用户目标）
                # 反归一化最后一步位置
                last_pred_pos_n = last_pred_pos.squeeze(1)  # (B, 12)
                last_true_pos_n = last_true_pos.squeeze(1)
                # 提取 scaler 的位置维度 mean/std
                pos_mean = _terminal_mean[pos_indices].to(device)
                pos_std = _terminal_std[pos_indices].to(device)
                eps = 1e-8
                last_pred_phys = last_pred_pos_n * (pos_std + eps) + pos_mean  # (B, 12)
                last_true_phys = last_true_pos_n * (pos_std + eps) + pos_mean  # (B, 12)
                # 重组为 (B, max_N, 3) 计算每个 agent 的 3D 距离
                B = pred.shape[0]
                max_N_local = MAX_N
                pp = last_pred_phys.reshape(B, max_N_local, 3)
                pt = last_true_phys.reshape(B, max_N_local, 3)
                dist = torch.norm(pp - pt, dim=-1)  # (B, max_N) 物理 km
                # 【改进 C】归一化到参考距离 → 量级从 ~4.0 km 降到 ~1.0（无量纲）
                dist_normalized = dist / TERMINAL_REF_DIST
                valid_pos = pos_mask.reshape(B, max_N_local, 3).any(dim=-1).float()
                # 对每个有效 agent 的归一化 3D 距离取均值
                l_terminal = (dist_normalized * valid_pos).sum() / valid_pos.sum().clamp(min=1)
            else:
                l_terminal = masked_mse_loss(last_pred_pos, last_true_pos, pos_mask)

            loss = loss + lambda_terminal * l_terminal
            total_l_terminal += l_terminal.item()

        loss.backward()
        # 记录梯度范数（裁剪前）
        batch_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2).item()
                batch_grad_norm += param_norm ** 2
        total_grad_norm += math.sqrt(batch_grad_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # type: ignore
        optimizer.step()
        total_loss += loss.item()

    n_batches = len(loader)
    loss_info = {
        "total": total_loss / n_batches,
        "pred": total_l_pred / n_batches,
        "physics": total_l_physics / n_batches,
        "mode": total_l_mode / n_batches,
        "terminal": total_l_terminal / n_batches,
        "bound": total_l_bound / n_batches if total_l_bound > 0 else 0.0,
        "cw_input": total_cw_input / n_batches,
        "cw_pred": total_cw_pred / n_batches,
        "dv_change": total_dv_change / n_batches,
        "dv_align": total_dv_align / n_batches if total_dv_align > 0 else 0.0,
        "grad_norm": total_grad_norm / n_batches,
        "tf_ratio": tf_ratio,
        "lambda_p": lambda_physics,
        "lambda_m": lambda_mode,
        "lambda_t": lambda_terminal,
        "lambda_b": lambda_bound,
        "lambda_pos": lambda_pos,
        "pos": total_l_pos / n_batches,
        "noise_std": noise_std,
    }
    return loss_info


@torch.no_grad()
def validate(model, loader, device, physics_loss_fn=None, scaler=None):
    """验证，不使用 teacher forcing。同时计算末端距离、成功率和 Δv 超限率。"""
    model.eval()
    total_loss = 0.0
    total_l_pred = 0.0
    total_l_physics = 0.0
    total_l_mode = 0.0
    total_cw_input = 0.0
    total_cw_pred = 0.0
    total_dv_change = 0.0
    total_dv_align = 0.0

    # 累积用于原始量纲指标计算
    all_preds = []
    all_targets = []
    all_masks = []
    all_dv = []

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

        # dv_alignment loss（内联计算指标）
        if PHYSICS_ENABLED and dv_all is not None:
            dv_cw_input = dv_all[:, :9]
            dv_model_seq = dv_all[:, 9:18]
            Bv, Tv, Dv = dv_model_seq.shape
            max_N = Dv // 3
            dm = dv_model_seq.reshape(Bv, Tv, max_N, 3)
            dc = dv_cw_input.reshape(Bv, Tv, max_N, 3)
            # 按 Δv 上限归一化（同 train_epoch，保持训练/验证口径一致）
            _dv_ref = DELTAV_LIMIT / 1000.0
            diff = ((dm - dc) / _dv_ref).pow(2)
            agent_mse = diff.mean(dim=(1, 3))
            valid = mask.reshape(Bv, max_N, 6).any(dim=-1).float()
            l_dv_align = (agent_mse * valid).sum() / valid.sum().clamp(min=1)
            total_dv_align += l_dv_align.item()

        # 物理损失（验证时始终计算，但不影响早停判断）
        if PHYSICS_ENABLED and physics_loss_fn is not None and dv_all is not None:
            phys_losses = physics_loss_fn(
                pred_states=pred,
                target_states=y,
                input_states=x,
                mask=mask,
                compute_all=True,
            )
            total_cw_input += phys_losses["cw_input"].item()
            total_cw_pred += phys_losses["cw_pred"].item()
            total_dv_change += phys_losses["dv_change"].item()
            total_l_physics += (phys_losses["cw_input"].item() + phys_losses["cw_pred"].item())

        total_loss += l_total.item()

        # 累积用于原始量纲指标
        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())
        all_masks.append(mask.cpu())
        if dv_all is not None:
            all_dv.append(dv_all.cpu())

    n_batches = len(loader)
    result = {
        "total": total_loss / n_batches,
        "pred": total_l_pred / n_batches,
        "physics": total_l_physics / n_batches,
        "mode": total_l_mode / n_batches,
        "cw_input": total_cw_input / n_batches,
        "cw_pred": total_cw_pred / n_batches,
        "dv_change": total_dv_change / n_batches,
        "dv_align": total_dv_align / n_batches,
    }

    # ── 原始量纲指标（末端距离、成功率、Δv 超限率）──
    if PHYSICS_ENABLED and scaler is not None and all_dv:
        preds_t = torch.cat(all_preds, dim=0)
        targets_t = torch.cat(all_targets, dim=0)
        masks_t = torch.cat(all_masks, dim=0)
        dv_t = torch.cat(all_dv, dim=0)

        preds_raw = scaler.inverse_transform(preds_t.numpy())
        targets_raw = scaler.inverse_transform(targets_t.numpy())
        dv_np = dv_t.numpy()
        masks_np = masks_t.numpy()

        # 末端距离：每个样本所有目标中末步 3D 位置误差最大值
        n_samples = preds_raw.shape[0]
        term_dists = []
        for i in range(n_samples):
            n_agents = int(masks_np[i].sum()) // 6
            max_dist = 0.0
            for a in range(n_agents):
                base = a * 6
                dx = preds_raw[i, -1, base + 0] - targets_raw[i, -1, base + 0]
                dy = preds_raw[i, -1, base + 1] - targets_raw[i, -1, base + 1]
                dz = preds_raw[i, -1, base + 2] - targets_raw[i, -1, base + 2]
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                if dist > max_dist:
                    max_dist = dist
            term_dists.append(max_dist)
        term_dists = np.array(term_dists)

        result["terminal_dist_mean"] = float(np.mean(term_dists))
        result["terminal_dist_std"] = float(np.std(term_dists))
        result["terminal_dist_median"] = float(np.median(term_dists))
        result["terminal_dist_min"] = float(np.min(term_dists))
        result["terminal_dist_max"] = float(np.max(term_dists))
        result["success_rate_1km"] = float(np.mean(term_dists < 1.0))
        result["success_rate_100m"] = float(np.mean(term_dists < 0.1))

        # Δv 超限率（从预测轨迹的速度变化计算原始量纲 Δv）
        dv_mags_all = []
        for i in range(n_samples):
            n_agents = int(masks_np[i].sum()) // 6
            for a in range(n_agents):
                base = a * 6
                vel = preds_raw[i, :, base+3:base+6]  # (10, 3)
                dv = vel[1:] - vel[:-1]                # (9, 3)
                mags = np.linalg.norm(dv, axis=-1)     # (9,)
                dv_mags_all.extend(mags.tolist())
        if dv_mags_all:
            dv_mags_arr = np.array(dv_mags_all)
            result["dv_over_limit_rate"] = float(np.mean(dv_mags_arr > (DELTAV_LIMIT / 1000.0)))
            result["dv_mag_mean"] = float(np.mean(dv_mags_arr) * 1000)  # m/s
            result["dv_mag_max"] = float(np.max(dv_mags_arr) * 1000)    # m/s

    return result


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
    # ── 随机种子（2026-09-11）──
    # 数据划分由 load_and_split 内部固定的 RANDOM_SEED 决定（不受此处影响），
    # 故本种子只影响**模型初始化与 dropout 采样**，从而保证
    # 「同一数据划分、不同初始权重」—— 这正是集成所需多样性的正确来源。
    if SEED > 0:
        import random as _random
        _random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        print(f"[seed] 已固定随机种子 SEED={SEED}")
    else:
        print("[seed] SEED=0，使用随机初始化（每次训练不同）")
    if RUN_TAG:
        print(f"[tag] RUN_TAG={RUN_TAG}，产物将带此后缀")

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
    # 传入 scaler：PI-LSTM 需将标准化输入还原到物理空间才能正确估计 Δv
    model = create_model(DEVICE, scaler)
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
    best_terminal_dist = float("inf")
    best_epoch = 0
    patience_counter = 0

    # 存放训练历史（可能从 checkpoint 恢复部分记录）
    train_history = None
    val_history = None
    epoch_list = None

    checkpoint_loaded = False
    if RESUME_TRAINING:
        # 优先加载最近存档点（latest_checkpoint），其次加载最佳模型
        ckpt_path = None
        if os.path.exists(CHECKPOINT_SAVE_PATH):
            ckpt_path = CHECKPOINT_SAVE_PATH
        elif os.path.exists(MODEL_SAVE_PATH):
            ckpt_path = MODEL_SAVE_PATH

        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=DEVICE)
            ckpt_state = checkpoint["model_state_dict"]
            model_keys = set(model.state_dict().keys())
            missing = sorted(model_keys - set(ckpt_state.keys()))
            if missing:
                # ⚠️ 2026-09-10：不再静默崩溃。架构不匹配时必须明确报错并给出可执行修复方案，
                # 否则会退回随机初始化（或直接抛 RuntimeError），两种情况都难以定位。
                raise RuntimeError(
                    f"\n检查点与当前模型架构不匹配：缺失 {len(missing)}/{len(model_keys)} 个参数。\n"
                    f"  检查点 : {ckpt_path}\n"
                    f"  缺失示例: {missing[:5]}\n"
                    f"  常见原因: 切换了 USE_TRANSFORMER / PHYSICS_ENABLED，或检查点来自其他架构。\n"
                    f"  修复方法: 删除或移走以下文件后重新训练，或设 config.RESUME_TRAINING=False\n"
                    f"            {CHECKPOINT_SAVE_PATH}\n"
                    f"            {MODEL_SAVE_PATH}"
                )
            # strict=False：容忍检查点中多余的键（例如已改为非持久化的派生 buffer）
            model.load_state_dict(ckpt_state, strict=False)
            # optimizer_state_dict 可能在重置后缺失（新损失场景），此时保持新优化器
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["val_loss"]
            best_terminal_dist = checkpoint.get("best_terminal_dist", float("inf"))
            best_epoch = checkpoint.get("best_epoch", checkpoint["epoch"])
            patience_counter = checkpoint.get("patience_counter", 0)
            # 恢复学习率调度器完整状态（精确恢复）
            if "scheduler_state_dict" in checkpoint and RESUME_FIXED_LR is None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                # 兼容旧版 checkpoint 或 RESUME_FIXED_LR 不为空：跳过 scheduler 恢复
                # 改用 RESUME_FIXED_LR 强制低 LR（避免 cosine scheduler 跨 epoch 跳变）
                if RESUME_FIXED_LR is not None:
                    for pg in optimizer.param_groups:
                        pg["lr"] = RESUME_FIXED_LR
                    print(f"⚠ 已强制 optimizer LR={RESUME_FIXED_LR}（跳过 scheduler 恢复）")
                else:
                    # 兼容旧版 checkpoint：回退到逐步前进
                    for _ in range(checkpoint["epoch"]):
                        scheduler.step()
            # 恢复部分训练历史（用于中断后最终的 .mat 保存完整性）
            train_history = checkpoint.get("train_history")
            val_history = checkpoint.get("val_history")
            epoch_list = checkpoint.get("epoch_list")
            source = "latest" if ckpt_path == CHECKPOINT_SAVE_PATH else "best"
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"从检查点恢复 [{source} 存档]: epoch {checkpoint['epoch']}, "
                  f"val_loss={best_val_loss:.6f}, "
                  f"当前 LR={current_lr:.2e}")
            checkpoint_loaded = True

    if not checkpoint_loaded and PHYSICS_ENABLED and os.path.exists(MODEL_SAVE_PATH):
        load_pretrained_encoder(model, MODEL_SAVE_PATH, DEVICE)

    # ── 日志 ──
    log_file = open(LOG_PATH, "a", encoding="utf-8") if (RESUME_TRAINING and start_epoch > 1) else open(LOG_PATH, "w", encoding="utf-8")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"\n训练开始/恢复: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"模型类型: {'Transformer-PI (v5)' if USE_TRANSFORMER else 'PINN-LSTM' if PHYSICS_ENABLED else '标准LSTM'}")
    log(f"训练样本: {len(train_X)}, 验证样本: {len(val_X)}, 测试样本: {len(test_X)}")
    # 模型架构信息（兼容 v3 PI-LSTM 和 v5 Transformer-PI）
    if hasattr(model, 'hidden_size'):
        log(f"Hidden size: {model.hidden_size}, LSTM layers: {model.num_layers}, Batch: {BATCH_SIZE}, Peak LR: {LEARNING_RATE}")
    elif hasattr(model, 'd_model'):
        log(f"d_model: {model.d_model}, Batch: {BATCH_SIZE}, Peak LR: {LEARNING_RATE}")
    else:
        log(f"Batch: {BATCH_SIZE}, Peak LR: {LEARNING_RATE}")
    log(f"Warmup: {WARMUP_EPOCHS} epochs, Weight decay: {WEIGHT_DECAY}")
    if PHYSICS_ENABLED:
        log(f"物理损失权重: {PHYSICS_LOSS_WEIGHT} → {PHYSICS_LOSS_WEIGHT_FINAL} (warmup {PHYSICS_WARMUP_EPOCHS} ep)")
        log(f"模式损失权重: {MODE_LOSS_WEIGHT} → {MODE_LOSS_WEIGHT_FINAL}")
        log(f"末端距离损失权重: {TERMINAL_LOSS_WEIGHT} → {TERMINAL_LOSS_WEIGHT_FINAL} (warmup {TERMINAL_WARMUP_EPOCHS} ep)")
        embed_dim = getattr(model, 'condition_embed_dim', CONDITION_EMBED_DIM)
        log(f"Δv 上限: {DELTAV_LIMIT} m/s, 条件嵌入维度: {embed_dim}")
        log(f"预测预热: 前 {PRED_WARMUP_EPOCHS} epoch 仅使用 L_pred")
    log(f"恢复模式: start_epoch={start_epoch}, best_epoch={best_epoch}, "
        f"best_val_loss={best_val_loss:.6f}, best_terminal_dist={best_terminal_dist:.4f} km")
    log("-" * 60)

    # ── 训练历史记录（用于保存 .mat；若从 checkpoint 恢复则续接）──
    # 兼容旧 checkpoint：若不含新字段，丢弃重建（保证数据一致性）
    if train_history is not None and "dv_align" not in train_history:
        print("[warn] 旧版 checkpoint 不含新字段 (dv_align/bound/lambda_b)，重建训练历史")
        train_history = None
    if val_history is not None and "dv_align" not in val_history:
        val_history = None
    if train_history is None:
        train_history = {
            "total": [], "pred": [], "physics": [], "mode": [], "terminal": [], "bound": [],
            "pos": [],
            "cw_input": [], "cw_pred": [], "dv_change": [], "dv_align": [],
            "grad_norm": [],
            "tf_ratio": [], "lambda_p": [], "lambda_m": [], "lambda_t": [], "lambda_b": [],
            "lambda_pos": [],
            "lr": [], "time": [],
        }
    if val_history is None:
        val_history = {
            "total": [], "pred": [], "physics": [], "mode": [],
            "cw_input": [], "cw_pred": [], "dv_change": [], "dv_align": [],
            "terminal_dist_mean": [], "terminal_dist_std": [],
            "terminal_dist_median": [], "terminal_dist_min": [], "terminal_dist_max": [],
            "success_rate_1km": [], "success_rate_100m": [],
            "dv_over_limit_rate": [], "dv_mag_mean": [], "dv_mag_max": [],
        }
    if epoch_list is None:
        epoch_list = []

    # ── 训练循环 ──
    pbar = tqdm(range(start_epoch, EPOCHS + 1), desc="训练", unit="epoch")

    for epoch in pbar:
        t0 = time.time()

        train_info = train_epoch(
            model, train_loader, optimizer, DEVICE, epoch, EPOCHS,
            physics_loss_fn=physics_loss_fn, scaler=scaler,
        )
        val_info = validate(model, val_loader, DEVICE, physics_loss_fn=physics_loss_fn,
                            scaler=scaler)

        scheduler.step()
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        # 记录训练历史（含学习率和耗时）
        train_info["lr"] = current_lr
        train_info["time"] = elapsed
        epoch_list.append(epoch)
        for key in train_history:
            train_history[key].append(train_info[key])
        for key in val_history:
            val_history[key].append(val_info.get(key, 0.0))

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
                f"phy={train_info['physics']:.6f} term={train_info['terminal']:.6f} "
                f"mode={train_info['mode']:.6f} bound={train_info.get('bound', 0.0):.6f} "
                f"pos={train_info.get('pos', 0.0):.6f}) | "
                f"Val: {val_info['total']:.6f} (pred={val_info['pred']:.6f} "
                f"phy={val_info['physics']:.6f} dvalign={val_info.get('dv_align', 0.0):.6f}) | "
                f"td={val_info.get('terminal_dist_mean', 0.0):.3f}km | "
                f"TF: {train_info['tf_ratio']:.2f} | "
                f"λ_p={train_info['lambda_p']:.3f} λ_m={train_info['lambda_m']:.3f} "
                f"λ_t={train_info['lambda_t']:.3f} λ_b={train_info.get('lambda_b', 0.0):.3f} "
                f"λ_pos={train_info.get('lambda_pos', 0.0):.3f} | "
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

        # 早停与模型保存（基于验证平均末端距离 terminal_dist_mean）
        val_terminal_dist = val_info["terminal_dist_mean"]
        if val_terminal_dist < best_terminal_dist:
            best_terminal_dist = val_terminal_dist
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_info["pred"],
                "val_terminal_dist": val_terminal_dist,
                "best_terminal_dist": best_terminal_dist,
                "model_type": "pinn_lstm" if PHYSICS_ENABLED else "lstm",
            }, MODEL_SAVE_PATH)
            log(f"  >> 最佳模型已保存（terminal_dist_mean={best_terminal_dist:.4f} km）")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                log(f"\n早停触发，最佳 epoch: {best_epoch}, "
                    f"best_terminal_dist={best_terminal_dist:.4f} km, "
                    f"best_val_loss={best_val_loss:.6f}")
                break

        # ── 保存最近存档点（latest_checkpoint），支持中断恢复 ──
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "best_terminal_dist": best_terminal_dist,
            "patience_counter": patience_counter,
            "train_history": train_history,
            "val_history": val_history,
            "epoch_list": epoch_list,
            "model_type": "pinn_lstm" if PHYSICS_ENABLED else "lstm",
        }, CHECKPOINT_SAVE_PATH)

    log_file.close()

    # ── 保存训练历史为 .mat ──
    mat_path = os.path.join(OUTPUT_DIR, f"training_history{('_' + RUN_TAG) if RUN_TAG else ''}.mat")
    try:
        mat_data = {
            # 基本迭代信息
            "epoch": np.array(epoch_list, dtype=np.int32),
            "best_epoch": np.int32(best_epoch),
            "best_val_loss": np.float64(best_val_loss),
            "best_terminal_dist": np.float64(best_terminal_dist),
            # 训练损失各分量
            "train_total": np.array(train_history["total"]),
            "train_pred": np.array(train_history["pred"]),
            "train_physics": np.array(train_history["physics"]),
            "train_mode": np.array(train_history["mode"]),
            "train_terminal": np.array(train_history["terminal"]),
            "train_bound": np.array(train_history["bound"]),
            # 物理损失子分量
            "train_cw_input": np.array(train_history["cw_input"]),
            "train_cw_pred": np.array(train_history["cw_pred"]),
            "train_dv_change": np.array(train_history["dv_change"]),
            "train_dv_align": np.array(train_history["dv_align"]),
            # 梯度与优化
            "train_grad_norm": np.array(train_history["grad_norm"]),
            "train_lr": np.array(train_history["lr"]),
            "train_time": np.array(train_history["time"]),
            # 训练策略
            "train_tf_ratio": np.array(train_history["tf_ratio"]),
            "train_lambda_p": np.array(train_history["lambda_p"]),
            "train_lambda_m": np.array(train_history["lambda_m"]),
            "train_lambda_t": np.array(train_history["lambda_t"]),
            "train_lambda_b": np.array(train_history["lambda_b"]),
            # 验证损失各分量
            "val_total": np.array(val_history["total"]),
            "val_pred": np.array(val_history["pred"]),
            "val_physics": np.array(val_history["physics"]),
            "val_mode": np.array(val_history["mode"]),
            # 验证物理损失子分量
            "val_cw_input": np.array(val_history["cw_input"]),
            "val_cw_pred": np.array(val_history["cw_pred"]),
            "val_dv_change": np.array(val_history["dv_change"]),
            "val_dv_align": np.array(val_history["dv_align"]),
            # 验证任务指标
            "val_terminal_dist_mean": np.array(val_history["terminal_dist_mean"]),
            "val_terminal_dist_std": np.array(val_history["terminal_dist_std"]),
            "val_terminal_dist_median": np.array(val_history["terminal_dist_median"]),
            "val_terminal_dist_min": np.array(val_history["terminal_dist_min"]),
            "val_terminal_dist_max": np.array(val_history["terminal_dist_max"]),
            "val_success_rate_1km": np.array(val_history["success_rate_1km"]),
            "val_success_rate_100m": np.array(val_history["success_rate_100m"]),
            "val_dv_over_limit_rate": np.array(val_history["dv_over_limit_rate"]),
            "val_dv_mag_mean": np.array(val_history["dv_mag_mean"]),
            "val_dv_mag_max": np.array(val_history["dv_mag_max"]),
        }
        sio.savemat(mat_path, mat_data)
        print(f"训练历史已保存至: {mat_path}")
    except Exception as e:
        print(f"保存训练历史 .mat 失败: {e}")

    print(f"\n训练完成，最佳模型: epoch {best_epoch}, "
          f"terminal_dist_mean={best_terminal_dist:.4f} km, "
          f"val_loss={best_val_loss:.6f}")
    print(f"模型已保存至: {MODEL_SAVE_PATH}")
    return model, scaler, test_loader


if __name__ == "__main__":
    train()
