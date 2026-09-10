"""
项目配置文件 —— 所有路径和超参数集中管理。
"""

import torch
import os

# ==================== 路径配置 ====================
DATA_DIR = os.path.join(os.path.dirname(__file__), "Dataset_Summary")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
CHECKPOINT_SAVE_PATH = os.path.join(OUTPUT_DIR, "latest_checkpoint.pth")
SCALER_SAVE_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
LOG_PATH = os.path.join(OUTPUT_DIR, "train_log.txt")

# ==================== 数据配置 ====================
MAX_N = 4                       # 最大目标数，用于 padding
MAX_DIM = MAX_N * 6             # 最大特征维度 = 24
INPUT_STEPS = 10                # 输入时间步数
OUTPUT_STEPS = 10               # 输出时间步数
TRAIN_RATIO = 0.70              # 训练集比例
VAL_RATIO = 0.15                # 验证集比例
TEST_RATIO = 0.15               # 测试集比例
RANDOM_SEED = 42                # 随机种子

# ==================== 模型配置 ====================
D_MODEL = 256                   # LSTM 隐层维度
NUM_LSTM_LAYERS = 3             # LSTM 层数
NHEAD = 8                       # 多头注意力头数（仅 Transformer 使用）
NUM_ENCODER_LAYERS = 4          # Encoder 层数（仅 Transformer 使用）
DIM_FEEDFORWARD = 512           # 前馈网络维度（仅 Transformer 使用）
DROPOUT = 0.15                  # Dropout 概率

# ==================== 物理信息配置 ====================
PHYSICS_ENABLED = True          # 是否启用物理信息条件 LSTM（False 则使用原版）
USE_TRANSFORMER = False         # v6: 退回 v3 PI-LSTM 架构（与 best_model.pth 权重兼容）
CONDITION_EMBED_DIM = 8         # 条件嵌入维度（模式嵌入 + 机动特征）
# v2 策略：物理信息作为架构创新保留，但物理损失基本关闭（数据违反 CW 单步递推）
PHYSICS_LOSS_WEIGHT = 0.0          # λ₁: CW 残差初始权重（关闭）
PHYSICS_LOSS_WEIGHT_FINAL = 1e-7   # λ₁: CW 残差最终权重（基本为 0，只作为数值稳定项）
PHYSICS_WARMUP_EPOCHS = 20
# Δv alignment 保留（PI-LSTM 核心约束），但权重较小
MODE_LOSS_WEIGHT = 0.001        # λ₂: Δv alignment 初始权重
MODE_LOSS_WEIGHT_FINAL = 0.05   # λ₂: Δv alignment 最终权重
DELTAV_LIMIT = 3.0              # Δv 幅值上限 (m/s)
CW_N = 0.001134                 # 轨道平均角速度 (rad/s)
CW_DT_H = 1.0                   # CW 递推短步长 (s)
CW_DT_T = 60.0                  # CW 递推长步长 (s)
PRED_WARMUP_EPOCHS = 3          # 预测损失预热轮数（缩短，让物理/末距损失早介入）

# ==================== Δv 边界软约束 ====================
DELTAV_BOUND_WEIGHT = 0.0005       # λ_b: Δv 边界软约束初始权重
DELTAV_BOUND_WEIGHT_FINAL = 0.005  # λ_b: Δv 边界软约束最终权重
DELTAV_BOUND_WARMUP_EPOCHS = 15

# ==================== 末端距离损失配置 ====================
TERMINAL_LOSS_WEIGHT = 0.01        # λ_t: 末端距离损失初始权重
TERMINAL_LOSS_WEIGHT_FINAL = 2.0   # λ_t: 末端距离损失最终权重（直接对齐用户目标，物理空间 km）
TERMINAL_WARMUP_EPOCHS = 15        # 末端距离损失权重 warmup 轮数
TERMINAL_PHYSICAL = True           # 是否在原始物理量纲空间计算末端距离损失
# ── 改进 C（2026-09-10）：末端距离归一化 ──
# 原方案 l_terminal 在物理空间直接求 3D 距离均值（量纲 km），数值 ~4.0 km，
# 与 l_pred（标准化空间 MSE ~0.008）量级差距 ~500×，导致末端距离以 1000× 优势主导梯度，
# 训练 loss 长时间停在 ~12，pred loss 几乎无梯度贡献。
# 改进：3D 距离除以参考距离 REF_DIST，让 l_terminal 量级降到 ~1.0（无量纲），
# 与 l_pred 量级差 ~100×，λ_t=2.0 仍主导（200× 优势）但不再压制 pred loss。
# 参考值取训练集/验证集 val td 均值（约 4.3 km），向上取整 4.5 km。
TERMINAL_REF_DIST = 4.5            # km 参考距离（用于归一化末端距离损失）

# ==================== 训练配置 ====================
BATCH_SIZE = 384               # 批大小（v4 增大到 384，梯度更稳定）
LEARNING_RATE = 1e-3            # 峰值学习率
MIN_LR = 1e-6                   # 最小学习率
WEIGHT_DECAY = 1e-5             # 权重衰减
EPOCHS = 80                     # 最大训练轮数（v4 延长）
EARLY_STOP_PATIENCE = 20        # 早停耐心值
WARMUP_EPOCHS = 3               # 学习率 warmup 轮数
RESUME_TRAINING = True          # v4 从 v3 best 继续训练（架构相同，仅 batch 增大）
# ── 续训时强制低 LR（避免 cosine scheduler 跨 epoch 跳变）──
# 设为 1e-6（≈ MIN_LR）相当于纯微调；设为 None 则保留 scheduler 自动行为
RESUME_FIXED_LR = 1e-6          # 2026-09-10 新增：续训时跳过 scheduler 恢复，强制低 LR

# ==================== 特征索引（用于分项评估） ====================
def _make_pos_vel_indices():
    """生成 24 维特征中位置和速度的索引列表。"""
    pos_idx, vel_idx = [], []
    for i in range(MAX_N):
        base = i * 6
        pos_idx.extend([base, base + 1, base + 2])
        vel_idx.extend([base + 3, base + 4, base + 5])
    return pos_idx, vel_idx

POS_INDICES, VEL_INDICES = _make_pos_vel_indices()

# ==================== 设备配置 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 自动创建输出目录 ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)
