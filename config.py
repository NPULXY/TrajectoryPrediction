"""
项目配置文件 —— 所有路径和超参数集中管理。
"""

import torch
import os

# ==================== 路径配置 ====================
DATA_DIR = os.path.join(os.path.dirname(__file__), "Dataset_Summary")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
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
CONDITION_EMBED_DIM = 8         # 条件嵌入维度（模式嵌入 + 机动特征）
PHYSICS_LOSS_WEIGHT = 0.005     # λ₁: 物理损失初始权重
PHYSICS_LOSS_WEIGHT_FINAL = 0.05# λ₁: 物理损失最终权重（warmup 后）
PHYSICS_WARMUP_EPOCHS = 30      # 物理损失权重 warmup 轮数
MODE_LOSS_WEIGHT = 0.001        # λ₂: 速度变化约束初始权重
MODE_LOSS_WEIGHT_FINAL = 0.01   # λ₂: 速度变化约束最终权重
DELTAV_LIMIT = 3.0              # Δv 幅值上限 (m/s)
CW_N = 0.001134                 # 轨道平均角速度 (rad/s)
CW_DT_H = 1.0                   # CW 递推短步长 (s)
CW_DT_T = 60.0                  # CW 递推长步长 (s)
PRED_WARMUP_EPOCHS = 10         # 预测损失预热轮数（仅用 L_pred）

# ==================== 训练配置 ====================
BATCH_SIZE = 128                # 批大小
LEARNING_RATE = 1e-3            # 峰值学习率
MIN_LR = 1e-6                   # 最小学习率
WEIGHT_DECAY = 1e-5             # 权重衰减
EPOCHS = 100                    # 最大训练轮数
EARLY_STOP_PATIENCE = 50        # 早停耐心值
WARMUP_EPOCHS = 5               # 学习率 warmup 轮数
RESUME_TRAINING = True          # 是否从最佳检查点恢复训练

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
