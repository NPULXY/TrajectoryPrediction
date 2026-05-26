"""
项目配置文件 —— 所有路径和超参数集中管理。
"""

import torch
import os

# ==================== 路径配置 ====================
DATA_DIR = os.path.join(os.path.dirname(__file__), "Dataset_new2")
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
D_MODEL = 128                   # Transformer 隐层维度
NHEAD = 8                       # 多头注意力头数
NUM_ENCODER_LAYERS = 4          # Encoder 层数
NUM_DECODER_LAYERS = 2          # Decoder 层数
DIM_FEEDFORWARD = 512           # 前馈网络维度
DROPOUT = 0.1                   # Dropout 概率
USE_DECODER = True              # 是否使用 Transformer Decoder（否则直接回归）

# ==================== 训练配置 ====================
BATCH_SIZE = 64                 # 批大小
LEARNING_RATE = 1e-3            # 初始学习率
WEIGHT_DECAY = 1e-5             # 权重衰减
EPOCHS = 200                    # 最大训练轮数
EARLY_STOP_PATIENCE = 20        # 早停耐心值
LR_REDUCE_FACTOR = 0.5          # 学习率衰减因子
LR_REDUCE_PATIENCE = 10         # 学习率衰减耐心值

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
