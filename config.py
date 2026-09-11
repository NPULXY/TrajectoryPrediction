"""
项目配置文件 —— 所有路径和超参数集中管理。

支持通过环境变量做轻量覆盖（用于并行训练与消融实验，默认值下行为与原来完全一致）：
  TP_RUN_TAG  产物文件名后缀，如 TP_RUN_TAG=seed1 → output/best_model_seed1.pth
  TP_SEED     模型初始化随机种子（0 = 不固定，使用系统熵）
  TP_HIDDEN   PI-LSTM 隐层维度覆盖（默认 384）
  TP_LAYERS   PI-LSTM 层数覆盖（默认 4）
"""

import torch
import os

# ==================== 运行标识与覆盖（环境变量）====================
RUN_TAG = os.environ.get("TP_RUN_TAG", "").strip()
_SUFFIX = f"_{RUN_TAG}" if RUN_TAG else ""

# 随机种子：0 表示不固定（每次训练随机初始化，用于集成所需的模型多样性）
SEED = int(os.environ.get("TP_SEED", "0"))

# ==================== 路径配置 ====================
DATA_DIR = os.path.join(os.path.dirname(__file__), "Dataset_Summary")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, f"best_model{_SUFFIX}.pth")
CHECKPOINT_SAVE_PATH = os.path.join(OUTPUT_DIR, f"latest_checkpoint{_SUFFIX}.pth")
SCALER_SAVE_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
LOG_PATH = os.path.join(OUTPUT_DIR, f"train_log{_SUFFIX}.txt")

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
D_MODEL = 256                   # 标准 LSTM 隐层维度（PI-LSTM 用下方 PI_HIDDEN_SIZE）
NUM_LSTM_LAYERS = 3             # 标准 LSTM 层数（PI-LSTM 用下方 PI_NUM_LAYERS）
# ── PI-LSTM 容量（2026-09-11 从 pinn_lstm.py 硬编码改为可配置）──
# 注意：当前最优权重是在 384/4 下训练的；改动这两个值会导致旧检查点键形状不匹配。
PI_HIDDEN_SIZE = int(os.environ.get("TP_HIDDEN", "384"))   # v3 PI-LSTM 隐层维度
PI_NUM_LAYERS = int(os.environ.get("TP_LAYERS", "4"))      # v3 PI-LSTM 层数
NHEAD = 8                       # 多头注意力头数（仅 Transformer 使用）
NUM_ENCODER_LAYERS = 4          # Encoder 层数（仅 Transformer 使用）
DIM_FEEDFORWARD = 512           # 前馈网络维度（仅 Transformer 使用）
DROPOUT = 0.15                  # Dropout 概率（0.25 曾使效果变差，2026-09-11 回退）

# ==================== 物理信息配置 ====================
PHYSICS_ENABLED = True          # 是否启用物理信息条件 LSTM（False 则使用原版）
USE_TRANSFORMER = False         # v6: 退回 v3 PI-LSTM 架构（与 best_model.pth 权重兼容）
CONDITION_EMBED_DIM = 8         # 条件嵌入维度（模式嵌入 + 机动特征）
# v3 策略：步长修正后（CW_DT_H=60s）CW 残差量级下降约 2 个数量级，重新启用物理约束
PHYSICS_LOSS_WEIGHT = 0.0          # λ₁: CW 残差初始权重（warmup 从 0 起）
PHYSICS_LOSS_WEIGHT_FINAL = 0.05   # λ₁: CW 残差最终权重（2026-09-10 修正步长后由 1e-7 重新启用）
PHYSICS_WARMUP_EPOCHS = 20
# Δv alignment 保留（PI-LSTM 核心约束），但权重较小
MODE_LOSS_WEIGHT = 0.001        # λ₂: Δv alignment 初始权重
MODE_LOSS_WEIGHT_FINAL = 0.05   # λ₂: Δv alignment 最终权重
DELTAV_LIMIT = 3.0              # Δv 幅值上限 (m/s)
CW_N = 0.001134                 # 轨道平均角速度 (rad/s)
# ── 时间步长（2026-09-10 修正）────────────────────────────────
# 实测判定（工具：_tools/verify_timestep.py）：数据集相邻状态的真实时间间隔是 60 s。
# 依据：CW 状态转移矩阵残差在 dt=60 s 时最小 —— X_now 3.4972→0.2114（小 16.6 倍），
#      X_next 4.3112→0.3267（中位数 0.0431）；dt=5/10/30/120/540/600 均明显更大。
# 旁证：序列内相邻步位移 3.5–4.4 km，按 1 s 对应 3.5 km/s 相对速度（不合理），
#      按 60 s 为 0.059 km/s（合理）。
# 修正前 CW_DT_H=1.0，导致 Δv 估计、CW 残差、Δv 边界三项物理计算全部错误。
CW_DT_H = 60.0                  # ★ CW 递推步长 = 数据真实采样间隔 (s)
CW_DT_T = 60.0                  # CW 外推步长 (s)（与数据步长一致）
CW_DT_RL = 1.0                  # RL 仿真步长 (s)，仅供 Δv 口径换算参考
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

# ==================== 全步位置损失（2026-09-11 新增，方向3：损失口径对齐）====================
# 动机：评估指标是**物理空间位置 RMSE (km)**，而 l_pred 是**标准化空间**的 Huber 损失。
# 标准化按各维 std 缩放（位置 std≈50 km，速度 std≈0.05 km/s），使速度误差在损失中被
# 显著放大，与评估口径不一致。本项直接监督全 10 步的物理空间 3D 位置误差。
# 默认 0.0（不改变原有行为）；通过 TP_POS_W 环境变量启用，便于 A/B 对照。
POS_LOSS_WEIGHT = 0.0                                        # λ_pos: 初始权重
POS_LOSS_WEIGHT_FINAL = float(os.environ.get("TP_POS_W", "0.0"))  # λ_pos: 最终权重
POS_LOSS_WARMUP_EPOCHS = 15                                  # warmup 轮数
POS_LOSS_REF_DIST = 4.5                                      # km 参考距离（与末端损失一致）

# ==================== 集成推理（2026-09-11 新增）====================
# 用法：把要参与集成的权重路径填入 ENSEMBLE_MODELS（相对项目根目录）；
#      留空列表 = 单模型模式（使用 MODEL_SAVE_PATH）。
# 集成方式：各成员在**标准化空间**等权平均后反归一化；成员容量可不同（自动推断架构）。
#
# 实测（测试集 32,647 样本，验证集选组合 → 测试集报告）：
#   推荐组合 = 512+位置损失 与 384+位置损失 的 3 成员集成
#     位置 RMSE 1.3707 → 1.2657 km（+7.66%）
#     位置 MAE  0.9423 → 0.8434 km（+10.50%）
#     末端距离  4.2780 → 4.1659 km（+2.62%）
#   启用：TP_ENSEMBLE="output/best_model_p1.pth,output/best_model_f1.pth,output/best_model_f2.pth"
ENSEMBLE_MODELS = [p for p in (
    os.environ.get("TP_ENSEMBLE", "").split(",")
) if p.strip()] or []

# ==================== 训练配置 ====================
BATCH_SIZE = 384               # 批大小（v4 增大到 384，梯度更稳定）
LEARNING_RATE = 1e-3            # 峰值学习率
MIN_LR = 1e-6                   # 最小学习率
WEIGHT_DECAY = 1e-5             # 权重衰减（1e-4 曾使效果变差，2026-09-11 回退）
EPOCHS = 80                     # 最大训练轮数
EARLY_STOP_PATIENCE = 20        # 早停耐心值
WARMUP_EPOCHS = 3               # 学习率 warmup 轮数

# ==================== 数据增强（2026-09-11 关闭）====================
# 关闭原因：经 _tools/verify_metrics_consistency.py 核验，模型**几乎不存在过拟合**
# （同度量下 val/train = 1.088×），正则化只会导致欠拟合。实测开启后位置 RMSE 由
# 1.3707 km 恶化到 1.4485 km（+5.7%）。保留开关以便未来在真出现过拟合时启用。
AUGMENT_ENABLED = False         # 训练输入加高斯噪声
AUGMENT_NOISE_STD = 0.005       # 噪声标准差（标准化空间）
AUGMENT_DECAY_EPOCHS = 60       # 噪声标准差线性衰减到 0 的 epoch 数
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
