# 航天器相对运动轨迹预测

基于 **物理信息条件 LSTM (PINN-LSTM)** 的序列到序列轨迹预测模型，根据前 10 个时间步的相对运动状态预测未来 10 个时间步的状态。

> GitHub: <https://github.com/NPULXY/TrajectoryPrediction>

---

## 目录

- [任务概述](#任务概述)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
- [模型架构](#模型架构)
- [损失函数](#损失函数)
- [数据集](#数据集)
- [训练策略](#训练策略)
- [评估与可视化](#评估与可视化)
- [配置说明](#配置说明)
- [迭代记录](#迭代记录)

---

## 任务概述

本项目的核心任务是**航天器相对运动轨迹预测**：给定追踪航天器相对于目标的过去 10 步运动状态（位置 + 速度），预测未来 10 步的状态序列。这是一个 **序列到序列（Seq2Seq）的时序预测问题**。

**物理场景**：近地圆轨道（高度 480 km，轨道角速度 n ≈ 0.001134 rad/s），目标位于 LVLH 坐标系原点。追踪航天器通过脉冲机动逼近/拦截非机动目标，Δv 幅值 ≤ 3 m/s。所有样本均为机动段，不存在纯自由漂移段。

**核心挑战**：仅凭 10 步（9 秒）的观测窗口，模型需理解当前机动状态并外推未来 10 步，要求模型具备强归纳偏置。

**数据集覆盖**五种场景：
- 交会（远距窄扇面逼近）
- 阻扰（近距广域拦截）
- 探测（中距全角搜索）
- 潜伏（远距全角跟踪）
- 混合（上述四种场景的混合）

---

## 项目结构

```
├── config.py                  # 所有超参数和路径配置
├── train.py                   # 训练入口（支持断点恢复）
├── evaluate.py                # 评估 + 可视化（含物理一致性指标）
├── predict.py                 # 推理 + 最佳预测可视化
├── requirements.txt           # Python 依赖
├── models/
│   ├── model.py               # 标准 TrajectoryLSTM + create_model 工厂
│   ├── pinn_lstm.py           # 物理信息条件 LSTM (v5)
│   └── physics_loss.py        # CW 残差、Δv 一致性等物理损失
├── utils/
│   └── data_loader.py         # 数据解析、padding、scaler、DataLoader
├── Dataset_new2/              # 交会场景子数据集（23,787 样本）
├── Dataset_Summary/           # 汇总数据集（217,642 样本，含标签 Y.csv）
└── output/                    # 输出目录
    ├── best_model.pth         # 最佳模型权重
    ├── scaler.pkl             # 标准化参数
    ├── train_log.txt          # 训练日志
    ├── loss_curve.png         # 损失曲线
    ├── sample_*.png           # 随机样本评估图
    └── best_predictions/      # 最佳预测可视化图
```

---

## 环境依赖

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy, Pandas, scikit-learn, Matplotlib, tqdm

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 训练

```bash
python train.py
```

训练过程会：
1. 自动加载 `Dataset_Summary/X_now.csv`、`X_next.csv` 和 `Y.csv`
2. 同步打乱并划分为 70%/15%/15% 训练/验证/测试集
3. 对训练集逐特征维度计算 z-score 标准化参数
4. 根据 `PHYSICS_ENABLED` 选择物理信息条件 LSTM 或标准 LSTM
5. Cosine warmup + Cosine annealing 学习率调度
6. 早停（patience=50）并保存最佳模型至 `output/best_model.pth`
7. **支持断点恢复**：通过 `RESUME_TRAINING=True`（默认）自动加载最近检查点，恢复优化器与调度器状态继续训练
8. 保存 scaler 至 `output/scaler.pkl`

### 评估

```bash
python evaluate.py
```

在测试集上计算 MSE、RMSE、MAE（整体 + 分位置/速度），输出物理一致性指标，并生成：
- `output/sample_*.png`：随机样本的真实 vs 预测轨迹对比（3D + 分量图）
- `output/loss_curve.png`：训练/验证损失曲线

### 推理

```bash
# 对完整 X_now.csv 进行推理并可视化最佳样本（默认）
python predict.py

# 仅推理，不生成可视化
python predict.py --no-visualize

# 对自定义文件推理
python predict.py --input path/to/input.csv --output path/to/output.csv

# 指定可视化最佳样本数量（默认 30，N=2/3/4 各 10 个）
python predict.py --top-k 30
```

**可视化特点**：
- 评估指标：**末端距离**（预测与真实末步位置的 3D 距离），而非 MSE
- 分层采样：分别从 N=2/3/4 各组中选取最优样本，确保展示样本多样性
- 学术图表风格：Times New Roman 字体、STIX 数学符号、英文标签

---

## 模型架构

### 物理信息条件 LSTM (v5, 默认)

```
输入 (B, 10, 24) + mask (B, 24)
  → Δv 估计模块（CW 逆推，可微）
  → 全局 Δv = mean(per-agent Δv)
  → 条件向量 c_t = mode_embed(机动=1) + MLP(Δv_t)
  → 扩展输入 = concat(state_i, c_t) per agent → 56 维
  → Encoder LSTM (3 层, hidden=256)
  → 上下文向量 + 初始 delta
  → Decoder LSTM (3 层) 自回归生成 10 步 delta
  → 最终输出 = 持久预测 + delta, (B, 10, 24)
  → 同时输出 Δv 估计 (B, 19, max_N×3) 用于物理损失
```

### 标准 LSTM (v4, 通过 `PHYSICS_ENABLED=False` 回退)

```
输入 (B, 10, 24)
  → Encoder LSTM (3 层, hidden=256) → 隐状态 (h_n, c_n)
  → 上下文向量 = MLP(h_n[-1])
  → Decoder LSTM (3 层) 自回归生成 10 步 delta
  → 输出 = 持久预测 + delta, (B, 10, 24)
```

### 关键设计

| 设计选择 | 方案 | 说明 |
|---------|------|------|
| 序列主干 | LSTM | 短序列（10 步）上比 Transformer 更高效，参数量更少 |
| 预测策略 | 残差 delta 预测 | 输出 = 持久预测 + delta，保证不低于朴素基线 |
| 物理先验 | CW 方程 + Δv 可微估计 | 利用轨道动力学约束引导模型学习 |
| 条件门控 | 模式嵌入 + Δv MLP | 将机动信息显式注入 LSTM，自适应调整行为 |
| 训练稳定 | Teacher forcing 线性衰减 | 75 epoch 内从 1.0 → 0，平滑过渡到自主生成 |
| 初始化 | LSTM 正交初始化 + forget gate bias=1 | 缓解梯度消失 |

---

## 损失函数

$$\mathcal{L} = \mathcal{L}_{pred} + \lambda_1 \mathcal{L}_{physics} + \lambda_2 \mathcal{L}_{mode}$$

| 损失分量 | 公式 | 物理含义 | 权重 |
|---------|------|---------|------|
| ℒ_pred | MSE(pred, target) | 预测精度 | 1.0 |
| ℒ_physics | MSE(x_{t+1} − Φ_h·x_t, B_eff·Δv_t) + Δv 边界惩罚 | CW 自洽性 + 机动合理性 | λ₁: 0.005→0.05（warmup 30 ep） |
| ℒ_mode | MSE(Δv_model, Δv_CW) | 模型估计与 CW 逆推的 Δv 一致性 | λ₂: 0.001→0.01（warmup 30 ep） |

**物理损失动机**：纯 MSE 训练可能产生物理上不合理的轨迹（违反 CW 动力学、Δv 超限）。
物理损失通过约束相邻步状态变化与机动贡献的一致性，以及惩罚超出 ±3 m/s 边界
的 Δv，引导模型学习符合轨道力学规律的预测。

**Warmup 策略**：前 10 epoch 仅使用 ℒ_pred，第 10-30 epoch 逐步增大物理损失权重，
避免训练早期物理约束过强导致无法收敛。

---

## 数据集

默认使用 `Dataset_Summary/`（汇总版），同时保留 `Dataset_new2/`（交会场景子集）供对比实验。

### Dataset_Summary（汇总版）

| 属性 | 值 |
|------|-----|
| 总样本数 | **217,642** |
| X_now.csv | ~323 MB |
| X_next.csv | ~320 MB |
| Y.csv | ~4.4 MB（新增标签文件） |

**场景构成**：

| 场景 | 样本数 | 占比 |
|------|-------:|:---:|
| 交会（Capture） | 23,787 | 10.9% |
| 阻扰（Obstruction） | 39,100 | 18.0% |
| 探测（Detection） | 21,166 | 9.7% |
| 潜伏（Lurk） | 24,766 | 11.4% |
| 混合（Mix） | 108,823 | 50.0% |

**数据结构**：
- `X_now.csv` / `X_next.csv`：每行为嵌套列表字符串 `[[step1], ..., [step10]]`，
  每步包含 N×6 个浮点数（N 个目标 × 位置 xyz + 速度 vx vy vz）
- `Y.csv`：每行为 `[N, min_distance, phi]`（目标数、全局最小距离、最远/最近目标夹角）

**Y.csv 标签说明**：
- `N`：目标数量 {2, 3, 4}
- `min_distance`：所有目标在所有时刻中距 LVLH 原点的全局最小距离（km）
- `phi`：最小距离时刻，最近与最远目标位置向量间的夹角（rad）

### Dataset_new2（交会场景子集）

- 23,787 样本，纯交会逼近场景
- 初始距离 100~120 km，初始俯仰角 π/2 ± π/6

### 公共物理参数

| 参数 | 值 |
|------|-----|
| 轨道半径 | 6851 km（地球半径 6371 + 高度 480） |
| 地球引力常数 μ | 398600 km³/s² |
| 轨道角速度 n | 0.001134 rad/s |
| 仿真步长 h | 1 s |
| CW 外推步长 T | 60 s |
| Δv 上限 | 3 m/s |

> 详细数据说明见 `Dataset_Summary/README.md` 和 `Dataset_new2/README.md`。

---

## 训练策略

| 配置 | 值 |
|------|-----|
| 优化器 | AdamW |
| 峰值学习率 | 1×10⁻³ |
| 最小学习率 | 1×10⁻⁶ |
| 权重衰减 | 1×10⁻⁵ |
| 学习率调度 | Cosine warmup（5 epoch）+ Cosine annealing |
| 批大小 | 128 |
| 最大训练轮数 | 100 |
| 早停耐心值 | 50 epoch |
| Dropout | 0.15 |
| 梯度裁剪 | max_norm = 1.0 |
| 预测预热 | 前 10 epoch 仅使用 ℒ_pred |
| Teacher forcing 衰减 | 线性 1.0 → 0（75 epoch） |
| 断点恢复 | 默认启用（`RESUME_TRAINING=True`） |

**断点恢复机制**：训练中断后重新启动，自动加载 `output/best_model.pth` 中的
模型权重、优化器状态和调度器状态，从中断 epoch 继续训练而非从头开始。

---

## 评估与可视化

### 评估指标

- **MSE** / **RMSE** / **MAE**：位置和速度分量的预测误差
- **末端距离**：预测轨迹末步与真实末步的 3D 欧氏距离（km）
- **物理一致性**：CW 自洽性、Δv 幅值合理性

### 可视化输出

| 图表 | 说明 |
|------|------|
| `loss_curve.png` | 训练/验证损失曲线 |
| `sample_*.png` | 随机测试样本的 3D 轨迹 + 位置分量对比 |
| `best_predictions/top*.png` | 最佳预测样本（按末端距离排序，按 N 分层采样） |

可视化图表采用**英文学术论文风格**（Times New Roman 字体、STIX 数学符号、坐标轴含单位）。

---

## 配置说明

所有可调参数集中在 `config.py`。关键配置项：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `PHYSICS_ENABLED` | `True` | 启用物理信息条件 LSTM |
| `CONDITION_EMBED_DIM` | 8 | 条件嵌入维度 |
| `D_MODEL` | 256 | LSTM 隐层维度 |
| `NUM_LSTM_LAYERS` | 3 | LSTM 层数 |
| `DROPOUT` | 0.15 | Dropout 概率 |
| `BATCH_SIZE` | 128 | 批大小 |
| `LEARNING_RATE` | 1×10⁻³ | 峰值学习率 |
| `EPOCHS` | 100 | 最大训练轮数 |
| `EARLY_STOP_PATIENCE` | 50 | 早停耐心值 |
| `RESUME_TRAINING` | `True` | 断点恢复 |
| `PHYSICS_LOSS_WEIGHT` | 0.005 | 物理损失初始权重 |
| `PHYSICS_LOSS_WEIGHT_FINAL` | 0.05 | 物理损失最终权重 |
| `PHYSICS_WARMUP_EPOCHS` | 30 | 物理损失 warmup 轮数 |
| `MODE_LOSS_WEIGHT` | 0.001 | Δv 一致性损失初始权重 |
| `MODE_LOSS_WEIGHT_FINAL` | 0.01 | Δv 一致性损失最终权重 |
| `DELTAV_LIMIT` | 3.0 | Δv 幅值上限（m/s） |
| `CW_N` | 0.001134 | 轨道平均角速度（rad/s） |
| `CW_DT_H` | 1.0 | CW 短步长（s） |
| `CW_DT_T` | 60.0 | CW 长步长（s） |

---

## 向后兼容

通过 `config.py` 中 `PHYSICS_ENABLED` 开关控制：
- `True`（默认）：使用物理信息条件 LSTM
- `False`：回退到标准 TrajectoryLSTM

也可通过 `CONDITION_EMBED_DIM=0` 退化条件门控机制。

---

## 迭代记录

| 版本 | 架构 | 验证损失 | 说明 |
|------|------|:--------:|------|
| v1 | Transformer Encoder-Decoder | 0.423 | Loss 不下降 |
| v2 | Conv1D + Transformer + Delta 残差 | 0.236 | 小幅改进 |
| v3 | Conv1D + Transformer + Cross-attn | 0.407 | 无法学习 |
| v4 | LSTM Encoder-Decoder + Delta 残差 + Teacher forcing | **0.017** | 效果良好 |
| v5 | 物理信息条件 LSTM + CW 残差约束 + Δv 估计 | — | 增强物理一致性 |
| v6 | 多场景汇总数据集 + 断点恢复 + 学术风格可视化 | — | 提升泛化与可分析性 |

**结论**：短时序（10 步）预测任务中，LSTM 比 Transformer 更有效；
引入物理约束可进一步提升预测的物理合理性。

---

## 引用

如使用本项目，请引用：

```bibtex
@software{trajectory_prediction,
  author = {NPULXY},
  title = {航天器相对运动轨迹预测},
  year = {2026},
  url = {https://github.com/NPULXY/TrajectoryPrediction}
}
```

---

## 许可证

本项目仅供学术研究使用。
