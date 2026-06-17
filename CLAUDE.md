# CLAUDE.md

> GitHub: <https://github.com/NPULXY/TrajectoryPrediction>

## 项目概述

航天器相对运动轨迹预测项目。支持两种模型架构：
- **标准 LSTM** (v4): Encoder-Decoder + 残差连接
- **物理信息条件 LSTM** (v5, 当前默认): 引入 CW 方程物理约束 + 机动条件门控

根据前 10 个时间步的相对运动状态（位置+速度）预测未来 10 步。数据集覆盖**交会、阻扰、探测、潜伏、混合**五种场景，共 217,642 个样本。

## 技术路线

### 1. 问题建模

本项目的核心任务是**航天器相对运动轨迹预测**：给定追踪航天器相对于非机动目标的过去 10 步运动状态（位置 + 速度），预测未来 10 步的状态序列。这是一个**序列到序列（Seq2Seq）的时序预测问题**。

**物理场景**：近地圆轨道（高度 480 km），目标位于 LVLH 坐标系原点。追踪航天器通过脉冲机动逼近/拦截目标，Δv 幅值 ≤ 3 m/s。数据集覆盖交会（远距窄扇面）、阻扰（近距广域）、探测（中距全角）、潜伏（远距全角）及混合共五种场景，所有样本均为机动段，不存在纯自由漂移段。

**输入输出**：
- 输入 `X_now`：(B, 10, 24) — 10 个观测步，每步含 N 个目标 × 6 维状态（位置 xyz + 速度 vx vy vz），N = 2/3/4，不足 4 的补零
- 输出 `X_next`：(B, 10, 24) — 紧接的 10 个未来步

**核心挑战**：仅凭 10 步（9 秒）的观测窗口，模型需理解当前机动状态并外推未来 10 步。观测窗口极短，要求模型具备强归纳偏置。

### 2. CW 方程 —— 物理先验

Clohessy-Wiltshire（CW）方程是近地圆轨道相对运动的线性化动力学模型，也是本项目的核心物理先验：

$$\ddot{x} - 2n\dot{y} - 3n^2 x = 0$$
$$\ddot{y} + 2n\dot{x} = 0$$
$$\ddot{z} + n^2 z = 0$$

其中 n ≈ 0.001134 rad/s 为轨道平均角速度。在无机动条件下，状态演化可由状态转移矩阵 Φ(Δt) 精确描述：

$$x(t+\Delta t) = \Phi(\Delta t) \cdot x(t)$$

当存在瞬时脉冲机动 Δv 时，状态变化可分解为自由演化 + 机动贡献：

$$x_{t+1} = \Phi_h \cdot x_t + B_{eff} \cdot \Delta v_t$$

其中 B_eff = Φ 的速度列（第 4-6 列），h = 1s 为仿真步长。该关系是可微的，因此可以反向求解 Δv：

$$\Delta v_t = B_{eff}^+ \cdot (x_{t+1} - \Phi_h \cdot x_t)$$

**CW 方程的局限性**：它是非线性动力学的线性化近似，忽略 J2 摄动、大气阻力等，仅在两航天器距离远小于轨道半径时准确。但对近距逼近场景，它是有效的物理先验。

### 3. LSTM Encoder-Decoder 架构

本项目经过 v1-v4 迭代，最终确定 LSTM 优于 Transformer 作为序列建模主干。原因：

- **短序列**（仅 10 步）：Transformer 的自注意力在极短序列上无法发挥长程建模优势，反而因过多的可学习参数导致过拟合
- **强时序因果性**：轨迹是连续动力学演化的结果，LSTM 的循环结构天然契合这种马尔可夫性
- **数据规模**（217,642 样本，原为 23,787）：LSTM 参数量更少，在大规模数据上训练效率更高且不易过拟合

#### 标准 LSTM (v4)

```
输入 (B, 10, 24)
  → Encoder LSTM (3层, hidden=256) → 隐状态 (h_n, c_n)
  → 上下文向量 = MLP(h_n[-1])
  → Decoder LSTM (3层) 自回归生成 10 步 delta
  → 输出 = 持久预测 + delta
```

关键设计：
- **残差连接（Delta 预测）**：模型不直接预测未来状态，而是预测相对于"持久预测"（将最后观测状态复制 10 步）的修正量 delta。这保证了模型输出不会比朴素基线更差，大大降低了学习难度。
- **Teacher Forcing**：训练时以一定概率用真实 delta 替代模型预测的 delta 作为下一步 Decoder 输入，线性衰减（75 epoch 内从 1.0 → 0），逐步从"跟练"过渡到"自主生成"。
- **LSTM 正交初始化** + forget gate bias = 1，缓解梯度消失。

#### 物理信息条件 LSTM (v5, 当前默认)

在 v4 基础上引入物理约束，使模型理解轨道动力学而非纯黑箱拟合：

```
输入 (B, 10, 24) + mask
  → [物理引导] Δv 估计模块：CW 逆推，可微地估计每步 Δv
  → [条件门控] 条件向量 c_t = mode_embed + MLP(Δv_t)
  → [扩展输入] concat(状态, 条件) per agent → 56 维
  → [Encoder] LSTM (3层, hidden=256)
  → [Decoder] LSTM (3层) 自回归 + teacher forcing
  → 输出 = 持久预测 + delta
  → [辅助输出] 全序列 Δv 估计，用于物理损失
```

新增组件的原理：

**(a) Δv 可微估计模块**：利用 CW 方程的可微性质，从状态序列中通过最小二乘反推 Δv。因为 CW 矩阵和伪逆运算均可微，该模块天然支持梯度传播，无需外部标签。

**(b) 条件门控机制**：将机动信息显式注入 LSTM。条件向量 c_t 由两部分相加构成：
- **模式嵌入**（mode_embed）：2 行可学习嵌入（非机动 / 机动），本项目中所有样本均为机动模式
- **Δv 机动特征**（MLP(Δv)）：将估计的 Δv 映射到嵌入空间

条件向量与每个目标的状态拼接后送入 LSTM，使门控单元能根据当前机动强度自适应调整信息流动。

### 4. 损失函数设计

v5 的损失函数包含三个分量：

$$\mathcal{L} = \mathcal{L}_{pred} + \lambda_1 \mathcal{L}_{physics} + \lambda_2 \mathcal{L}_{mode}$$

| 分量 | 公式 | 物理含义 | 权重策略 |
|------|------|---------|---------|
| L_pred | MSE(pred, target) | 预测精度 | 固定 1.0 |
| L_physics | MSE(x_{t+1} - Φ_h·x_t, B_eff·Δv_t) + Δv 边界惩罚 | CW 自洽性 + 机动合理性 | 0 → 0.05，warmup 30 ep |
| L_mode | MSE(Δv_model, Δv_CW) | 模型估计与 CW 逆推的 Δv 一致性 | 0 → 0.01，warmup 30 ep |

**物理损失的动机**：纯 MSE 损失训练的模型可能输出物理上不合理的轨迹（如违反 CW 动力学、Δv 超限）。物理损失通过约束相邻步状态变化与机动贡献的一致性，以及惩罚超出 ±3 m/s 边界的 Δv，引导模型学习符合轨道力学规律的预测。

**权重 warmup 策略**：训练初期仅优化 L_pred（前 10 epoch），让模型先学会基本预测。第 10-30 epoch 逐步增大物理损失权重至最终值，避免训练早期物理约束过强导致模型收敛到平凡解（如输出恒为零）。

### 5. 训练策略

- **优化器**：AdamW（解耦权重衰减），peak LR = 1e-3，weight decay = 1e-5
- **学习率调度**：Cosine warmup（5 epoch）→ Cosine annealing（至 1e-6）
- **批大小**：128
- **正则化**：Dropout 0.15，梯度裁剪 max_norm = 1.0
- **早停**：验证损失 50 epoch 不降即停止，最大 300 epoch
- **Teacher forcing 衰减**：线性从 1.0 → 0（75 epoch），平衡训练稳定性与推理一致性

### 6. 技术选型总结

| 设计选择 | 选用方案 | 淘汰方案 | 原因 |
|---------|---------|---------|------|
| 序列主干 | LSTM | Transformer, Conv1D | 短序列上 LSTM 更高效，参数量更少 |
| 预测策略 | 残差 delta 预测 | 直接预测 | 降低学习难度，保证不差于基线 |
| 物理先验 | CW 方程 + Δv 估计 | 纯数据驱动 | 提升物理合理性，减少对数据的依赖 |
| 条件注入 | 模式嵌入 + Δv MLP | 无 | 使模型感知机动状态，自适应调整行为 |
| 训练稳定性 | Teacher forcing 线性衰减 | 固定比例 | 平滑过渡，避免 train-inference mismatch |
| 物理损失引入 | Warmup 渐进权重 | 固定权重 | 避免早期物理约束过强干扰基本拟合 |

## 运行方式

```bash
python train.py                     # 训练模型（支持断点恢复）
python evaluate.py                  # 评估并生成可视化图表
python predict.py                   # 推理 + 可视化最佳预测样本（--visualize 默认开启）
python predict.py --no-visualize    # 仅推理，不生成可视化
python predict.py --input path/to/input.csv --output path/to/output.csv
python predict.py --top-k 30        # 可视化最佳样本数量（默认 30，N=2/3/4 各 10 个）
```

## 项目结构

```
├── config.py                  # 所有超参数和路径
├── train.py                   # 训练入口（支持双模式）
├── evaluate.py                # 评估 + 可视化（含物理一致性指标）
├── predict.py                 # 推理 + 最佳预测可视化（--visualize）
├── models/
│   ├── model.py               # 标准 TrajectoryLSTM + create_model 工厂
│   ├── pinn_lstm.py           # 物理信息条件 LSTM (v5, 新增)
│   └── physics_loss.py        # CW 残差、Δv 一致性等物理损失 (新增)
├── utils/data_loader.py       # 数据解析、padding、scaler、DataLoader
├── Dataset_new2/              # 交会场景子数据集 (23,787 样本)
├── Dataset_Summary/           # 汇总数据集 (217,642 样本, 含 X_now/X_next/Y)
└── output/                    # 模型权重、scaler、图表、最佳预测可视化 (best_predictions/)
```

## 模型架构 (v5: 物理信息条件 LSTM)

**PhysicsInformedTrajectoryLSTM** (条件 LSTM Encoder-Decoder + CW 物理约束):

```
输入 (B, 10, 24) + mask (B, 24)
  → Δv 估计模块 (CW 逆推, 可微)
  → 全局 Δv = mean(per-agent Δv)
  → 条件向量 c_t = mode_embed(机动=1) + MLP(Δv_t), (d_embed=8)
  → 扩展输入 = concat(state_i, c_t) per agent, (max_N × (6+8) = 56D)
  → Encoder LSTM (3层, hidden=256)
  → 上下文向量 + 初始 delta
  → Decoder LSTM (3层) 自回归生成 10 步 delta
  → 最终输出 = 持久预测 + delta, (B, 10, 24)
  → 同时输出 Δv 估计 (B, 19, max_N×3) 用于物理损失
```

关键设计：
- **条件门控**: 机动模式嵌入 (2×d_embed) + Δv 机动特征 MLP → 扩展 LSTM 输入
- **Δv 估计模块**: CW 方程逆推，天然可微，参与梯度传播
- **残差连接**: 输出 = 持久预测 + delta，保证不低于朴素基线
- **Teacher forcing**: 线性衰减 (75 epoch 内从 1.0 → 0)
- **LSTM 正交初始化**, forget gate bias=1

## 损失函数 (v5)

$$\\mathcal{L} = \\mathcal{L}_{pred} + \\lambda_1 \\mathcal{L}_{physics} + \\lambda_2 \\mathcal{L}_{mode}$$

| 损失分量 | 说明 | 权重 |
|---------|------|------|
| L_pred | 预测与真实状态的 MSE (masked) | 1.0 |
| L_physics | CW 自洽损失 + Δv 边界惩罚 | λ₁: 0.005→0.05 (warmup 30 ep) |
| L_mode | Δv 一致性损失 (模型估计 vs CW 逆推) | λ₂: 0.001→0.01 (warmup 30 ep) |

物理损失具体包含:
- **CW 自洽损失**: MSE(x_{t+1} - Φ_h·x_t, B_eff·Δv_t)，约束状态变化与机动贡献一致
- **Δv 边界惩罚**: 超出 ±3 m/s 的部分施加二次惩罚
- **平滑性损失** (可选): 位置 jerk 最小化
- **边界约束** (可选): 相对距离 ≤ 200 km

## 训练策略

- AdamW + Cosine warmup (5 epochs) + Cosine annealing
- Peak LR: 1e-3, weight decay: 1e-5
- Batch size: 128
- 早停 patience: 50, max epochs: 100
- 梯度裁剪 max_norm=1.0
- 预测预热: 前 10 epoch 仅使用 L_pred
- 断点恢复: `RESUME_TRAINING=True` 自动加载最近检查点，恢复优化器与调度器状态继续训练

## 向后兼容

通过 `config.py` 中 `PHYSICS_ENABLED` 开关控制：
- `True` (默认): 使用物理信息条件 LSTM
- `False`: 回退到标准 TrajectoryLSTM
- 条件嵌入维度设为 0 也可退化（通过 CONDITION_EMBED_DIM=0）

## 数据背景

默认使用 `Dataset_Summary/`（汇总版）进行训练，同时保留 `Dataset_new2/`（交会场景子集）供对比实验。

### Dataset_Summary（汇总版）

- **规模**：217,642 样本，由五个子数据集合并并随机打乱
- **场景构成**：交会（23,787）、阻扰（39,100）、探测（21,166）、潜伏（24,766）、混合（108,823）
- **标签文件**：新增 `Y.csv`，每行包含 `[N, min_distance, phi]`（目标数、全局最小距离、最远/最近目标夹角）
- N=2/3/4 分布约为 46%/31%/23%
- 输入步长 h=1s（10步观测窗口），CW 外推步长 T=60s
- 所有样本视为机动段，单次脉冲 Δv 幅值 ≤ 3 m/s
- X_next 包含真实仿真步 (1s) 和可能的 CW 外推步 (60s)
- 详见 `Dataset_Summary/README.md`

### Dataset_new2（交会场景子集）

- **规模**：23,787 样本，纯交会逼近场景
- 详见 `Dataset_new2/README.md`

### 公共物理参数

- 坐标系：LVLH（目标在原点）
- 轨道参数: n=0.001134 rad/s, μ=398600 km³/s², r=6851 km

## 迭代记录

- v1: Transformer Encoder-Decoder → loss 不下降 (0.423)
- v2: Conv1D + Transformer + Delta 残差 → minor improvement (0.236)
- v3: Conv1D + Transformer + Cross-attn → 无法学习 (0.407)
- v4: LSTM Encoder-Decoder + Delta 残差 + Teacher forcing → 效果良好 (0.017)
- **v5: 物理信息条件 LSTM + CW 残差约束 + Δv 估计 → 增强物理一致性**
- **v6: 多场景汇总数据集 + 断点恢复 + 学术风格可视化 → 提升泛化与可分析性**
  - Dataset_Summary: 217K 样本覆盖 5 场景，引入 Y.csv 标签
  - RESUME_TRAINING: 检查点断点恢复，支持长周期训练
  - 可视化重构: 末端距离取代 MSE，按 N 分层采样，英文学术图表样式

结论：短时序（10步）预测任务中，LSTM 比 Transformer 更有效；
引入物理约束可进一步提升预测的物理合理性。
