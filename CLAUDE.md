# CLAUDE.md

> GitHub: <https://github.com/NPULXY/TrajectoryPrediction>

## 项目概述

航天器相对运动轨迹预测项目。支持两种模型架构：
- **标准 LSTM** (v4): Encoder-Decoder + 残差连接
- **物理信息条件 LSTM** (v5, 当前默认): 引入 CW 方程物理约束 + 机动条件门控

根据前 10 个时间步的相对运动状态（位置+速度）预测未来 10 步。

## 运行方式

```bash
python train.py      # 训练模型
python evaluate.py   # 评估并生成可视化图表
python predict.py    # 对新样本推理
python predict.py --input path/to/input.csv --output path/to/output.csv
python predict.py --visualize --top-k 10  # 推理 + 可视化最佳预测样本
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
├── Dataset_new2/              # 数据集 (23,787 样本)
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
| L_physics | CW 自洽损失 + Δv 边界惩罚 | λ₁: 0→0.5 (warmup 20 ep) |
| L_mode | Δv 一致性损失 (模型估计 vs CW 逆推) | λ₂: 0→0.05 (warmup 20 ep) |

物理损失具体包含:
- **CW 自洽损失**: MSE(x_{t+1} - Φ_h·x_t, B_eff·Δv_t)，约束状态变化与机动贡献一致
- **Δv 边界惩罚**: 超出 ±3 m/s 的部分施加二次惩罚
- **平滑性损失** (可选): 位置 jerk 最小化
- **边界约束** (可选): 相对距离 ≤ 200 km

## 训练策略

- AdamW + Cosine warmup (5 epochs) + Cosine annealing
- Peak LR: 1e-3, weight decay: 1e-5
- Batch size: 128
- 早停 patience: 50, max epochs: 300
- 梯度裁剪 max_norm=1.0
- 预测预热: 前 5 epoch 仅使用 L_pred

## 向后兼容

通过 `config.py` 中 `PHYSICS_ENABLED` 开关控制：
- `True` (默认): 使用物理信息条件 LSTM
- `False`: 回退到标准 TrajectoryLSTM
- 条件嵌入维度设为 0 也可退化（通过 CONDITION_EMBED_DIM=0）

## 数据背景

- 物理场景：近地圆轨道 (480 km)，追踪航天器相对非机动目标逼近
- 坐标系：LVLH（目标在原点）
- 轨道参数: n=0.001134 rad/s, μ=398600 km³/s², r=6851 km
- 23,787 样本，N=2/3/4 分布为 46%/31%/23%
- 输入步长 h=1s (10步观测窗口)，CW 外推步长 T=60s
- 所有样本视为机动段，单次脉冲 Δv 幅值 ≤ 3 m/s
- X_next 包含真实仿真步 (1s) 和可能的 CW 外推步 (60s)
- 详见 `Dataset_new2/README.md`

## 迭代记录

- v1: Transformer Encoder-Decoder → loss 不下降 (0.423)
- v2: Conv1D + Transformer + Delta 残差 → minor improvement (0.236)
- v3: Conv1D + Transformer + Cross-attn → 无法学习 (0.407)
- v4: LSTM Encoder-Decoder + Delta 残差 + Teacher forcing → 效果良好 (0.017)
- **v5: 物理信息条件 LSTM + CW 残差约束 + Δv 估计 → 增强物理一致性**

结论：短时序（10步）预测任务中，LSTM 比 Transformer 更有效；
引入物理约束可进一步提升预测的物理合理性。
