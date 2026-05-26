# CLAUDE.md

## 项目概述

航天器相对运动轨迹预测项目。使用 LSTM Encoder-Decoder 模型，
根据前 10 个时间步的相对运动状态（位置+速度）预测未来 10 步。

## 运行方式

```bash
python train.py      # 训练模型
python evaluate.py   # 评估并生成可视化图表
python predict.py    # 对新样本推理
python predict.py --input path/to/input.csv --output path/to/output.csv
```

## 项目结构

```
├── config.py              # 所有超参数和路径（直接修改此文件调参）
├── train.py               # 训练入口
├── evaluate.py            # 评估 + 可视化
├── predict.py             # 推理
├── models/model.py        # TrajectoryLSTM 模型
├── utils/data_loader.py   # 数据解析、padding、scaler、DataLoader
├── Dataset_new2/           # 数据集 (23,787 样本)
└── output/                 # 模型权重、scaler、图表
```

## 模型架构

**TrajectoryLSTM** (Encoder-Decoder + 残差连接):

```
输入 (B, 10, 24)
  → Encoder LSTM (3层, hidden=256)
  → 上下文向量 + 初始 delta 预测
  → Decoder LSTM (3层) 自回归生成 10 步 delta
  → 最终输出 = 持久预测 + delta
  → 输出 (B, 10, 24)
```

关键设计：
- **残差连接**：模型预测对未来轨迹的"修正量"（delta），加到持久预测（重复最后一步）上。
  保证模型至少等于朴素基线，训练初期更快收敛。
- **Teacher forcing**：训练初期 TF ratio=1.0，在 75 个 epoch 内线性衰减至 0。
- **LSTM 正交初始化**，forget gate bias=1

## 训练策略

- AdamW + Cosine warmup (5 epochs) + Cosine annealing
- Peak LR: 1e-3, weight decay: 1e-5
- Batch size: 128
- 早停 patience: 50, max epochs: 300
- 梯度裁剪 max_norm=1.0

## 性能

测试集指标（原始量纲）：
- 整体 RMSE: 1.03
- 位置 RMSE: 1.46 km (数据范围 0-200 km)
- 速度 RMSE: 0.0037 km/s = 3.7 m/s (数据范围 0-0.1 km/s)

## 数据背景

- 物理场景：近地圆轨道 (480 km)，追踪航天器相对非机动目标逼近
- 坐标系：LVLH（目标在原点）
- 23,787 样本，N=2/3/4 分布为 46%/31%/23%
- X_next 包含真实仿真步 (1s 步长) 和可能的 CW 外推步 (60s 步长)
- 详见 `Dataset_new2/README.md`

## 迭代记录

- v1: Transformer Encoder-Decoder → loss 不下降 (0.423)
- v2: Conv1D + Transformer + Delta 残差 → minor improvement (0.236)
- v3: Conv1D + Transformer + Cross-attn → 无法学习 (0.407)
- **v4: LSTM Encoder-Decoder + Delta 残差 + Teacher forcing → 效果良好 (0.017)**

结论：短时序（10步）预测任务中，LSTM 比 Transformer 更有效。
