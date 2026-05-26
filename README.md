# 航天器相对运动轨迹预测

基于 Transformer 的 seq2seq 轨迹预测模型，根据前 10 个时间步的相对运动状态预测未来 10 个时间步的状态。

## 任务概述

- **输入**：追踪航天器相对于目标的近期 10 步相对状态（位置 x,y,z + 速度 vx,vy,vz），多目标场景 N∈{2,3,4}
- **输出**：紧接的 10 步未来相对状态
- **方法**：Transformer Encoder-Decoder 统一模型 + padding 处理变长目标数

## 项目结构

```
├── config.py              # 超参数和路径配置
├── train.py               # 训练脚本
├── evaluate.py            # 评估与可视化
├── predict.py             # 推理脚本
├── requirements.txt       # Python 依赖
├── models/
│   └── model.py           # Transformer 模型定义
├── utils/
│   └── data_loader.py     # 数据加载、标准化、DataLoader
├── Dataset_new2/           # 数据集（X_now.csv, X_next.csv）
└── output/                 # 输出（模型、scaler、图表）
```

## 环境依赖

- Python >= 3.9
- PyTorch >= 2.0
- NumPy, Pandas, scikit-learn, Matplotlib, tqdm

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 训练

```bash
python train.py
```

训练过程会：
1. 自动加载 `Dataset_new2/X_now.csv` 和 `X_next.csv`
2. 同步打乱并划分为 70%/15%/15% 训练/验证/测试集
3. 对训练集计算 z-score 标准化参数（逐特征维度）
4. 训练 Transformer 模型，监控验证损失
5. 早停（patience=20）并保存最佳模型至 `output/best_model.pth`
6. 保存 scaler 至 `output/scaler.pkl`

### 评估

```bash
python evaluate.py
```

在测试集上计算 MSE、RMSE、MAE（整体 + 分位置/速度），并生成：
- `output/sample_*.png`：随机样本的真实 vs 预测轨迹对比（3D + 分量图）
- `output/loss_curve.png`：训练/验证损失曲线

### 推理

```bash
# 对完整 X_now.csv 进行预测
python predict.py

# 对自定义文件推理
python predict.py --input path/to/input.csv --output path/to/output.csv
```

输出格式与输入兼容：嵌套列表字符串，保留 4 位小数。

## 模型架构

```
输入 (B, 10, 24)
  → Linear(input_dim=24 → d_model=128)
  → + 可学习位置编码 (10, d_model)
  → Transformer Encoder (4 层, 8 头)
  → 可学习输出查询 + Transformer Decoder (2 层, 8 头)
  → Output Head (Linear→GELU→Dropout→Linear)
  → 输出 (B, 10, 24)
```

- 使用 GELU 激活函数
- Xavier 初始化
- 梯度裁剪 (max_norm=1.0)
- 支持 Encoder-Only 模式（`config.py` 中 `USE_DECODER=False`）

## 配置说明

所有可调参数集中在 `config.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `D_MODEL` | 128 | Transformer 隐层维度 |
| `NHEAD` | 8 | 多头注意力头数 |
| `NUM_ENCODER_LAYERS` | 4 | Encoder 层数 |
| `NUM_DECODER_LAYERS` | 2 | Decoder 层数 |
| `DIM_FEEDFORWARD` | 512 | FFN 维度 |
| `DROPOUT` | 0.1 | Dropout |
| `BATCH_SIZE` | 64 | 批大小 |
| `LEARNING_RATE` | 1e-3 | 初始学习率 |
| `WEIGHT_DECAY` | 1e-5 | 权重衰减 |
| `EPOCHS` | 200 | 最大训练轮数 |
| `EARLY_STOP_PATIENCE` | 20 | 早停耐心值 |

## 数据说明

数据位于 `Dataset_new2/`，包含 23,787 个样本，其中：
- N=2：46.2%，N=3：30.8%，N=4：23.1%

输入和输出均通过 padding 至 24 维（N=4），并在损失计算时通过 mask 忽略 padding 部分。详细说明见 `Dataset_new2/README.md`。
