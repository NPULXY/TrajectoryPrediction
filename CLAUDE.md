# CLAUDE.md

## 项目概述

航天器相对运动轨迹预测项目。使用 Transformer seq2seq 模型，
根据前 10 个时间步的相对运动状态（位置+速度）预测未来 10 步。

## 运行方式

```bash
python train.py      # 训练模型
python evaluate.py   # 评估并生成可视化图表
python predict.py    # 对新样本推理
python predict.py --input path/to/input.csv --output path/to/output.csv
```

训练前会先从 `output/` 删除旧的 scaler/模型文件（如存在），确保不会错误复用。

## 项目结构

```
├── config.py              # 所有超参数和路径（直接修改此文件调参）
├── train.py               # 训练入口
├── evaluate.py            # 评估 + 可视化
├── predict.py             # 推理
├── models/model.py        # TrajectoryTransformer 模型
├── utils/data_loader.py   # 数据解析、padding、scaler、DataLoader
├── Dataset_new2/           # 数据集
└── output/                 # 模型权重、scaler、图表
```

## 关键设计决策

### 数据层面

- **变长目标数处理**：N∈{2,3,4} 的目标数导致特征维度分别为 12/18/24。
  统一 padding 至 24 维（N=4），同时生成 mask 标记有效维度。
  损失计算时仅对 mask=True 的维度求 MSE。
- **CSV 解析**：嵌套列表字符串含逗号，不能用 pandas CSV 解析器（会将一行误判为多列）。
  改为逐行读取 + `ast.literal_eval`。
- **标准化**：逐特征维度 z-score（24 个独立均值和标准差），
  统计量仅从训练集有效数据计算（padding 部分不参与）。
- **数据划分**：X_now 和 X_next 必须同步打乱，采用相同 random_state 的 sklearn train_test_split。

### 模型架构

- **TrajectoryTransformer**：输入投影 → 可学习位置编码 → Transformer Encoder → Decoder（可选）→ 输出头
- 默认 `USE_DECODER=True`，Decode r 使用可学习输出查询做交叉注意力。
  设为 False 时直接从 Encoder 输出回归（参数更少，收敛更快）。
- GELU 激活、Xavier 初始化、梯度裁剪 max_norm=1.0

### 训练策略

- AdamW 优化器 + ReduceLROnPlateau
- 早停（patience=20），保存在验证集上最优的模型
- 损失为标准化空间 MSE，评估指标在原始量纲计算

## 数据背景

- 物理场景：近地圆轨道 (480 km)，追踪航天器相对非机动目标逼近
- 坐标系：LVLH（目标在原点）
- 位置量级 ~0-200 km，速度量级 ~0-0.1 km/s
- X_next 包含真实仿真步 (1s 步长) 和可能的 CW 外推步 (60s 步长)
- 详见 `Dataset_new2/README.md`

## 环境

- Python 3.9+, PyTorch 2.0+, CUDA（可选）
- 安装：`pip install -r requirements.txt`
