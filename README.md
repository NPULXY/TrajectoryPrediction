# 航天器相对运动轨迹预测

基于 **物理信息条件 LSTM（PI-LSTM）** 的序列到序列轨迹预测模型：给定追踪航天器相对于目标的过去 10 步运动状态，预测未来 10 步的状态序列。

> GitHub: <https://github.com/NPULXY/TrajectoryPrediction>
> 当前主线版本：**v6 体系**（PI-LSTM v3 架构 + v6 训练配置），详见 [版本谱系](#版本谱系)

---

## 任务概述

**物理场景**：近地圆轨道（高度 480 km，轨道角速度 n ≈ 0.001134 rad/s），目标位于 LVLH 坐标系原点。追踪航天器通过脉冲机动逼近/拦截非机动目标，Δv 幅值 ≤ 3 m/s。所有样本均为机动段，不存在纯自由漂移段。

**核心挑战**：仅凭 10 步（9 秒）的观测窗口推断当前机动状态并外推未来 10 步，要求模型具备强归纳偏置。

**场景覆盖**：交会（Capture）、阻扰（Obstruction）、探测（Detection）、潜伏（Lurk）、混合（Mix）。

---

## 项目结构

```
TrajectoryPrediction/
├── config.py                    # 全部超参数与路径（单一配置源）
├── train.py                     # 训练主循环（断点恢复 / 早停 / .mat 历史）
├── evaluate.py                  # 测试集评估 + 物理一致性指标 + 可视化
├── predict.py                   # 推理 + 最佳预测样本可视化
├── plot_training_history.py     # 训练历史独立绘图
├── plot_best_predictions.py     # 最佳预测样本独立绘图
├── launch_train.py              # Windows 无窗体启动封装
├── run_train.bat / .vbs         # 启动脚本
├── requirements.txt
├── models/                      # ★ 模型定义
│   ├── model.py                 #   标准 TrajectoryLSTM + create_model 工厂（训练实际入口）
│   ├── pinn_lstm.py             #   PI-LSTM v3（当前架构）
│   ├── physics_loss.py          #   CW 矩阵 / 残差 / Δv 边界 / Δv alignment（全向量化）
│   ├── cw_baseline.py           #   标准化空间 CW 演化基线（v7 残差学习用）
│   ├── transformer_pi.py        #   v5 Transformer-PI（备选架构）
│   └── pinn_lstm_v7.py          #   v7a/v7b 变容量变体（集成多样性用）
├── utils/data_loader.py         # CSV 解析 / padding / z-score / DataLoader / 指标
├── Dataset_Summary/             # 汇总数据集（217,642 样本，主用）
├── Dataset_new2/                # 交会场景子集（23,787 样本）
├── output/                      # 训练与评估产物（权重 / scaler / 日志 / 图表）
├── _tools/                      # 诊断与对比工具（见 _tools/README.md）
├── _experiments/                # 消融实验脚本归档（见 _experiments/README.md）
├── _archive/                    # 历史权重、旧备份、.bak、v6 体系快照
│   └── v6_core/                 #   ★ v6 体系精简备份（代码 + 关键权重 + 说明）
├── _docs/                       # 历史提示词存档与参考文献
└── _rendered/                   # 渲染产物
```

---

## 环境依赖

- Python ≥ 3.9
- PyTorch ≥ 2.0（本项目验证于 PyTorch 2.1.0+cu118）
- NumPy / Pandas / scikit-learn / Matplotlib / tqdm / SciPy

```bash
pip install -r requirements.txt
```

推荐环境：conda `torch` 环境。脚本内含 Windows conda MKL DLL 路径自动修复。

---

## 快速开始

所有命令均在**项目根目录**执行。

### 训练

```bash
python train.py
```

流程：加载 `Dataset_Summary/` → 同步打乱并按 70/15/15 划分（seed=42）→ 训练集拟合 z-score scaler
→ 构建模型 → Cosine warmup + annealing 调度 → 以 `val_terminal_dist_mean` 为判据早停
→ 保存 `output/best_model.pth` / `latest_checkpoint.pth` / `scaler.pkl` / `training_history.mat`。

### 评估

```bash
python evaluate.py
```

输出整体与分位置/速度的 MSE、RMSE、MAE，物理一致性指标（CW 残差、Δv 幅值分布），并生成
`loss_curve.png`、`cw_residual.png`、`dv_distribution.png`、`sample_*.png`（均附同名 `.mat`）。

### 推理

```bash
python predict.py                      # 推理 + 可视化最佳样本（默认）
python predict.py --no-visualize       # 仅推理
python predict.py --top-k 30           # 可视化数量（N=2/3/4 各 10 个）
python predict.py --input in.csv --output out.csv
```

---

## 模型架构（PI-LSTM v3，当前主线）

```
输入 (B, 10, 24) + mask
  → DeltaVEstimator：CW 逆推估计每步脉冲 Δv（伪逆 B_eff⁺，可微，全向量化）
  → dv_global = 有效目标 Δv 的均值                      (B, 9, 3)
  → ConditionBuilder：mode_embed(2×8) + MLP(Δv) → 8 维条件向量
  → 扩展输入 = concat(state, cond) 逐目标 → 4 × (6+8) = 56 维
  → Encoder LSTM（4 层 × hidden 384）+ LayerNorm
  → 可学习 query 注意力聚合上下文
  → 位置专用 head / 速度专用 head（各自 MLP）
  → delta = concat(pos, vel)                            (B, 10, 24)
  → pred = persistence + delta（持久预测锚定，非自回归）
  → 辅助输出 dv_all = concat(9 步 CW 估计, 9 步速度差分)  (B, 18, max_N×3)
```

**可训练参数：4,915,176**

| 关键设计 | 方案 | 说明 |
|---------|------|------|
| 序列主干 | LSTM（4 层 × 384） | 10 步短序列上显著优于 Transformer（实测 v5 劣化 27%） |
| 预测策略 | 非自回归 delta 预测 | 一次性输出 10 步，消除自回归误差累积；锚定持久预测保证不劣于朴素基线 |
| 物理先验 | CW 方程 + 可微 Δv 估计 | 用伪逆从相邻状态反解脉冲 Δv，作为条件信号注入 |
| 条件门控 | 模式嵌入 + Δv MLP | 显式注入机动强度信息，自适应调整预测行为 |
| 训练稳定 | LayerNorm + 正交初始化 + forget bias=1 | 缓解梯度消失 |

> ⚠️ `hidden_size=384` / `num_layers=4` 硬编码于 `models/pinn_lstm.py`，
> `config.py` 中的 `D_MODEL` / `NUM_LSTM_LAYERS` 对 PI-LSTM **不生效**（仅供标准 LSTM 使用）。

---

## 损失函数

$$\mathcal{L} = w_{s}\mathcal{L}_{Huber} + \tfrac{1}{2}\lambda_t \mathcal{L}_{term}^{multi} + \lambda_t \mathcal{L}_{term}^{last} + \lambda_m \mathcal{L}_{align} + \lambda_p \mathcal{L}_{cw} + \lambda_b \mathcal{L}_{bound}$$

| 分量 | 含义 | 权重（初始 → 终值） | warmup |
|------|------|-------------------|--------|
| `L_Huber` | 逐样本 Huber（δ=1.0），**末端距离大的样本降权**（>10 km → 0.1，5–10 km → 0.3） | 1.0 | — |
| `L_term^multi` | t=3/6/9 的 3D 距离，阶梯权重 0.3/0.5/1.0，除以 `TERMINAL_REF_DIST=4.5` | 0.01 → 2.0 | 15 ep |
| `L_term^last` | 末步 3D 距离，同样归一化 | 同上 | 15 ep |
| `L_align` | 模型预测 Δv 与 CW 逆推 Δv 的对齐（PI-LSTM 核心约束） | 0.001 → 0.05 | 20 ep |
| `L_cw` | CW 单步递推残差（分维度归一化） | 0.0 → **1e-7**（基本关闭） | 20 ep |
| `L_bound` | Δv 边界软约束（超出 3 m/s 惩罚，原始量纲） | 0.0005 → 0.005 | 15 ep |

前 `PRED_WARMUP_EPOCHS=3` 轮仅使用 `L_Huber`。

**关于 `TERMINAL_REF_DIST=4.5`（改进 C）**：末端距离损失原本在物理量纲直接计算（量级 ~4.0 km），
与标准化空间的 `L_Huber`（~0.008）相差约 500×，乘以 λ_t=2.0 后以约 1500× 优势主导梯度，
导致引入数据增强后 loss 爆炸（0.02 → 9.9）。归一化后量级降至 ~1.0，梯度分配恢复均衡，
完整 80 epoch 训练不再发散。

**关于 `L_cw` 被关闭**：数据集相邻步为 1 s 间隔，但用于构建 `X_next` 的部分样本含 60 s 步长的
CW 外推补齐点，与 `Φ_h(1s)` 假设不符，故 CW 单步残差约束不适用于本数据，仅保留极小权重作数值稳定项。

---

## 数据集

默认 `Dataset_Summary/`（217,642 样本），保留 `Dataset_new2/`（交会子集）供对比。

| 场景 | 样本数 | 占比 | 初始距离 | 成功阈值 |
|------|-------:|:---:|---------|---------|
| 交会 Capture | 23,787 | 10.9% | 100–120 km | ≤ 5 km |
| 阻扰 Obstruction | 39,100 | 18.0% | 20–100 km | ≤ 5 km |
| 探测 Detection | 21,166 | 9.7% | 35–120 km | ≤ 35 km |
| 潜伏 Lurk | 24,766 | 11.4% | 100–150 km | ≤ 50 km |
| 混合 Mix | 108,823 | 50.0% | 混合 | 混合 |
| **合计** | **217,642** | 100% | — | — |

实测 N 分布：N=2 → 100,451（46.2%）｜N=3 → 66,967（30.8%）｜N=4 → 50,224（23.1%）
实测划分（seed=42）：训练 152,349 / 验证 32,646 / 测试 32,647

**数据格式**（详见 `Dataset_Summary/README.md`）：

- `X_now.csv` / `X_next.csv`：每行为**嵌套列表字符串** `[[step1], …, [step10]]`，
  每步 N×6 个浮点数（N 个目标 × [x,y,z,vx,vy,vz]）。**必须用 `json.loads` 逐行解析，不能按逗号分列。**
- `Y.csv`：`[N, min_distance, phi]`
- **`X_now` / `X_next` / `Y` 行序严格对应**，且汇总版已在合并时全局打乱，**绝不可单独打乱任一文件**
- `X_next` 可能混入 CW 外推补齐点（步长 60 s），并非全是 1 s 步长的仿真步
- 位置单位 km，速度单位 km/s，坐标系 LVLH

| 公共物理参数 | 值 |
|------|-----|
| 轨道半径 | 6851 km（6371 + 480） |
| 地球引力常数 μ | 398600 km³/s² |
| 轨道角速度 n | 0.001134 rad/s |
| RL 仿真步长 h | 1 s |
| CW 外推步长 T | 60 s |
| Δv 上限 | 3 m/s |

---

## 训练策略

| 配置 | 值 |
|------|-----|
| 优化器 | AdamW（weight_decay 1e-5） |
| 学习率 | 1e-3 → 1e-6（Cosine warmup 3 epoch + annealing） |
| 批大小 | 384 |
| 最大轮数 | 80 |
| 早停耐心 | 20（判据：`val_terminal_dist_mean`，**非** val loss） |
| Dropout | 0.15 |
| 梯度裁剪 | max_norm = 1.0 |
| 断点恢复 | `RESUME_TRAINING=True`（优先 `latest_checkpoint.pth`，回退 `best_model.pth`） |

---

## 评估与可视化

- **指标**：整体及分位置/速度的 MSE / RMSE / MAE（原始量纲）；末端距离统计（均值/中位数/min/max、<1 km 与 <100 m 成功率）；Δv 超限率
- **物理一致性**：CW 单步残差分布、Δv 幅值分布
- **图表**：`loss_curve.png`、`cw_residual.png`、`dv_distribution.png`、`sample_*.png`、`best_predictions/top*.png`
- 所有图表均输出同名 `.mat`，可直接用 MATLAB 后处理
- 图表采用英文学术风格（Times New Roman + STIX 数学字体）

**当前主线实测（217,642 全量测试集）**：位置 RMSE 1.343 km / MAE 0.925 km；速度 RMSE 4.71e-3 km/s；
最佳样本末端距离 N=2 → 0.2133 km，N=3 → 0.6741 km，N=4 → 0.7569 km。

---

## 配置说明

全部参数集中于 `config.py`。常用项：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `PHYSICS_ENABLED` | `True` | 启用 PI-LSTM（False 则回退标准 LSTM） |
| `USE_TRANSFORMER` | `False` | True 则改用 v5 Transformer-PI |
| `CONDITION_EMBED_DIM` | 8 | 条件向量维度 |
| `BATCH_SIZE` | 384 | 批大小 |
| `EPOCHS` | 80 | 最大训练轮数 |
| `EARLY_STOP_PATIENCE` | 20 | 早停耐心 |
| `TERMINAL_REF_DIST` | 4.5 | 末端距离损失归一化参考值（km） |
| `TERMINAL_LOSS_WEIGHT_FINAL` | 2.0 | 末端距离损失终值权重 |
| `MODE_LOSS_WEIGHT_FINAL` | 0.05 | Δv alignment 终值权重 |
| `PHYSICS_LOSS_WEIGHT_FINAL` | 1e-7 | CW 残差终值权重（基本关闭） |
| `DELTAV_LIMIT` | 3.0 | Δv 幅值上限（m/s） |
| `RESUME_TRAINING` | `True` | 断点恢复 |
| `RESUME_FIXED_LR` | 1e-6 | 续训时强制低学习率 |

---

## 版本谱系

> ⚠️ 本项目存在**两套并存的版本编号**，是最易混淆之处：
> **代码内的架构编号**（`pinn_lstm.py` 自称 v3、`transformer_pi.py` 自称 v5）
> 与 **训练配置/实验编号**（v5~v11 训练脚本）。
> 目录名 `_archive/v6_core/` 与文件名 `*_v6_*.pth` 中的 "v6" 指**后者**。

| 配置版本 | 内容 | 结果 |
|---------|------|------|
| v4 | LSTM Encoder-Decoder + Delta 残差 + Teacher forcing | 验证 loss 0.017，效果良好 |
| v5 | Transformer-PI（自注意力替代 LSTM） | **劣化 27%**，早停（td 5.4947 km） |
| **v6** | **多场景汇总数据集 + 断点恢复 + 学术风格可视化**（架构仍是 v3 PI-LSTM） | **当前主线** |
| v7 | CW baseline 残差学习 / 变容量 v7a(256) v7b(320) | v7a 仅 td 5.2040 km，失败 |
| v8 | 知识蒸馏（v6 teacher → v7 student） | 未产出可用权重 |
| v9 | 输入高斯噪声数据增强 | loss 爆炸早停 |
| v10 | hidden=512 + 物理一致增强 | 早停于 ep1（td 4.7567 km） |
| v11 | 3-seed bagging | 早停于 ep1（td 4.7243 km） |

**v7~v11 的共同失败原因**：末端距离损失量级失衡导致 loss 爆炸，**已由改进 C 修复**，
因此这些实验在原理上可基于当前主线重试（脚本见 `_experiments/`）。

**当前最优（统一验证集口径）**：

| 策略 | td_max | 位置 RMSE |
|------|-------:|----------:|
| v6 与 v6o 等权集成 | **4.2400 km** | **1.3100 km** |
| v6o 单独 | 4.2936 km | 1.3478 km |
| v6 单独 | 4.3171 km | 1.3227 km |

即**集成优于最优单模型约 1.25%**；集成需同时加载两个模型，而 `predict.py` / `evaluate.py`
目前仅支持单模型，启用需先改造。

---

## 已知问题

1. **续训可能崩溃**：`train.py` 加载检查点处（`RESUME_TRAINING=True` 分支）**没有 try/except**。
   若 `output/latest_checkpoint.pth` 残留了其他架构（如 v5 Transformer）的权重，
   `load_state_dict(strict=True)` 会直接抛 `RuntimeError`。
   **切换架构或权重来源后，请先清理该文件。**
2. **`predict.py` 推理未传 mask**：`model(batch, return_dv=True)` 使 mask 默认全 True，
   导致全局 Δv 被除以 `max_N=4` 而非实际 N。实测 N=2 样本位置偏差平均 0.271 km（最大 2.422 km），
   N=3 平均 0.171 km，N=4 无偏差——即 **77% 的样本推理口径与验证不一致**。
   （`evaluate.py` 传了 mask，是正确的。）
3. **`best_val_loss` 恒为 inf**：`train()` 中该变量初始化后从未更新，故 `latest_checkpoint.pth`
   的 `val_loss` 字段不可用（应看 `val_terminal_dist` 或用 `_tools/inspect_checkpoints.py`）。
4. **scheduler 跨 epoch 跳变**：`lr_lambda` 依赖 `total_epochs`，修改 `EPOCHS` 后续训会导致
   学习率相位突变；`RESUME_FIXED_LR` 补丁会被下一轮 `scheduler.step()` 覆盖，尚未根治。
5. **`dv_all` 为 18 步**（9 CW + 9 速度差分），代码注释中多处误写为 19。
6. **`models/model.py` 与 `models/pinn_lstm.py` 各有一个 `create_model`**，训练实际使用前者的版本。

---

## 引用

```bibtex
@software{trajectory_prediction,
  author = {NPULXY},
  title = {航天器相对运动轨迹预测},
  year = {2026},
  url = {https://github.com/NPULXY/TrajectoryPrediction}
}
```

## 许可证

本项目仅供学术研究使用。
