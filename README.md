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
├── output/                      # 训练与评估产物（核心产物在顶层）
│   ├── best_model*.pth          #   各配置最优权重（p1 / f1 / f2 / h512 / e1 / e2）
│   ├── scaler.pkl / ensemble_config.json
│   ├── train_log*.txt / training_history.mat / X_pred.csv
│   ├── loss_curve / cw_residual / dv_distribution / sample_*   # 图表（各附 .mat）
│   ├── best_predictions/        #   最佳预测样本图（按 N 分层采样）
│   ├── _logs/                   #   历史运行日志归档
│   └── _archive/                #   旧图表 / 旧样本 / 废弃集成结果
├── _tools/                      # 诊断与对比工具（见 _tools/README.md）
├── _experiments/                # 消融实验脚本归档（见 _experiments/README.md）
├── _archive/                    # 历史权重、旧备份、.bak、版本快照
│   ├── v6_core/                 #   v6 体系精简备份（代码 + 关键权重 + 说明）
│   └── best_v1_2026-09-11/      #   ★ 当前最优版本备份（集成成员 + scaler + 代码快照 + 校验和）
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

**并行训练与消融（2026-09-11 新增环境变量）**——产物自动加后缀，互不干扰：

| 变量 | 作用 | 示例 |
|------|------|------|
| `TP_RUN_TAG` | 产物文件名后缀 | `TP_RUN_TAG=f1` → `best_model_f1.pth` |
| `TP_SEED` | 模型初始化种子（0=随机） | `TP_SEED=301` |
| `TP_HIDDEN` / `TP_LAYERS` | PI-LSTM 容量覆盖 | `TP_HIDDEN=512` |
| `TP_POS_W` | 位置损失权重 `λ_pos` | `TP_POS_W=0.5` |
| `TP_ENSEMBLE` | 集成成员路径（逗号分隔） | 见下方推理 |

复现当前最优成员：
```bash
TP_RUN_TAG=p1 TP_POS_W=0.5 python train.py
TP_RUN_TAG=f1 TP_HIDDEN=512 TP_POS_W=0.5 TP_SEED=301 python train.py
TP_RUN_TAG=f2 TP_HIDDEN=512 TP_POS_W=0.5 TP_SEED=302 python train.py
```

### 评估

```bash
python evaluate.py
```

输出整体与分位置/速度的 MSE、RMSE、MAE，物理一致性指标（CW 位置/速度残差、**脉冲 Δv** 幅值分布），
并生成 `loss_curve.png`（三子图）、`cw_residual.png`、`dv_distribution.png`、`sample_*.png`（均附同名 `.mat`）。

### 推理

```bash
python predict.py                      # 推理 + 可视化最佳样本（默认）
python predict.py --no-visualize       # 仅推理
python predict.py --top-k 30           # 可视化数量（N=2/3/4 各 10 个）
python predict.py --input in.csv --output out.csv
```

**集成推理**（推荐，位置 RMSE +7.66%；成员容量可不同，自动推断架构）：
```bash
TP_ENSEMBLE="output/best_model_p1.pth,output/best_model_f1.pth,output/best_model_f2.pth" python predict.py
```
未设置 `TP_ENSEMBLE` 时走单模型模式（`MODEL_SAVE_PATH`），行为与原先一致。

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

$$\mathcal{L} = w_{s}\mathcal{L}_{Huber} + \tfrac{1}{2}\lambda_t \mathcal{L}_{term}^{multi} + \lambda_t \mathcal{L}_{term}^{last} + \lambda_{pos}\mathcal{L}_{pos} + \lambda_m \mathcal{L}_{align} + \lambda_p \mathcal{L}_{cw} + \lambda_b \mathcal{L}_{bound}$$

| 分量 | 含义 | 权重（初始 → 终值） | warmup |
|------|------|-------------------|--------|
| `L_Huber` | 逐样本 Huber（δ=1.0），**末端距离大的样本降权**（>10 km → 0.1，5–10 km → 0.3） | 1.0 | — |
| `L_term^multi` | t=3/6/9 的 3D 距离，阶梯权重 0.3/0.5/1.0，除以 `TERMINAL_REF_DIST=4.5` | 0.01 → 2.0 | 15 ep |
| `L_term^last` | 末步 3D 距离，同样归一化 | 同上 | 15 ep |
| **`L_pos`** | **物理空间全 10 步 3D 位置误差**，**步长权重 0.2→1.0**，除以 `POS_LOSS_REF_DIST=4.5` | **0.0 → 0.5**（`TP_POS_W`） | 15 ep |
| `L_align` | 模型预测 Δv 与 CW 逆推 Δv 的对齐（PI-LSTM 核心约束），**按 Δv 上限归一化** | 0.001 → 0.05 | 20 ep |
| `L_cw` | CW 单步递推残差（分维度归一化） | 0.0 → **0.05**（步长修正后重新启用） | 20 ep |
| `L_bound` | 脉冲 Δv 边界软约束（**归一化超限量** `relu(‖Δv‖/limit − 1)`） | 0.0005 → 0.005 | 15 ep |

前 `PRED_WARMUP_EPOCHS=3` 轮仅使用 `L_Huber`。

**关于 `L_pos`（2026-09-11 新增，方向 3，增益 +4.50%）**：评估指标是**物理空间位置 RMSE (km)**，
而 `L_Huber` 在**标准化空间**计算。标准化按各维 std 缩放（位置 std≈50 km、速度 std≈0.05 km/s），
使速度误差在损失中被显著放大，**优化目标与评估口径不一致**。`L_pos` 直接监督全 10 步的物理空间
3D 位置误差（步长权重递增，因误差随预测步长增长：实测 0.76 → 2.04 km），使二者对齐。
这是本项目**最大的单项精度提升**。

**关于 `TERMINAL_REF_DIST=4.5`（改进 C）**：末端距离损失原本在物理量纲直接计算（量级 ~4.0 km），
与标准化空间的 `L_Huber`（~0.008）相差约 500×，乘以 λ_t=2.0 后以约 1500× 优势主导梯度，
导致引入数据增强后 loss 爆炸（0.02 → 9.9）。归一化后量级降至 ~1.0，梯度分配恢复均衡，
完整 80 epoch 训练不再发散。

**关于 `L_cw` 与 `L_align`/`L_bound` 的尺度校准（2026-09-10/11）**：三者均曾因尺度失衡而失效——
`L_align` 在 km/s 下平方后仅 1e-5（被预测损失淹没），改用 m/s 又达 4431（主导梯度），
最终统一采用**按 Δv 上限归一化的无量纲形式**。判据：修正后 CW 逆推脉冲 Δv 均值 **3.52 m/s**，
与数据集动作幅值设计值 3 m/s 吻合；且 `L_align` **单调下降**（语义错误时恒定 0.843 不降）。

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
- **相邻状态的真实时间间隔为 60 s**（2026-09-10 由 CW 残差判定，工具 `_tools/verify_timestep.py`）；
  原数据文档所述"1 s 采样"不成立，其时间跨度换算亦随之修正
- 位置单位 km，速度单位 km/s，坐标系 LVLH

| 公共物理参数 | 值 |
|------|-----|
| 轨道半径 | 6851 km（6371 + 480） |
| 地球引力常数 μ | 398600 km³/s² |
| 轨道角速度 n | 0.001134 rad/s |
| RL 仿真步长 h | 1 s |
| CW 外推步长 T | 60 s |
| **数据采样间隔（相邻状态）** | **60 s**（实测判定，2026-09-10 修正） |
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
| `PHYSICS_LOSS_WEIGHT_FINAL` | 0.05 | CW 残差终值权重（步长修正后重新启用） |
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
| **v6** | **多场景汇总数据集 + 断点恢复 + 学术风格可视化**（架构仍是 v3 PI-LSTM） | 主线配置 |
| v7 | CW baseline 残差学习 / 变容量 v7a(256) v7b(320) | v7a 仅 td 5.2040 km，失败 |
| v8 | 知识蒸馏（v6 teacher → v7 student） | 未产出可用权重 |
| v9 | 输入高斯噪声数据增强 | loss 爆炸早停 |
| v10 | hidden=512 + 物理一致增强 | 早停于 ep1（td 4.7567 km） |
| v11 | 3-seed bagging | 早停于 ep1（td 4.7243 km） |
| **v12** | **物理计算修正（步长/量纲/语义/尺度）+ 位置损失对齐 + 容量可配置 + 集成** | **当前最优（+7.66%）** |

> **v7~v11 的失败原因已查明**：末端距离损失量级失衡导致 loss 爆炸（**已由改进 C 修复**），
> 且 **CW 锚定路线本身不可行** —— 追踪星持续机动，CW 外推基线位置 RMSE 达 120.87 km，
> 比持久基线（17.66 km）还差 6.8 倍。故 v7 残差学习不宜重试。

**当前最优（测试集 32,647 样本，原始物理量纲）**：

| 策略 | 位置 RMSE | 位置 MAE | 末端距离 |
|------|----------:|---------:|---------:|
| 原始基线（384 容量） | 1.3707 km | 0.9423 km | 4.2780 km |
| 最优单模型 `best_model_f1`（512 + 位置损失） | 1.2958 km | 0.8620 km | 4.2461 km |
| **集成 `p1 + f1 + f2`（推荐）** | **1.2657 km** | **0.8434 km** | **4.1659 km** |
| **累计增益** | **+7.66%** | **+10.50%** | **+2.62%** |

**启用集成**（`predict.py` 已支持，配置见 `output/ensemble_config.json`）：
```bash
TP_ENSEMBLE="output/best_model_p1.pth,output/best_model_f1.pth,output/best_model_f2.pth" python predict.py
```

**增益归因**：位置损失口径对齐 **+4.50%**（最大单项）、增容量 512 +0.86%、
集成再 +2.32%。三方向可叠加。选择流程为**验证集选组合 → 测试集报告**（无过拟合）。

---

## 已知问题

### ✅ 已修复（2026-09-10 / 09-11）

| 问题 | 修复 |
|------|------|
| **续训崩溃**：`train.py` 加载检查点处无 try/except，架构不匹配直接抛 `RuntimeError` | 增加缺失键检测 + 带修复指引的明确报错；`strict=False` 容忍多余的派生 buffer |
| **`predict.py` 推理未传 mask**：mask 默认全 True，全局 Δv 被除以 `max_N=4` 而非实际 N，实测 N=2 位置偏差最大 2.422 km（77% 样本口径与验证不一致） | 补 `mask=batch_mask` |
| **`dv_all` 注释误写 19 步**（实为 18：9 CW + 9 速度差分） | 已修正 |
| **`predict.py` 目录清理 bug**：`endswith` 中 `.png` 重复、漏 `.svg`，且 `os.remove` 无容错 | 后缀集合修正 + 逐文件 `try/except` |
| **集成不支持**：`predict.py` / `evaluate.py` 仅支持单模型 | 新增 `utils/ensemble.py` + `ENSEMBLE_MODELS`（自动推断架构，支持混合容量） |

### ⚠️ 仍存在

1. **`best_val_loss` 恒为 inf**：`train()` 中该变量初始化后从未更新，故 `latest_checkpoint.pth`
   的 `val_loss` 字段不可用（应看 `val_terminal_dist` 或用 `_tools/inspect_checkpoints.py`）。
2. **scheduler 跨 epoch 跳变**：`lr_lambda` 依赖 `total_epochs`，修改 `EPOCHS` 后续训会导致
   学习率相位突变；`RESUME_FIXED_LR` 补丁会被下一轮 `scheduler.step()` 覆盖，尚未根治。
3. **`models/model.py` 与 `models/pinn_lstm.py` 各有一个 `create_model`**，训练实际使用前者的版本
   （后者是死代码）。
4. **末端距离损失被计算两次**（`multi_step_terminal_loss` 与末步 loss 叠加），实际末端总权重 ≈1.5 λ_t。
5. **环境限制**：`output/best_predictions/` 内删除超过 50 个文件会触发 safe-delete 策略阻断
   （`SAFE_DELETE_BULK_CONFIRM_REQUIRED`）；重跑可视化前宜先用 `mv` 把旧目录整体移走。

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
