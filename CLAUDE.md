# CLAUDE.md

> GitHub: <https://github.com/NPULXY/TrajectoryPrediction>
> 航天器相对运动轨迹预测 —— PI-LSTM 序列到序列模型

## 项目定位

给定追踪航天器相对目标的**过去 10 步**状态（位置 km + 速度 km/s，LVLH 坐标系），预测**未来 10 步**。
数据集覆盖交会/阻扰/探测/潜伏/混合五场景，共 217,642 样本。

**当前主线 = v6 体系（2026-09-11 经物理修正与精度优化）**：
- 架构：**PI-LSTM v3**（`models/pinn_lstm.py`，`PhysicsInformedTrajectoryLSTM`），
  容量由 `PI_HIDDEN_SIZE`/`PI_NUM_LAYERS` 配置（默认 384/4，可训 512）
- 配置：`config.py` 中 `PHYSICS_ENABLED=True`、`USE_TRANSFORMER=False`
- **最优单模型**：`output/best_model_f1.pth`（hidden 512 + 位置损失，测试集位置 RMSE **1.2958 km**）
- **最优集成（推荐）**：`best_model_p1` + `best_model_f1` + `best_model_f2` 等权
  （测试集位置 RMSE **1.2657 km**，相对原始基线 1.3707 km **+7.66%**）
  配置见 `output/ensemble_config.json`，启用：`TP_ENSEMBLE="output/best_model_p1.pth,output/best_model_f1.pth,output/best_model_f2.pth"`
- **备份**：`_archive/best_v1_2026-09-11/`（权重 + scaler + 代码快照 + 校验和 + 说明）

> ⚠️ **两套版本编号并存，最易混淆**：
> - **代码内架构编号**：`pinn_lstm.py` 自称 **v3**、`transformer_pi.py` 自称 **v5**
> - **训练配置编号**：v5~v11 训练脚本，**v6 = 当前主线配置**
>
> `_archive/v6_core/` 与 `*_v6_*.pth` 中的 "v6" 指后者。

## 运行环境

| 项 | 值 |
|----|-----|
| Conda 环境 | `C:\Users\Hasee\anaconda3\envs\torch`（Python 3.9.18 + PyTorch 2.1.0+cu118） |
| 工作目录 | **必须从项目根目录运行**（脚本内 `sys.path.insert(0, '.')` 依赖 cwd） |
| GPU | CUDA 可用时自动使用 |

```bash
python train.py       # 训练（默认断点恢复，约 12 s/epoch）
python evaluate.py    # 测试集评估 + 可视化（约 30 s）
python predict.py     # 推理 + 最佳样本可视化（约 1 min）
```

## ⚠️ 已知陷阱与修复状态

### ✅ 已于 2026-09-10 修复

| 问题 | 修复方式 |
|------|---------|
| **★ 时间步长错误**：数据真实步长为 **60 s**，但代码按 1 s 计算 | `config.CW_DT_H` 1.0 → **60.0**；`DeltaVEstimator(dt=)` 与 `PhysicsLoss(dt_h=)` 改为显式读 config（原为硬编码/默认值）；`Phi/B_eff/B_pinv` 改为 `persistent=False`（派生矩阵不再写入 checkpoint，避免旧权重携带错误步长矩阵） |
| **Δv 边界约束语义错误**：对相邻步速度差分施加约束，而该差分含 60 s 自由演化（~4 m/s） | 改为约束**脉冲 Δv**（扣除 `Φ_h·x_t` 自由演化项），见 `physics_loss._velocity_change_loss` |
| **续训无容错**：架构不匹配时 `load_state_dict` 直接抛 `RuntimeError` 且无提示 | `train.py` 加载处增加缺失键检测：缺失即抛带修复指引的异常；`strict=False` 容忍多余的派生 buffer |
| **`predict.py` 推理未传 mask**：`dv_global` 被除以 `max_N=4` 而非实际 N | 补 `mask=batch_mask`（实测原缺陷使 N=2 位置偏差平均 0.271 km、最大 2.422 km） |
| **`predict.py` 目录清理 bug**：`endswith(('.png','.mat','.png'))` 中 `.png` 重复、漏 `.svg`，且 `os.remove` 无容错 | 后缀集合改为 `(.png,.mat,.svg)`，逐文件 `try/except OSError` |
| **损失曲线误导**：主曲线用"总损失"，其上升-峰值形状源自 λ_t warmup，易误读为发散 | 改为三子图：预测损失（对数轴）/ 验证末端距离 td / 物理损失分量 |
| **CW 残差单位混标**：位置 (km) 与速度 (km/s) 混为单一欧氏范数却标 "km/s" | 拆为位置、速度两子图分别统计 |
| **Δv 分布图口径**：绘制的"速度差分"含自由演化，超限率虚高 | 改为绘制**脉冲 Δv** 并给出超限占比 |
| `dv_all` 步数注释（误写 19，实为 18） | 已修正 |

**步长判定依据**（工具 `_tools/verify_timestep.py`）：CW 残差在 dt=60 s 时最小 ——
X_now 3.4972→**0.2114**、X_next 4.3112→**0.3267**（小 16 倍）；旁证：序列内相邻步位移 3.5–4.4 km，
按 1 s 对应 3.5 km/s（不合理）、按 60 s 为 0.059 km/s（合理）；修正后 CW 逆推脉冲 Δv 均值
**3.5 m/s**，与数据集动作幅值设计值 3 m/s 吻合。

### ⚠️ 仍存在（未修复）

1. **`hidden_size=384` / `num_layers=4` 硬编码**在 `pinn_lstm.py`，`config.py` 的
   `D_MODEL` / `NUM_LSTM_LAYERS` **对 PI-LSTM 不生效**。
2. **`best_val_loss` 恒为 inf**（`train()` 中初始化后从未更新）→ `latest_checkpoint.pth` 的
   `val_loss` 字段不可用，请改看 `val_terminal_dist`。
3. **scheduler 跨 epoch 跳变**：`lr_lambda` 依赖 `total_epochs`，改动 `EPOCHS` 后续训会导致
   学习率相位突变（ep81 从 1e-6 跳回 6.6e-4）；`RESUME_FIXED_LR` 补丁会被 `scheduler.step()` 覆盖。
4. **两个 `create_model` 重复**：`models/model.py` 与 `models/pinn_lstm.py` 各有一个；
   训练实际用前者，后者是死代码。
5. **末端距离损失被计算两次**（`multi_step_terminal_loss` 与末步 loss 叠加），实际总权重 ≈ 1.5 λ_t。

## 数据硬约束

- `X_now.csv` / `X_next.csv` / `Y.csv` **行序严格对应**，且汇总版已在合并时**全局打乱**——
  **绝不可单独打乱任一文件**
- 每行是**单列嵌套列表字符串** `[[step1], …, [step10]]`，**必须 `json.loads` 逐行解析**，
  不能用 `pd.read_csv` 默认逗号分隔
- N ∈ {2,3,4} → 每步 12/18/24 维，代码 padding 至 `MAX_DIM=24` 并生成 mask
- **相邻状态的真实时间间隔 = 60 s**（2026-09-10 实测判定；原数据文档标注的 "1 s" 有误）
- `FeatureScaler` 逐维 z-score，**只统计非 padding 值**，在训练集上拟合；
  `scaler.pkl` 必须与权重配套使用，换数据集须重新拟合
- DataLoader **num_workers 默认 0**（Windows 多进程 pickling 嵌套列表会出错）

## 模型架构（PI-LSTM v3）

```
输入 (B,10,24) + mask
  → DeltaVEstimator：CW 逆推 Δv（伪逆 B_eff⁺，可微，向量化）      (B, 9, max_N×3)
  → dv_global = 有效目标 Δv 均值                                 (B, 9, 3)
  → ConditionBuilder：mode_embed(2×8) + MLP(Δv) → 8 维条件
  → 扩展输入 = concat(state, cond) 逐目标 → 4×(6+8) = 56 维
  → Encoder LSTM（4 层 × 384）+ LayerNorm
  → 可学习 query 注意力聚合上下文
  → 位置 head / 速度 head（各 MLP）→ delta                       (B, 10, 24)
  → pred = persistence + delta（非自回归，一次出 10 步）
  → dv_all = concat(9 CW 估计, 9 速度差分)                        (B, 18, max_N×3)
```

## 损失函数

$$\mathcal{L} = w_s\mathcal{L}_{Huber} + \tfrac{1}{2}\lambda_t \mathcal{L}_{term}^{multi} + \lambda_t \mathcal{L}_{term}^{last} + \lambda_{pos}\mathcal{L}_{pos} + \lambda_m \mathcal{L}_{align} + \lambda_p \mathcal{L}_{cw} + \lambda_b \mathcal{L}_{bound}$$

| 分量 | 终值权重 | warmup |
|------|---------|--------|
| `L_Huber`（逐样本，末端>10 km 降权至 0.1、5–10 km 至 0.3） | 1.0 | — |
| `L_term^multi`（t=3/6/9，权重 0.3/0.5/1.0，÷`TERMINAL_REF_DIST`=4.5） | 0.01→2.0 | 15 ep |
| `L_term^last`（末步，同样归一化） | 同上 | 15 ep |
| **`L_pos`（物理空间全 10 步位置误差，步长权重 0.2→1.0）** | **0→0.5**（`TP_POS_W`） | 15 ep |
| `L_align`（模型 Δv 对齐 CW 逆推 Δv，**按 Δv 上限归一化**） | 0.001→0.05 | 20 ep |
| `L_cw`（CW 单步残差，分维度归一化） | 0→**0.05**（步长修正后重新启用） | 20 ep |
| `L_bound`（**归一化超限量** `relu(‖Δv‖/limit−1)`） | 0.0005→0.005 | 15 ep |

前 `PRED_WARMUP_EPOCHS=3` 轮仅用 `L_Huber`。

**改进 C（2026-09-10）**：末端损失原本在物理量纲直接算 3D 距离（~4.0 km），与标准化空间
`L_Huber`（~0.008）差约 500×，λ_t=2.0 后以 ~1500× 主导梯度 → 数据增强时 loss 爆炸。
引入 `TERMINAL_REF_DIST=4.5` 归一化后量级降至 ~1.0，训练稳定（train loss 12.6 → 2.81）。

**方向 3（2026-09-11，增益 +4.50%，最大单项）**：新增 `L_pos` 直接监督物理空间 3D 位置误差。
原因：评估指标是物理空间位置 RMSE，而 `L_Huber` 在标准化空间（位置 std≈50 km、速度 std≈0.05 km/s，
速度误差被放大），**优化目标与评估口径不一致**。步长权重递增（0.2→1.0）因误差随预测步长增长
（实测 0.76 → 2.04 km）。

**Δv 类损失的尺度校准（2026-09-10/11）**：`L_align` 与 `L_bound` 均曾因尺度失衡失效——
km/s 下平方后仅 1e-5（被淹没），改 m/s 又达 4431（主导梯度），最终统一为**按 Δv 上限归一化的
无量纲形式**。验收判据：CW 逆推脉冲 Δv 均值 **3.52 m/s**（设计值 3 m/s），且 `L_align` 单调下降
（语义错误时恒定 0.843 不降）。

## 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 / 学习率 | AdamW，1e-3 → 1e-6（Cosine warmup 3 ep + annealing） |
| BATCH_SIZE / EPOCHS | 384 / 80 |
| EARLY_STOP_PATIENCE | 20（判据 `val_terminal_dist_mean`，**非** val loss） |
| DROPOUT / 梯度裁剪 | 0.15 / max_norm 1.0 |
| WEIGHT_DECAY | 1e-5 |
| 数据集划分 | 70/15/15，seed=42 → 152,349 / 32,646 / 32,647 |
| RESUME_TRAINING | True（优先 `latest_checkpoint.pth`，回退 `best_model.pth`） |
| 数据增强 | **关闭**（经核验模型无过拟合，开启反而变差） |

**并行训练与消融的环境变量**（2026-09-11 新增，产物自动加后缀互不干扰）：

| 变量 | 作用 |
|------|------|
| `TP_RUN_TAG` | 产物文件名后缀（`best_model_<tag>.pth` / `train_log_<tag>.txt`） |
| `TP_SEED` | 模型初始化种子（0=随机；数据划分不受影响，由 `load_and_split` 内部固定） |
| `TP_HIDDEN` / `TP_LAYERS` | PI-LSTM 容量覆盖（默认 384 / 4） |
| `TP_POS_W` | 位置损失权重的终值 `λ_pos`（默认 0.0 = 不启用） |
| `TP_ENSEMBLE` | 集成成员路径（逗号分隔），供 `predict.py` 使用 |

> ⚠️ **容量变更会导致旧检查点不可用**（键形状不匹配）。
> **`strict=False` 会静默保留随机初始化**，故加载不同容量的权重必须按检查点形状
> 推断架构（见 `utils/ensemble.py::infer_arch`）。

## 目录约定

| 目录 | 用途 |
|------|------|
| `models/`、`utils/` | 模型定义与数据管线（核心代码） |
| `output/` | 训练与评估产物：核心产物在顶层；`_logs/` 归档历史运行日志；`_archive/` 归档旧图表/旧样本/废弃集成结果——**整体 git 忽略** |
| `_tools/` | 诊断与对比工具（有效清单见 `_tools/README.md`） |
| `_experiments/` | v7~v11 消融脚本归档（脚本内路径引用已失效，见其 README） |
| `_archive/v6_core/` | **v6 体系精简备份**（代码 + 2 个关键权重 + 完整说明） |
| `_archive/best_v1_2026-09-11/` | **★ 当前最优版本备份**（3 个集成成员权重 + scaler + `ensemble_config.json` + 代码快照 + `CHECKSUMS.md5` + README） |
| `_archive/weights/` | 历史权重（backup_v2/v3/v6/v7/v10/v11 + 各组实验的 best/latest） |
| `_docs/` | 历史提示词存档与参考文献 |
| `_rendered/` | 渲染产物 |

## 评估口径（关键）

**标准口径**：seed=42 划分（train 152,349 / val 32,646 / test 32,647），
`td` 对齐 `train.py::validate()` —— 每样本取「各有效目标末步 3D 距离的**最大值**」再对样本求均值。

### 当前最优（2026-09-11，测试集 32,647 样本）

| 方案 | 位置 RMSE | 位置 MAE | 末端距离 |
|------|----------:|---------:|---------:|
| 原始基线（384 容量） | 1.3707 km | 0.9423 km | 4.2780 km |
| 最优单模型 `best_model_f1`（512 + 位置损失） | 1.2958 km | 0.8620 km | 4.2461 km |
| **集成 `p1 + f1 + f2`（推荐）** | **1.2657 km** | **0.8434 km** | **4.1659 km** |
| **累计增益** | **+7.66%** | **+10.50%** | **+2.62%** |

> 选择流程：在**验证集**上枚举 91 个子集选组合（val RMSE 1.2726），再在**测试集**报告（1.2657）
> —— 避免测试集过拟合。工具：`_tools/select_ensemble.py`

### 历史结论（2026-09-10 前，步长修正前的旧模型，**不宜引用**）

| 策略 | td_max | 位置 RMSE |
|------|-------:|----------:|
| v6 + v6o 等权集成 | 4.2400 km | 1.3100 km |
| v6o 单独 | 4.2936 km | 1.3478 km |
| v6 单独 | 4.3171 km | 1.3227 km |

> ⚠️ `_tools/ensemble_*.py` 系列的历史结论**不可采信**（评估集是 `Dataset_Summary` 前 5000 个
> 原始样本，未划分，且 `ensemble_final.py` 有硬编码保存 bug）。请改用
> `_tools/eval_ensemble.py` 与 `_tools/select_ensemble.py`。

## 技术背景

### CW 方程
近地圆轨道相对运动线性化模型：

$$\ddot{x} - 2n\dot{y} - 3n^2x = 0,\quad \ddot{y} + 2n\dot{x} = 0,\quad \ddot{z} + n^2z = 0$$

有脉冲机动时 $x_{t+1} = \Phi_h x_t + B_{eff}\Delta v_t$，可反解
$\Delta v_t = B_{eff}^{+}(x_{t+1} - \Phi_h x_t)$。本项目据此构造可微 Δv 估计与物理约束。

### 历史性能优化（v7 时代测量）
全向量化重写消除 Python 级循环：

| 模块 | 优化前 | 优化后 | 加速比 |
|------|-------|-------|-------|
| DeltaVEstimator | 1633 ms/batch | 0.6 ms/batch | 2720× |
| PhysicsLoss | 1479 ms/batch | 6.7 ms/batch | 220× |
| 单 epoch | ~27 min | ~12 s | 135× |

## 版本谱系

| 版本 | 内容 | 结果 |
|------|------|------|
| v1–v3 | Transformer / Conv1D 系列 | 无法学习（loss 0.24–0.42） |
| v4 | LSTM Encoder-Decoder + Delta 残差 | 验证 loss 0.017，成功 |
| v5 | Transformer-PI | 劣化 27%，td 5.4947 km |
| **v6** | **多场景汇总数据 + 断点恢复 + 学术可视化**（架构 = v3 PI-LSTM） | **当前主线** |
| v7 | CW baseline 残差学习 / 变容量 | v7a 仅 5.2040 km |
| v8 | 知识蒸馏 | 未产出可用权重 |
| v9 | 输入噪声增强 | loss 爆炸早停 |
| v10 | hidden=512 + 物理一致增强 | 早停 ep1（4.7567 km） |
| v11 | 3-seed bagging | 早停 ep1（4.7243 km） |
| **v12** | **物理计算修正（步长/量纲/语义/尺度）+ 位置损失对齐 + 容量可配置 + 集成** | **当前最优：位置 RMSE 1.2657 km（+7.66%）** |

**结论**：
1. 短时序（10 步）任务中 LSTM 显著优于 Transformer；全向量化在不改变数学结果的前提下
   将训练速度提升近两个数量级。
2. **v7~v11 失败有两个独立原因**：① 末端损失量级失衡导致 loss 爆炸（**已由改进 C 修复**）；
   ② **CW 锚定路线本身不可行** —— 追踪星持续机动，CW 外推基线位置 RMSE 达 120.87 km，
   比持久基线（17.66 km）还差 6.8 倍。**故 v7 残差学习不应重试**。
3. **v9/v11（增强与 bagging）在 loss 稳定后可重试**，但需注意：经度量核验模型**几乎无过拟合**
   （同度量下 val/train = 1.088×），正则化类方法（含噪声增强）预期收益有限甚至为负
   —— 实测开启增强后位置 RMSE 由 1.3707 恶化到 1.4485 km。
