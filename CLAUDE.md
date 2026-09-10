# CLAUDE.md

> GitHub: <https://github.com/NPULXY/TrajectoryPrediction>
> 航天器相对运动轨迹预测 —— PI-LSTM 序列到序列模型

## 项目定位

给定追踪航天器相对目标的**过去 10 步**状态（位置 km + 速度 km/s，LVLH 坐标系），预测**未来 10 步**。
数据集覆盖交会/阻扰/探测/潜伏/混合五场景，共 217,642 样本。

**当前主线 = v6 体系**：
- 架构：**PI-LSTM v3**（`models/pinn_lstm.py`，`PhysicsInformedTrajectoryLSTM`），可训练参数 4,915,176
- 配置：`config.py` 中 `PHYSICS_ENABLED=True`、`USE_TRANSFORMER=False`
- 最新最优单模型：`output/best_model.pth`（epoch 80，val_terminal_dist **4.3171 km**）
- 集成最优：`output/best_model.pth` 与 `_archive/v6_core/weights/best_model_v6o_ep80.pth` 等权平均
  （td 4.2400 km，优于最优单模型 1.25%）

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

## ⚠️ 已知陷阱（务必先读）

1. **续训可能直接崩溃**：`train.py` 加载检查点处**无 try/except**。若 `output/latest_checkpoint.pth`
   残留了**其他架构**的权重（例如 v5 Transformer），`load_state_dict(strict=True)` 会抛 `RuntimeError`
   并中断。**切换架构/权重来源后必须先清理该文件**；可用 `_tools/diagnose_baseline.py` 预检。
2. **`predict.py` 推理未传 mask**（`model(batch, return_dv=True)`，缺 `mask=`）：
   全局 Δv 被除以 `max_N=4` 而非实际 N，导致 N=2/3 样本的条件向量与训练口径不一致。
   实测位置偏差：N=2 平均 0.271 km / 最大 2.422 km；N=3 平均 0.171 km；N=4 无偏差。
   **`evaluate.py` 传了 mask，是正确的**——改动时保持一致。
3. **`hidden_size=384` / `num_layers=4` 硬编码**在 `pinn_lstm.py`，`config.py` 的
   `D_MODEL` / `NUM_LSTM_LAYERS` **对 PI-LSTM 不生效**。
4. **`best_val_loss` 恒为 inf**（`train()` 中初始化后从未更新）→ `latest_checkpoint.pth` 的
   `val_loss` 字段不可用，请改看 `val_terminal_dist`。
5. **scheduler 跨 epoch 跳变**：`lr_lambda` 依赖 `total_epochs`，改动 `EPOCHS` 后续训会导致
   学习率相位突变（ep81 从 1e-6 跳回 6.6e-4）；`RESUME_FIXED_LR` 补丁会被 `scheduler.step()` 覆盖。
6. **`dv_all` 实际为 18 步**（9 CW 逆推 + 9 速度差分），代码注释多处误写为 19。
7. **两个 `create_model` 重复**：`models/model.py` 与 `models/pinn_lstm.py` 各有一个；
   train/evaluate/predict 均从 `models.model` 导入，`pinn_lstm` 那个是死代码。
8. **末端距离损失被计算两次**（`multi_step_terminal_loss` 与末步 loss 叠加），实际总权重 ≈ 1.5 λ_t。

## 数据硬约束

- `X_now.csv` / `X_next.csv` / `Y.csv` **行序严格对应**，且汇总版已在合并时**全局打乱**——
  **绝不可单独打乱任一文件**
- 每行是**单列嵌套列表字符串** `[[step1], …, [step10]]`，**必须 `json.loads` 逐行解析**，
  不能用 `pd.read_csv` 默认逗号分隔
- N ∈ {2,3,4} → 每步 12/18/24 维，代码 padding 至 `MAX_DIM=24` 并生成 mask
- `X_next` 可能混入 **CW 外推补齐点（步长 60 s）**，不全是 1 s 步长的真实仿真步
  → 这是 `L_cw` 被关闭（权重 1e-7）的原因：数据不满足 `Φ_h(1s)` 单步递推假设
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

$$\mathcal{L} = w_s\mathcal{L}_{Huber} + \tfrac{1}{2}\lambda_t \mathcal{L}_{term}^{multi} + \lambda_t \mathcal{L}_{term}^{last} + \lambda_m \mathcal{L}_{align} + \lambda_p \mathcal{L}_{cw} + \lambda_b \mathcal{L}_{bound}$$

| 分量 | 终值权重 | warmup |
|------|---------|--------|
| `L_Huber`（逐样本，末端>10 km 降权至 0.1、5–10 km 至 0.3） | 1.0 | — |
| `L_term^multi`（t=3/6/9，权重 0.3/0.5/1.0，÷`TERMINAL_REF_DIST`=4.5） | 0.01→2.0 | 15 ep |
| `L_term^last`（末步，同样归一化） | 同上 | 15 ep |
| `L_align`（模型 Δv 对齐 CW 逆推 Δv，PI-LSTM 核心） | 0.001→0.05 | 20 ep |
| `L_cw`（CW 单步残差，分维度归一化） | 0→**1e-7**（关闭） | 20 ep |
| `L_bound`（Δv 越界 3 m/s 惩罚） | 0.0005→0.005 | 15 ep |

前 `PRED_WARMUP_EPOCHS=3` 轮仅用 `L_Huber`。

**改进 C（2026-09-10）**：末端损失原本在物理量纲直接算 3D 距离（~4.0 km），与标准化空间
`L_Huber`（~0.008）差约 500×，λ_t=2.0 后以 ~1500× 主导梯度 → 数据增强时 loss 爆炸。
引入 `TERMINAL_REF_DIST=4.5` 归一化后量级降至 ~1.0，训练稳定（train loss 12.6 → 2.81）。

## 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 / 学习率 | AdamW，1e-3 → 1e-6（Cosine warmup 3 ep + annealing） |
| BATCH_SIZE / EPOCHS | 384 / 80 |
| EARLY_STOP_PATIENCE | 20（判据 `val_terminal_dist_mean`，**非** val loss） |
| DROPOUT / 梯度裁剪 | 0.15 / max_norm 1.0 |
| 数据集划分 | 70/15/15，seed=42 → 152,349 / 32,646 / 32,647 |
| RESUME_TRAINING | True（优先 `latest_checkpoint.pth`，回退 `best_model.pth`） |

## 目录约定

| 目录 | 用途 |
|------|------|
| `models/`、`utils/` | 模型定义与数据管线（核心代码） |
| `output/` | 训练与评估产物（权重/scaler/日志/图表）——**git 忽略** |
| `_tools/` | 诊断与对比工具（有效清单见 `_tools/README.md`） |
| `_experiments/` | v7~v11 消融脚本归档（脚本内路径引用已失效，见其 README） |
| `_archive/v6_core/` | **v6 体系精简备份**（代码 + 2 个关键权重 + 完整说明） |
| `_archive/weights/` | 历史权重（backup_v2/v3/v6/v7/v10/v11） |
| `_docs/` | 历史提示词存档与参考文献 |
| `_rendered/` | 渲染产物 |

## 评估口径（关键）

**标准口径**：seed=42 验证集（32,646 样本），`td` 对齐 `train.py::validate()` ——
每样本取「各有效目标末步 3D 距离的**最大值**」再对样本求均值。

| 策略 | td_max | 位置 RMSE |
|------|-------:|----------:|
| v6 + v6o 等权集成 | **4.2400 km** | **1.3100 km** |
| v6o 单独（`_archive/weights/backup_v6/best_model_v6_ep80_td4.294.pth`） | 4.2936 km | 1.3478 km |
| v6 单独（`output/best_model.pth`） | 4.3171 km | 1.3227 km |

> ⚠️ `_tools/ensemble_*.py` 系列的历史结论**不可采信**（评估集是 `Dataset_Summary` 前 5000 个
> 原始样本，未划分，且 `ensemble_final.py` 有硬编码保存 bug）。请改用
> `_tools/compare_best.py` 与 `_tools/verify_ensemble.py`。

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

**结论**：短时序（10 步）任务中 LSTM 显著优于 Transformer；v7~v11 的失败均源于末端损失
量级失衡（**已由改进 C 修复**，可基于当前主线重试）；全向量化在不改变数学结果的前提下
将训练速度提升近两个数量级。
