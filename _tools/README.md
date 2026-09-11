# _tools —— 诊断、对比与集成工具

> 所有脚本均为**只读诊断/分析**性质，不修改训练产物。
> 统一运行方式（从项目根目录）：
> ```bash
> cd TrajectoryPrediction
> python _tools/<script>.py
> ```
> 环境：conda `torch`（Python 3.9.18 + PyTorch 2.1.0+cu118）

## 有效工具（当前维护）

| 脚本 | 用途 | 说明 |
|------|------|------|
| `verify_timestep.py` | **时间步长核验** | 用 CW 转移矩阵在不同 dt 假设下比较序列内相邻步残差，判定数据真实采样间隔。**2026-09-10 据此确定步长为 60 s**（原代码误用 1 s），并量化跨序列边界的位置/速度跳变 |
| `diagnose_accuracy.py` | **精度瓶颈诊断** | 回答四个问题：① 与持久基线/CW 基线的差距 ② 误差随预测步长的增长 ③ 不同 N 下的误差 ④ 误差分布与初距相关性。**据此排除 CW 锚定路线**（CW 基线 120.87 km，比持久基线还差 6.8 倍） |
| `verify_metrics_consistency.py` | **度量口径核验** | 核验 `train.py` 训练端（Huber+outlier 加权）与验证端（纯 MSE）是否同口径。**据此证伪"过拟合"误判**：表面 2.4× 差距实为度量不一致，同度量下真实差距仅 1.088× |
| `eval_ensemble.py` | **集成评估** | 单模型 / 两两组合 / 全成员等权集成对比，测试集统一口径。**按检查点权重形状自动推断架构**（支持 hidden 384 与 512 混合成员） |
| `select_ensemble.py` | **集成子集选择（严谨版）** | 枚举全部 2~k 元子集，在**验证集**上按位置 RMSE 选组合，再在**测试集**上报告 —— 避免测试集过拟合。结果落盘 `output/ensemble_config.json` |
| `diagnose_baseline.py` | 基线一致性诊断 | ① config 与 checkpoint 的架构一致性（判断续训是否会崩）② `best_model.pth` strict 加载核验 ③ 量化 `predict.py` 缺失 mask 造成的推理偏差 |
| `compare_best.py` | 候选权重横向对比 | 在同一 seed=42 验证集（32,646 样本）上统一口径评估所有候选；自动跳过架构不兼容者 |
| `verify_ensemble.py` | 集成策略核验 | 单模型 vs 两两/三组加权网格 + 中位数集成，判断集成是否真有增益 |
| `inspect_checkpoints.py` | 检查点元数据盘点 | 列出所有 `.pth` 的 epoch / val_loss / val_terminal_dist / 参数量 / 架构 |

**标准口径**（务必遵守）：`seed=42` 划分，`td` 对齐 `train.py::validate()`——每样本取各有效目标末步 3D 距离的**最大值**，再对样本求均值。偏离此口径的数值不可跨表比较。

**各脚本使用的数据集**（避免混用）：

| 脚本 | 验证集 | 测试集 | 用途 |
|------|:------:|:------:|------|
| `compare_best.py` / `verify_ensemble.py` | ✅ | — | 候选筛选 |
| `eval_ensemble.py` | — | ✅ | 集成方案评估 |
| `select_ensemble.py` | ✅（选组合） | ✅（报告） | 子集选择的严谨流程 |

> 选组合必须在**验证集**上做、在**测试集**上报告。在测试集上枚举子集并选优会产生选择偏差。

## ⚠️ 已废弃（历史结论不可采信）

以下脚本的原因是：评估集是 `Dataset_Summary` 的**前 5000 个原始样本**（`XN[:5000]`，未做 train/val/test 划分），与验证集不可比；且 `ensemble_final.py` 存在硬编码 bug（无论哪个策略最优都固定保存 `weights:[0.4,0.6]` 元数据，导致其记录的 4.0072 km 与同脚本另一份硬编码参照值 4.0048 自相矛盾）。

| 脚本 | 废弃原因 |
|------|---------|
| `ensemble_v7.py` | 前 5000 原始样本评估；路径引用 `backup_*` 已失效 |
| `ensemble_final.py` | 同上 + 硬编码保存 bug |
| `final_ensemble_v11.py` | 同上 |
| `compare_versions.py` | 路径引用 `backup_*` 已失效 |
| `final_summary.py` | 旧汇总脚本，结论已并入本文档 |

**替代方案**：请改用 `compare_best.py` 与 `verify_ensemble.py`。

## 备查结论

### 2026-09-11（当前最优，测试集 32,647 样本）

| 方案 | 位置 RMSE | 位置 MAE | 末端距离 |
|------|----------:|---------:|---------:|
| 原始基线（384 容量） | 1.3707 km | 0.9423 km | 4.2780 km |
| 最优单模型 `best_model_f1`（512 + 位置损失） | 1.2958 km | 0.8620 km | 4.2461 km |
| **集成 `p1 + f1 + f2`（推荐）** | **1.2657 km** | **0.8434 km** | **4.1659 km** |
| **增益** | **+7.66%** | **+10.50%** | **+2.62%** |

配置：`output/ensemble_config.json`；启用：
`TP_ENSEMBLE="output/best_model_p1.pth,output/best_model_f1.pth,output/best_model_f2.pth"`
选择流程：验证集选出（val RMSE 1.2726）→ 测试集报告（1.2657），无过拟合迹象。

### 2026-09-10（历史，步长修正前的旧模型，不宜引用）

> ⚠️ 以下为 `CW_DT_H=1.0` 时代的结论，其模型存在物理计算缺陷（详见 `CLAUDE.md` 修复表），
> 保留仅作追溯。

| 策略 | td_max | 位置 RMSE |
|------|-------:|----------:|
| v6 与 v6o 等权集成 | 4.2400 km | 1.3100 km |
| v6o 单独 | 4.2936 km | 1.3478 km |
| v6 单独 | 4.3171 km | 1.3227 km |
