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
| `diagnose_baseline.py` | 基线一致性诊断 | ① config 与 checkpoint 的架构一致性（判断续训是否会崩）② `best_model.pth` strict 加载核验 ③ 量化 `predict.py` 缺失 mask 造成的推理偏差 |
| `compare_best.py` | 候选权重横向对比 | 在同一 seed=42 验证集（32,646 样本）上统一口径评估所有候选；自动跳过架构不兼容者 |
| `verify_ensemble.py` | 集成策略核验 | 单模型 vs 两两/三组加权网格 + 中位数集成，判断集成是否真有增益 |
| `inspect_checkpoints.py` | 检查点元数据盘点 | 列出所有 `.pth` 的 epoch / val_loss / val_terminal_dist / 参数量 / 架构 |
| `verify_timestep.py` | **时间步长核验** | 用 CW 转移矩阵在不同 dt 假设下比较序列内相邻步残差，判定数据真实采样间隔。**2026-09-10 据此确定步长为 60 s**（原代码误用 1 s），并量化跨序列边界的位置/速度跳变 |

**标准口径**（务必遵守）：`seed=42` 验证集，`td` 对齐 `train.py::validate()`——每样本取各有效目标末步 3D 距离的**最大值**，再对样本求均值。偏离此口径的数值不可跨表比较。

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

## 备查结论（2026-09-10 实测）

在统一验证集上：

| 策略 | td_max | 位置 RMSE |
|------|-------:|----------:|
| v6 与 v6o 等权集成 | **4.2400 km** | **1.3100 km** |
| v6o 单独（最优单模型） | 4.2936 km | 1.3478 km |
| v6 单独（当前挂载） | 4.3171 km | 1.3227 km |

即**集成优于任何单模型约 1.25%**。
