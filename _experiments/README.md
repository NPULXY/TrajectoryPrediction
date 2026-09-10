# _experiments —— 消融实验脚本（历史归档）

> 状态：**已终止的实验**。全部未产出超越 v6 主线的结果，保留仅供追溯。

## 脚本清单与结论

| 脚本 | 设计思路 | 实测结论 |
|------|---------|---------|
| `train_v7.py` | CW 演化 baseline + 残差学习（模型只学 `X_next − CW_baseline`） | 未产出可用权重 |
| `train_v7_cw.py` | v7 残差学习修复版（`model.forward` 接受 baseline 参数） | 未产出可用权重 |
| `train_v7_variants.py` | v7a(hidden=256, 3层) / v7b(hidden=320, 4层) 变容量 | v7a 最好仅 td=5.2040 km（明显劣于 v6 的 4.29） |
| `train_v8_distill.py` | 知识蒸馏：v6 teacher → v7 CW student（T=2.0） | 未产出可用权重 |
| `train_v9_augment.py` | 输入加高斯噪声（std 随 epoch 衰减） | loss 爆炸早停 |
| `train_v10_big_augment.py` | hidden 512 + 物理一致增强（旋转/速度缩放/时间扭曲） | 早停在 ep1，td=4.7567 km |
| `train_v11_bagging.py` | 3 seed bagging 集成 | 早停在 ep1，td=4.7243 km |

**共同失败原因（2026-09-10 已定位）**：`multi_step_terminal_loss` 与末步 terminal loss 在物理量纲直接求 3D 距离（量级 ~4.0 km），与 `l_pred`（~0.008，标准化空间）量级差约 500×，乘以 λ_t=2.0 后以约 1500× 优势主导梯度 → 一旦引入数据增强/多 seed，样本差异被放大即触发 loss 爆炸（0.02 → 9.9）。

**该问题已在主线修复**（改进 C：引入 `TERMINAL_REF_DIST=4.5` 归一化，见 `config.py`），修复后完整 80 epoch 从零训练不再发散。**因此这些实验在原理上可以基于当前主线重试** —— 这是它们唯一的剩余价值。

## ⚠️ 运行前必读：路径引用已失效

这些脚本中的硬编码路径指向**已移动的目录**，直接运行会报 `FileNotFoundError`：

| 脚本中的原路径 | 现已移至 |
|---------------|---------|
| `backup_v3/best_model_v3_ep58_td4.30.pth` | `_archive/weights/backup_v3/…` |
| `backup_v7/best_model_v7a.pth` | `_archive/weights/backup_v7/…` |
| `backup_v7/train_v7a_log.txt` | `_archive/weights/backup_v7/…` |
| `backup_v11/best_model_seed{seed}.pth` | `_archive/weights/backup_v11/…` |

如需重跑，请先全局替换 `backup_v` → `_archive/weights/backup_v`。

## 运行方式

从**项目根目录**运行（脚本内 `sys.path.insert(0, '.')` 依赖 cwd 为根目录）：

```bash
cd TrajectoryPrediction
python _experiments/train_v9_augment.py
```

## 相关

- 权重横向对比与集成核验请用 `_tools/compare_best.py`、`_tools/verify_ensemble.py`
- v6 体系备份见 `_archive/v6_core/`
