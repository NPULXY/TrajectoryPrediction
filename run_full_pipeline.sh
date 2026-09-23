#!/usr/bin/env bash
# =============================================================================
# 轨迹预测全流程重跑脚本：训练 → 评估 → 预测 → 论文图件
# 模型：3 成员等权集成（PI-LSTM-384 + 2×PI-LSTM-512，均带位置损失 λ_pos=0.5）
# 用法：从 TrajectoryPrediction/ 目录执行  bash run_full_pipeline.sh
# 说明：新标签 + 固定种子，不覆盖既有 best_model*.pth；预测写入独立文件，
#       不覆盖下游 IntentRecognition_V2 在用的 Dataset/X_pred.csv。
# 环境：Conda torch（PyTorch 2.1.0+cu118），GPU RTX 4060
# =============================================================================
set -e
PY="C:/Users/Hasee/anaconda3/envs/torch/python.exe"
cd "$(dirname "$0")"

# ── 可调参数 ────────────────────────────────────────────────────────────────
TAG_A=rp1; SEED_A=101; HIDDEN_A=384      # 成员 A：384 维
TAG_B=rf1; SEED_B=102; HIDDEN_B=512      # 成员 B：512 维
TAG_C=rf2; SEED_C=103; HIDDEN_C=512      # 成员 C：512 维（不同种子）
POS_W=0.5                                # 位置损失最终权重 λ_pos
PRED_OUT=../Dataset/X_pred_ensemble.csv  # 预测输出（独立文件，勿指向 X_pred.csv）
FIG_DIR=output/paper_figures             # 论文图件输出目录

# ── 1. 训练（单成员约 30~60 min；如需跳过重训，注释本段即可）─────────────────
TP_RUN_TAG=$TAG_A TP_SEED=$SEED_A TP_POS_W=$POS_W TP_HIDDEN=$HIDDEN_A "$PY" train.py
TP_RUN_TAG=$TAG_B TP_SEED=$SEED_B TP_POS_W=$POS_W TP_HIDDEN=$HIDDEN_B "$PY" train.py
TP_RUN_TAG=$TAG_C TP_SEED=$SEED_C TP_POS_W=$POS_W TP_HIDDEN=$HIDDEN_C "$PY" train.py

# ── 2. 评估（测试集 32,647 样本统一口径：单模型 vs 集成）─────────────────────
"$PY" _tools/eval_ensemble.py \
  output/best_model_$TAG_A.pth output/best_model_$TAG_B.pth output/best_model_$TAG_C.pth

# ── 3. 预测（集成推理 + 最佳样本可视化；best_predictions/ 若已存在大量历史图，
#         先 mv 移走再运行，避免批量删除触发 safe-delete 阻断）──────────────────
TP_ENSEMBLE="output/best_model_$TAG_A.pth,output/best_model_$TAG_B.pth,output/best_model_$TAG_C.pth" \
  "$PY" predict.py --output "$PRED_OUT" --visualize --top-k 30

# ── 4. 论文图件（600dpi PNG + 可编辑 SVG + .mat 数据）────────────────────────
"$PY" plot_paper_figures.py \
  --members output/best_model_$TAG_A.pth output/best_model_$TAG_B.pth output/best_model_$TAG_C.pth \
  --member-names "PI-LSTM-384" "PI-LSTM-512-a" "PI-LSTM-512-b" \
  --logs output/train_log_$TAG_A.txt output/train_log_$TAG_B.txt output/train_log_$TAG_C.txt \
  --out "$FIG_DIR"

echo "全流程完成：图件见 $FIG_DIR，预测见 $PRED_OUT"
