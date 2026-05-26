"""
推理脚本 —— 加载训练好的模型对新样本进行预测并保存结果。
用法: python predict.py [--input INPUT_CSV] [--output OUTPUT_CSV]
"""

import os
import ast
import argparse
import numpy as np
import pandas as pd
import torch

from config import (
    DATA_DIR, OUTPUT_DIR, MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    DEVICE, MAX_DIM, INPUT_STEPS, OUTPUT_STEPS,
)
from utils.data_loader import FeatureScaler, parse_csv
from models.model import create_model


def predict(input_path, output_path, model_path=MODEL_SAVE_PATH, scaler_path=SCALER_SAVE_PATH):
    """
    加载模型和 scaler，对输入 CSV 进行推理，输出预测的 X_next。

    Args:
        input_path:  输入 CSV 文件路径（格式同 X_now.csv）
        output_path: 输出 CSV 文件路径
        model_path:  训练好的模型权重路径
        scaler_path: 保存的 scaler 路径
    """
    # ── 加载 scaler ──
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler 文件不存在: {scaler_path}")
    scaler = FeatureScaler()
    scaler.load(scaler_path)
    print(f"Scaler 已加载: {scaler_path}")

    # ── 加载模型 ──
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    model = create_model(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"模型已加载: {model_path} (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6f})")

    # ── 读取并解析输入数据 ──
    samples, masks = parse_csv(input_path)
    n_samples = len(samples)
    print(f"读取到 {n_samples} 个样本")

    X_raw = np.stack(samples, axis=0)       # (N, 10, 24)
    X_norm = scaler.transform(X_raw)
    X_tensor = torch.from_numpy(X_norm).float().to(DEVICE)

    # ── 分批推理 ──
    batch_size = 256
    all_preds = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = X_tensor[i:i + batch_size]
            pred = model(batch)             # (B, 10, 24)
            all_preds.append(pred.cpu().numpy())

    preds_norm = np.concatenate(all_preds, axis=0)   # (N, 10, 24)
    preds_raw = scaler.inverse_transform(preds_norm)

    # ── 还原为嵌套列表字符串格式 ──
    rows = []
    for i in range(n_samples):
        mask = masks[i]
        n_features = mask.sum()  # 实际特征数 = N*6
        # 只取有效部分
        valid_pred = preds_raw[i, :, :n_features]    # (10, N*6)
        # 格式化为嵌套列表
        inner_lists = []
        for t in range(OUTPUT_STEPS):
            step_vals = valid_pred[t].tolist()
            # 格式化为 %.4f
            formatted = [f"{v:.4f}" for v in step_vals]
            inner_lists.append("[" + ", ".join(formatted) + "]")
        row_str = "[" + ", ".join(inner_lists) + "]"
        rows.append(row_str)

    # ── 保存 ──
    df = pd.DataFrame({"X_next": rows})
    df.to_csv(output_path, index=False)
    print(f"预测结果已保存至: {output_path} ({n_samples} 个样本)")


def main():
    parser = argparse.ArgumentParser(description="轨迹预测推理")
    parser.add_argument("--input", type=str, default=None,
                        help="输入 CSV 文件路径（默认使用测试集第一个样本演示）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 CSV 文件路径（默认保存到 output/predictions.csv）")
    parser.add_argument("--model", type=str, default=MODEL_SAVE_PATH,
                        help="模型权重路径")
    parser.add_argument("--scaler", type=str, default=SCALER_SAVE_PATH,
                        help="Scaler 路径")
    args = parser.parse_args()

    # 默认输入输出路径
    input_path = args.input
    if input_path is None:
        input_path = os.path.join(DATA_DIR, "X_now.csv")
        print(f"未指定输入文件，使用默认: {input_path}")

    output_path = args.output
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "predictions.csv")

    predict(input_path, output_path, args.model, args.scaler)


if __name__ == "__main__":
    main()
