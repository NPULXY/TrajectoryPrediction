"""
数据加载模块：解析 CSV、padding、z-score 标准化、数据集划分。
"""

import ast
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR, MAX_N, MAX_DIM, INPUT_STEPS, OUTPUT_STEPS,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED, BATCH_SIZE,
    SCALER_SAVE_PATH, POS_INDICES, VEL_INDICES,
)


class FeatureScaler:
    """逐特征维度的 z-score 标准化器，仅对有效（非 padding）数据计算统计量。"""

    def __init__(self):
        self.mean = None     # (max_dim,)
        self.std = None      # (max_dim,)

    def fit(self, X, masks):
        """
        在训练集上计算每个特征维度的均值和标准差。

        Args:
            X: numpy 数组，形状 (N, 10, max_dim)
            masks: numpy 布尔数组，形状 (N, max_dim)
        """
        N, T, D = X.shape
        self.mean = np.zeros(D)
        self.std = np.ones(D)   # 对无有效数据的维度，std 保持 1 避免除零

        for d in range(D):
            # 收集该维度在所有样本、所有时间步中的有效值
            valid_mask = masks[:, d]                 # (N,)
            if valid_mask.sum() == 0:
                continue
            vals = X[valid_mask, :, d].flatten()    # 该维度全部有效值
            self.mean[d] = vals.mean()
            std_val = vals.std()
            if std_val > 1e-8:
                self.std[d] = std_val

    def transform(self, X):
        """对数据做 z-score 标准化。"""
        eps = 1e-8
        return (X - self.mean) / (self.std + eps)

    def inverse_transform(self, X):
        """将标准化后的数据还原为原始量纲。"""
        eps = 1e-8
        return X * (self.std + eps) + self.mean

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean, "std": self.std}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.mean = data["mean"]
        self.std = data["std"]


def parse_csv(filepath):
    """
    读取 CSV 文件并解析嵌套列表字符串。

    Returns:
        samples: list of numpy arrays, 每个形状 (10, N*6)
        masks:   list of numpy arrays, 每个形状 (max_dim,) bool
    """
    df = pd.read_csv(filepath)
    col_name = df.columns[0]
    samples = []
    masks = []

    for val in df[col_name]:
        nested = ast.literal_eval(val)               # list of 10 lists
        arr = np.array(nested, dtype=np.float32)     # (10, N*6)
        n_features = arr.shape[1]

        # Padding 至 max_dim
        if n_features < MAX_DIM:
            padded = np.zeros((INPUT_STEPS, MAX_DIM), dtype=np.float32)
            padded[:, :n_features] = arr
            mask = np.zeros(MAX_DIM, dtype=bool)
            mask[:n_features] = True
        else:
            padded = arr
            mask = np.ones(MAX_DIM, dtype=bool)

        samples.append(padded)
        masks.append(mask)

    return samples, masks


def load_and_split(data_dir=DATA_DIR, seed=RANDOM_SEED):
    """
    加载 X_now.csv 和 X_next.csv，同步打乱并划分训练/验证/测试集。

    Returns:
        train_X, val_X, test_X: 各为 (n, 10, max_dim) numpy 数组
        train_Y, val_Y, test_Y: 同上
        train_masks, val_masks, test_masks: 各为 (n, max_dim) bool 数组
        scaler: 在训练集上拟合好的 FeatureScaler
    """
    x_path = f"{data_dir}/X_now.csv"
    y_path = f"{data_dir}/X_next.csv"

    print(f"正在加载数据: {x_path}")
    X_list, X_masks = parse_csv(x_path)
    print(f"正在加载数据: {y_path}")
    Y_list, Y_masks = parse_csv(y_path)

    X = np.stack(X_list, axis=0)      # (23787, 10, max_dim)
    Y = np.stack(Y_list, axis=0)
    masks = np.stack(X_masks, axis=0)  # (23787, max_dim)

    print(f"数据形状: X={X.shape}, Y={Y.shape}, masks={masks.shape}")

    # 按 N 统计样本数
    for n in range(1, MAX_N + 1):
        count = (masks[:, n * 6 - 1]).sum()  # 该 N 的最后一位是否为 True
        print(f"  N={n}: {count} 个样本 ({count / len(X) * 100:.1f}%)")

    # 同步打乱索引
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    X = X[indices]
    Y = Y[indices]
    masks = masks[indices]

    # 划分数据集: 70% train, 15% val, 15% test
    val_test_ratio = VAL_RATIO + TEST_RATIO
    train_X, temp_X, train_Y, temp_Y, train_masks, temp_masks = train_test_split(
        X, Y, masks, test_size=val_test_ratio, random_state=seed
    )
    val_ratio_in_temp = VAL_RATIO / val_test_ratio
    test_ratio_in_temp = TEST_RATIO / val_test_ratio
    val_X, test_X, val_Y, test_Y, val_masks, test_masks = train_test_split(
        temp_X, temp_Y, temp_masks,
        test_size=test_ratio_in_temp / (val_ratio_in_temp + test_ratio_in_temp),
        random_state=seed
    )

    print(f"\n数据集划分: 训练={len(train_X)}, 验证={len(val_X)}, 测试={len(test_X)}")

    # 在训练集上拟合 scaler
    scaler = FeatureScaler()
    scaler.fit(train_X, train_masks)

    # 对所有数据集做标准化
    train_X = scaler.transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)
    train_Y = scaler.transform(train_Y)
    val_Y = scaler.transform(val_Y)
    test_Y = scaler.transform(test_Y)

    return (train_X, val_X, test_X,
            train_Y, val_Y, test_Y,
            train_masks, val_masks, test_masks,
            scaler)


class TrajectoryDataset(Dataset):
    """轨迹预测 PyTorch Dataset。"""

    def __init__(self, X, Y, masks):
        """
        Args:
            X: numpy (n, 10, max_dim)
            Y: numpy (n, 10, max_dim)
            masks: numpy (n, max_dim) bool
        """
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.masks = torch.from_numpy(masks).bool()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.masks[idx]


def create_dataloaders(train_X, val_X, test_X,
                       train_Y, val_Y, test_Y,
                       train_masks, val_masks, test_masks,
                       batch_size=BATCH_SIZE):
    """创建训练/验证/测试 DataLoader。"""
    train_dataset = TrajectoryDataset(train_X, train_Y, train_masks)
    val_dataset = TrajectoryDataset(val_X, val_Y, val_masks)
    test_dataset = TrajectoryDataset(test_X, test_Y, test_masks)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def masked_mse_loss(pred, target, mask):
    """
    计算带 mask 的 MSE 损失。

    Args:
        pred:   (B, 10, max_dim)
        target: (B, 10, max_dim)
        mask:   (B, max_dim) bool

    Returns:
        标量损失（仅对 mask 为 True 的元素求均值）
    """
    mask_expanded = mask.unsqueeze(1).expand_as(pred)   # (B, 10, max_dim)
    diff = (pred - target) ** 2
    return diff[mask_expanded].mean()


def compute_metrics(pred, target, mask):
    """
    计算整体及分项（位置/速度）的 MSE、MAE、RMSE。

    Args:
        pred, target, mask: 均为 torch.Tensor，未归一化（原始量纲）

    Returns:
        dict: 各项指标
    """
    mask_expanded = mask.unsqueeze(1).expand_as(pred)

    def safe_mean(vals):
        return vals[mask_expanded].mean().item()

    mse = safe_mean((pred - target) ** 2)
    mae = safe_mean(torch.abs(pred - target))
    rmse = np.sqrt(mse)

    def component_metric(idx_list):
        p = pred[:, :, idx_list]
        t = target[:, :, idx_list]
        m = mask_expanded[:, :, idx_list]
        diff = (p - t) ** 2
        return diff[m].mean().item()

    pos_mse = component_metric(POS_INDICES)
    vel_mse = component_metric(VEL_INDICES)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MSE_pos": pos_mse,
        "RMSE_pos": np.sqrt(pos_mse),
        "MAE_pos": component_metric_mae(POS_INDICES, pred, target, mask_expanded),
        "MSE_vel": vel_mse,
        "RMSE_vel": np.sqrt(vel_mse),
        "MAE_vel": component_metric_mae(VEL_INDICES, pred, target, mask_expanded),
    }


def component_metric_mae(idx_list, pred, target, mask_expanded):
    p = pred[:, :, idx_list]
    t = target[:, :, idx_list]
    m = mask_expanded[:, :, idx_list]
    return torch.abs(p - t)[m].mean().item()


if __name__ == "__main__":
    # 自检：加载数据并打印基本信息
    (train_X, val_X, test_X,
     train_Y, val_Y, test_Y,
     train_masks, val_masks, test_masks,
     scaler) = load_and_split()

    print(f"\n训练集: {train_X.shape}, 验证集: {val_X.shape}, 测试集: {test_X.shape}")
    print(f"Scaler mean 前 6 维: {scaler.mean[:6]}")
    print(f"Scaler std 前 6 维:  {scaler.std[:6]}")

    loaders = create_dataloaders(
        train_X, val_X, test_X,
        train_Y, val_Y, test_Y,
        train_masks, val_masks, test_masks
    )
    for x, y, m in loaders[0]:
        print(f"一个 batch: x={x.shape}, y={y.shape}, mask={m.shape}")
        print(f"mask 示例（第一条）: {m[0]}")
        break
