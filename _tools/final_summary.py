"""
重建完整训练曲线 v3+v4 + 历史基线
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('_rendered', exist_ok=True)

# v3 epoch 27-58 数据（从 train_log 之前提取）
v3_data = [
    (27, 5.157, 0.019190), (28, 5.061, 0.016008), (29, 5.010, 0.016084),
    (30, 4.995, 0.015828), (31, 4.939, 0.016122), (32, 4.911, 0.016221),
    (33, 4.879, 0.016115), (34, 4.887, 0.016190), (35, 4.812, 0.016137),
    (36, 4.785, 0.016027), (37, 4.696, 0.015955), (38, 4.797, 0.016268),
    (39, 4.733, 0.016121), (40, 4.792, 0.016187), (41, 4.605, 0.015787),
    (42, 4.533, 0.015768), (43, 4.578, 0.016144), (44, 4.577, 0.016048),
    (45, 4.523, 0.015927), (46, 4.512, 0.015803), (47, 4.494, 0.016021),
    (48, 4.461, 0.015727), (49, 4.447, 0.015658), (50, 4.422, 0.015570),
    (51, 4.404, 0.015538), (52, 4.382, 0.015470), (53, 4.373, 0.015415),
    (54, 4.347, 0.015398), (55, 4.321, 0.015316), (56, 4.328, 0.015297),
    (57, 4.316, 0.015302), (58, 4.302, 0.015268),
]

v3_epochs = [d[0] for d in v3_data]
v3_td = [d[1] for d in v3_data]
v3_val_pred = [d[2] for d in v3_data]

# v4 数据（从当前 train_log.txt 提取）
import re
with open('output/train_log.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
v4_epochs, v4_td, v4_val_pred = [], [], []
for line in lines:
    m = re.match(r'Epoch\s+(\d+)/\d+.*td=([\d.]+)km.*Val:\s*([\d.]+)', line)
    if m:
        v4_epochs.append(int(m.group(1)))
        v4_td.append(float(m.group(2)))
        v4_val_pred.append(float(m.group(3)))

# 画图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：td 曲线
axes[0].plot(v3_epochs, v3_td, 'b-', linewidth=2.5, marker='o', markersize=4, label='v3 PI-LSTM')
axes[0].plot(v4_epochs, v4_td, 'g--', linewidth=2, marker='s', markersize=4, label='v4 PI-LSTM (larger batch)')
axes[0].axhline(y=5.68, color='r', linestyle=':', linewidth=1.5, label='v1 best (5.68 km)')
axes[0].axhline(y=4.41, color='orange', linestyle=':', linewidth=1.5, label='v2 best (4.41 km)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('末距 td (km)')
axes[0].set_title('末距训练曲线 (v3+v4)')
axes[0].legend(loc='upper right', fontsize=10)
axes[0].grid(True, alpha=0.3)

# 右图：val_pred 曲线
axes[1].plot(v3_epochs, v3_val_pred, 'b-', linewidth=2.5, marker='o', markersize=4, label='v3 val_pred')
axes[1].plot(v4_epochs, v4_val_pred, 'g--', linewidth=2, marker='s', markersize=4, label='v4 val_pred')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Val Pred Loss')
axes[1].set_title('验证集预测损失 (v3+v4)')
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('_rendered/final_training_summary.png', dpi=130, bbox_inches='tight')
plt.close()
print('saved _rendered/final_training_summary.png')