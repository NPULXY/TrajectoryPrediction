"""
v7 集成学习：多个模型推理取平均
- Model A: v3 best (td=4.30)
- Model B: v6 best (td=4.29)
- Model C: 新训练 - hidden 320（v3/v6 之间）+ 不同 seed
- Model D: 新训练 - 同 v3 架构 + 不同 seed
"""
import sys, torch, numpy as np, os
sys.path.insert(0, '.')
from utils.data_loader import parse_csv, FeatureScaler
from models.model import create_model
from config import DEVICE
import config

# 强制 v3 架构
config.USE_TRANSFORMER = False
config.D_MODEL = 384  # 实际不影响 PhysicsInformedTrajectoryLSTM

print('加载测试集（前 5000 样本）...')
XN, XN_masks = parse_csv('Dataset_Summary/X_now.csv')
XN2, _ = parse_csv('Dataset_Summary/X_next.csv')
X_raw = np.stack(XN[:5000], axis=0)
Y_raw = np.stack(XN2[:5000], axis=0)
masks = np.stack(XN_masks[:5000], axis=0)

scaler = FeatureScaler(); scaler.load('output/scaler.pkl')
X_norm = scaler.transform(X_raw)
X_t = torch.from_numpy(X_norm).float().to(DEVICE)
masks_t = torch.from_numpy(masks).bool().to(DEVICE)

# 加载模型
def load_model(path, name):
    model = create_model(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'  {name} 加载完成（{sum(p.numel() for p in model.parameters())/1e6:.2f}M 参数）')
    return model

print('加载模型...')
model_v3 = load_model('backup_v3/best_model_v3_ep58_td4.30.pth', 'v3')
model_v6 = load_model('output/best_model.pth', 'v6')

# 推理
print('推理中...')
preds_norm = []
with torch.no_grad():
    for m in [model_v3, model_v6]:
        pred, _ = m(X_t, return_dv=True, mask=masks_t)
        preds_norm.append(pred.cpu().numpy())

# 集成
ensemble_pred = np.mean(preds_norm, axis=0)
preds_raw = scaler.inverse_transform(ensemble_pred)
preds_raw_v3 = scaler.inverse_transform(preds_norm[0])
preds_raw_v6 = scaler.inverse_transform(preds_norm[1])

# 计算 td
def calc_td(preds_raw, masks, Y_raw, n_samples):
    dists = []
    for i in range(n_samples):
        n = int(masks[i].sum()) // 6
        max_d = 0
        for a in range(n):
            base = a * 6
            d = np.linalg.norm(preds_raw[i, -1, base:base+3] - Y_raw[i, -1, base:base+3])
            if d > max_d:
                max_d = d
        dists.append(max_d)
    return np.array(dists)

N = 5000
td_v3 = calc_td(preds_raw_v3, masks, Y_raw, N)
td_v6 = calc_td(preds_raw_v6, masks, Y_raw, N)
td_ens = calc_td(preds_raw, masks, Y_raw, N)

print()
print(f'{"模型":<20} {"td_mean":>10} {"td_med":>10} {"td_min":>10} {"td_max":>10} {"<1km%":>8} {"<3km%":>8}')
print('-' * 85)
for name, td in [('v3 (单独)', td_v3), ('v6 (单独)', td_v6), ('v3+v6 集成', td_ens)]:
    line = '{:<20} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>8.4f} {:>8.4f}'.format(
        name, td.mean(), np.median(td), td.min(), td.max(),
        np.mean(td < 1), np.mean(td < 3))
    print(line)

# 集成 vs 单独的改善
improvement_v3 = (td_v3.mean() - td_ens.mean()) / td_v3.mean() * 100
improvement_v6 = (td_v6.mean() - td_ens.mean()) / td_v6.mean() * 100
print()
print(f'集成 vs v3 改善: {improvement_v3:+.2f}%')
print(f'集成 vs v6 改善: {improvement_v6:+.2f}%')

# 保存集成预测结果
out_path = 'output/v7_ensemble_pred.npy'
np.save(out_path, preds_raw)
print(f'\n集成预测已保存: {out_path}')