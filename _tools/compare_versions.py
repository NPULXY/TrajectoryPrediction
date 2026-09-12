"""对比 v1/v2/v3/v6 在相同 5000 样本上的末距指标"""
import sys, torch, numpy as np
sys.path.insert(0, '.')
from utils.data_loader import parse_csv, FeatureScaler
from models.model import create_model
from config import DEVICE
import config

# 强制 v3 架构
config.USE_TRANSFORMER = False

print('加载测试集（前 5000 样本）...')
XN, XN_masks = parse_csv(f'{config.DATA_DIR}/X_now.csv')
XN2, _ = parse_csv(f'{config.DATA_DIR}/X_next.csv')
X_raw = np.stack(XN[:5000], axis=0)
Y_raw = np.stack(XN2[:5000], axis=0)
masks = np.stack(XN_masks[:5000], axis=0)

scaler = FeatureScaler(); scaler.load('output/scaler.pkl')
X_norm = scaler.transform(X_raw)
X_t = torch.from_numpy(X_norm).float().to(DEVICE)
masks_t = torch.from_numpy(masks).bool().to(DEVICE)

versions = {
    'v3 (td=4.30)': 'backup_v3/best_model_v3_ep58_td4.30.pth',
    'v6 (td=4.29)': 'output/best_model.pth',
}

header = '{:<18} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}'.format(
    '版本', 'td_mean', 'td_med', 'td_min', 'td_max', '<1km%', '<3km%')
print(header)
print('-' * 80)
results = {}
for name, path in versions.items():
    model = create_model(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    # 用 strict=False 兼容不同架构的 state_dict
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    with torch.no_grad():
        pred, _ = model(X_t, return_dv=True, mask=masks_t)
    preds_raw = scaler.inverse_transform(pred.cpu().numpy())

    dists = []
    for i in range(5000):
        n = int(masks[i].sum()) // 6
        max_d = 0
        for a in range(n):
            base = a * 6
            d = np.linalg.norm(preds_raw[i, -1, base:base+3] - Y_raw[i, -1, base:base+3])
            if d > max_d:
                max_d = d
        dists.append(max_d)
    dists = np.array(dists)
    results[name] = dists
    line = '{:<18} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>8.4f} {:>8.4f}'.format(
        name, dists.mean(), np.median(dists), dists.min(), dists.max(),
        np.mean(dists < 1), np.mean(dists < 3))
    print(line)

# 排名
print()
print('=== 按 mean td 排名 ===')
for i, (name, dists) in enumerate(sorted(results.items(), key=lambda x: x[1].mean())):
    print(f'  #{i+1}: {name} mean={dists.mean():.4f} km')