"""
最终集成评估：v3 + v6 + v11 seed42 + v7a + v8
用 5000 样本的客观测试集对比 11 种集成策略
"""
import sys, torch, numpy as np
sys.path.insert(0, '.')
from utils.data_loader import parse_csv, FeatureScaler
from models.model import create_model
from models.pinn_lstm_v7 import create_pinn_lstm_v7
from config import DEVICE
import config

# 强制 v3 架构
config.USE_TRANSFORMER = False

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


def load_v3_or_v6(path, name):
    model = create_model(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def load_v7a(path, hidden=256, layers=3):
    model = create_pinn_lstm_v7(hidden_size=hidden, num_layers=layers, device=DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    state = ckpt['model_state_dict']
    model_state = model.state_dict()
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
    model.load_state_dict(model_state)
    model.eval()
    return model


def predict_norm(model, x, mask):
    with torch.no_grad():
        pred, _ = model(x, return_dv=True, mask=mask)
    return pred.cpu().numpy()


def calc_td(preds_raw, masks, Y_raw):
    dists = []
    for i in range(len(preds_raw)):
        n = int(masks[i].sum()) // 6
        max_d = 0
        for a in range(n):
            base = a * 6
            d = np.linalg.norm(preds_raw[i, -1, base:base+3] - Y_raw[i, -1, base:base+3])
            if d > max_d:
                max_d = d
        dists.append(max_d)
    return np.array(dists)


print('加载模型...')
m_v3 = load_v3_or_v6('backup_v3/best_model_v3_ep58_td4.30.pth', 'v3')
m_v6 = load_v3_or_v6('output/best_model.pth', 'v6')
m_v7a = load_v7a('backup_v7/best_model_v7a.pth', 256, 3)
m_v11 = load_v3_or_v6('backup_v11/best_model_seed42.pth', 'v11_seed42')

print('推理中...')
p_v3 = predict_norm(m_v3, X_t, masks_t)
p_v6 = predict_norm(m_v6, X_t, masks_t)
p_v7a = predict_norm(m_v7a, X_t, masks_t)
p_v11 = predict_norm(m_v11, X_t, masks_t)

# 多策略集成
print()
print('=' * 95)
print(f'{"策略":<35} {"td_mean":>10} {"td_med":>10} {"td_min":>10} {"<1km%":>8} {"<3km%":>8}')
print('-' * 95)
strategies = {
    'v6 单独（当前 best）': p_v6,
    'v3 单独': p_v3,
    'v11 seed42 单独': p_v11,
    'v7a 单独': p_v7a,
    'v3 + v6 简单平均': (p_v3 + p_v6) / 2,
    'v3 + v6 + v11 平均': (p_v3 + p_v6 + p_v11) / 3,
    'v3 + v6 + v11 + v7a 平均': (p_v3 + p_v6 + p_v11 + p_v7a) / 4,
    '0.3 v3 + 0.5 v6 + 0.2 v11': 0.3 * p_v3 + 0.5 * p_v6 + 0.2 * p_v11,
    '0.2 v3 + 0.6 v6 + 0.2 v11': 0.2 * p_v3 + 0.6 * p_v6 + 0.2 * p_v11,
    '0.5 v6 + 0.5 v11': 0.5 * p_v6 + 0.5 * p_v11,
    'v6 + v11 中位数': np.median(np.stack([p_v6, p_v11]), axis=0),
    'v3 + v6 + v11 中位数': np.median(np.stack([p_v3, p_v6, p_v11]), axis=0),
}
results = {}
for name, p_norm in strategies.items():
    preds_raw = scaler.inverse_transform(p_norm)
    td = calc_td(preds_raw, masks, Y_raw)
    results[name] = td
    line = '{:<35} {:>10.4f} {:>10.4f} {:>10.4f} {:>8.4f} {:>8.4f}'.format(
        name, td.mean(), np.median(td), td.min(),
        np.mean(td < 1), np.mean(td < 3))
    print(line)

print()
print('=== 按 mean td 排名 ===')
for i, (name, td) in enumerate(sorted(results.items(), key=lambda x: x[1].mean())):
    print(f'  #{i+1}: {name} mean={td.mean():.4f} km')

best_name, best_td = min(results.items(), key=lambda x: x[1].mean())
print(f'\n最佳: {best_name} -> td={best_td.mean():.4f} km')
print(f'  vs v6 单独: {(best_td.mean() - 4.0048):.4f} km ({"改善" if best_td.mean() < 4.0048 else "无改善"})')

# 保存最佳集成
best_pred_norm = strategies[best_name]
best_pred_raw = scaler.inverse_transform(best_pred_norm)
np.save('output/v11_ensemble_pred.npy', best_pred_raw)
print(f'集成预测已保存到 output/v11_ensemble_pred.npy')