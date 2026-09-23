# -*- coding: utf-8 -*-
"""
论文仿真图生成脚本（轨迹预测部分，PI-LSTM / 集成）。

用途：基于「训练-评估-预测」全流程产物，生成可放入论文（Elsevier 双栏）的仿真图。
所有图为英文标注、Times New Roman + stix 数学字体、600 dpi PNG + 可编辑 SVG + .mat 数据。

图件清单（对应论文 §5.2 候选）：
  fig1_trajectory_examples  代表性样本轨迹预测（N=2/3/4 各 1 例：3D 轨迹 + 逐步位置误差）
  fig2_error_growth         预测域内位置误差随时间增长（均值 ±1σ，分 N 曲线）
  fig3_error_distribution   样本平均位置误差 / 末端距离分布直方图
  fig4_training_curves      训练曲线（预测损失 + 验证末端距离，多成员）
  fig5_dv_distribution      脉冲 Δv 幅值分布与超越概率（物理一致性，3 m/s 上限）
  fig6_cw_residual          CW 单步递推残差分布（位置 / 速度分量）

用法：
  ① 在 VSCode 中直接点「运行 Python 文件」即可 —— 使用下方 DEFAULT_* 默认配置
     （默认 = 当前最佳 3 成员集成 rp1/rf1/rf2 + 对应训练日志 + output/paper_figures）
  ② 需要覆盖时用命令行参数：
     python plot_paper_figures.py \
         --members output/best_model_rp1.pth output/best_model_rf1.pth output/best_model_rf2.pth \
         --logs output/train_log_rp1.txt output/train_log_rf1.txt output/train_log_rf2.txt \
         --out output/paper_figures
     # 单模型：--members output/best_model.pth
     # 快速自检（少量样本）：--max-samples 4000
     # 换 fig1 选样策略：--sample-select median

可调参数集中在「论文图样式配置」区，按需修改后重跑即可。
本文件已按「逐行注释」要求标注，便于断点调试。
"""

# ============================== 标准库导入 ==============================
import os            # 路径拼接、目录创建、文件存在性判断
import sys           # sys.path 注入、平台判断（sys.platform）
import json          # 指标汇总写 metrics_summary.json
import argparse      # 命令行参数解析（VSCode 无参运行时全部走默认值）

# ================== conda MKL DLL 搜索路径修复 (Windows) ==================
# 背景：conda 环境的 MKL/OpenMP DLL 不在 PATH 时，torch/numpy 导入可能因找不到
#      libiomp5md.dll 等报错；这里把 conda 的 Library/bin 前置到 PATH。
if sys.platform == "win32":                                      # 仅在 Windows 上执行
    # 依次尝试 CONDA_PREFIX 环境变量、当前解释器前缀两个候选根目录
    for _prefix in [os.environ.get("CONDA_PREFIX", ""), sys.prefix]:
        # 候选目录下拼接 Library/bin（conda 存放原生 DLL 的位置）
        _lib_bin = os.path.join(_prefix, "Library", "bin") if _prefix else ""
        # 目录存在且尚未加入 PATH 时才处理（幂等，避免重复追加）
        if _lib_bin and os.path.isdir(_lib_bin) and _lib_bin not in os.environ.get("PATH", ""):
            # 前置到 PATH 最前面，保证优先命中
            os.environ["PATH"] = _lib_bin + os.pathsep + os.environ.get("PATH", "")

# ============ VSCode 直接运行支持：确保项目根目录在 sys.path 中（无论 cwd 为何） ============
_HERE = os.path.dirname(os.path.abspath(__file__))                # 本文件所在目录 = 项目根 TrajectoryPrediction/
if _HERE not in sys.path:                                        # 若未被自动加入（cwd 不同时可能如此）
    sys.path.insert(0, _HERE)                                    # 手动插到最前，保证 import config/utils 成功

# ============================== 第三方库导入 ==============================
import numpy as np                                               # 数值计算（矩阵、范数、统计）
import torch                                                     # 模型加载与推理（GPU 张量）
import matplotlib                                                # 绘图总入口
matplotlib.use("Agg")                                            # 用无界面后端，避免需要 GUI/显示器（批量出图必需）
import matplotlib.pyplot as plt                                  # 绘图 API
import scipy.io as sio                                           # .mat 数据导出（供 MATLAB 复现图件）
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401             # 注册 3D 投影（仅副作用导入，故标 noqa）

# ============================== 项目内模块 ==============================
import config as C                                               # 全局配置：DATA_DIR / DEVICE / POS_INDICES / CW_DT_H 等
from utils.data_loader import load_and_split, FeatureScaler       # 数据集加载划分 + z-score 标准化器
from utils.ensemble import load_members, predict_batch            # 集成成员加载（自动推断架构）+ 集成推理
from models.physics_loss import compute_cw_matrix, N_MEAN         # CW 状态转移矩阵 Φ(Δt) + 轨道平均角速度 n

# ==================== 论文图样式配置（可调） ====================
plt.rcParams["font.family"] = "serif"                             # 正文用衬线体（期刊排版惯例）
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]  # 首选 Times New Roman，缺字回退 DejaVu
plt.rcParams["mathtext.fontset"] = "stix"                         # 数学字体与 Times 协调，变量自动斜体
plt.rcParams["axes.unicode_minus"] = False                        # 负号用 ASCII 连字符（避免字体缺 U+2212 显示为方框）
plt.rcParams["svg.fonttype"] = "none"                             # SVG 中文本保持可编辑（Inkscape/Illustrator）
plt.rcParams.update({
    "font.size": 8,              # 全局字号（按最终印刷尺寸设定，8 pt 适合双栏正文图）
    "axes.labelsize": 9,         # 坐标轴标签字号
    "axes.titlesize": 9,         # 子图标题字号
    "xtick.labelsize": 8,        # x 轴刻度字号
    "ytick.labelsize": 8,        # y 轴刻度字号
    "legend.fontsize": 7.5,      # 图例字号
    "lines.linewidth": 1.2,      # 默认线宽
    "axes.linewidth": 0.8,       # 坐标轴边框线宽
    "xtick.major.width": 0.8,    # x 主刻度线宽
    "ytick.major.width": 0.8,    # y 主刻度线宽
    "xtick.direction": "in",     # 刻度朝内（期刊常用）
    "ytick.direction": "in",     # 同上
    "xtick.top": True,           # 顶部也画刻度
    "ytick.right": True,         # 右侧也画刻度
    "legend.framealpha": 0.9,    # 图例边框透明度（略透明不遮挡）
    "legend.edgecolor": "0.7",   # 图例边框浅灰
})

# ============================== 全局常量（可调） ==============================
FIG_DPI = 600                      # PNG 分辨率（论文要求 ≥300，600 便于放大检查）
W_DOUBLE = 7.2                     # 双栏图宽 (inch) ≈ 183 mm
W_SINGLE = 3.5                     # 单栏图宽 (inch) ≈ 89 mm
TARGET_COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]  # 色盲安全配色（T1..T4）
C_KNOWN = "#999999"                # 已知轨迹颜色（灰，弱化处理）
C_TRUE = "#000000"                 # 真实未来轨迹颜色（黑）
C_PRED = "#E69F00"                 # 预测轨迹颜色（橙，目前 fig1 按目标配色，此项备用）
LINE_STYLES = ["-", "--", ":"]     # 成员/分组辅助线型（实线/虚线/点线循环）

# ==================== VSCode「运行 Python 文件」默认配置（可调） ====================
# 直接运行不传任何参数时使用以下配置；命令行参数会覆盖同名项。
_PROJ = _HERE                                                    # 项目根 = 本文件所在目录

DEFAULT_MEMBERS = [                                              # 集成成员（按顺序等权平均）
    "output/best_model_rp1.pth",                                 # 384 维 + 位置损失
    "output/best_model_rf1.pth",                                 # 512 维 + 位置损失（种子 102）
    "output/best_model_rf2.pth",                                 # 512 维 + 位置损失（种子 103）
]
DEFAULT_MEMBER_NAMES = ["PI-LSTM-384", "PI-LSTM-512-a", "PI-LSTM-512-b"]  # fig4 图例显示名
DEFAULT_LOGS = [                                                 # fig4 训练曲线用的日志（缺失自动跳过）
    "output/train_log_rp1.txt",                                  # 成员 A 训练日志
    "output/train_log_rf1.txt",                                  # 成员 B 训练日志
    "output/train_log_rf2.txt",                                  # 成员 C 训练日志
]
DEFAULT_OUT = os.path.join(_PROJ, "output", "paper_figures")     # 图件输出目录（绝对路径，与 cwd 无关）
DEFAULT_SAMPLE_SELECT = "best"                                   # fig1 选样：best（末端最小）| median（中位）
DEFAULT_MAX_SAMPLES = None                                       # None = 全量测试集；调试可设 4000 加速


# ============================== 默认值解析工具 ==============================

def _resolve_default_members():
    """默认成员不存在时回退：优先用 3 成员集成，缺失则用单模型 best_model.pth。"""
    # 过滤出磁盘上真实存在的成员（路径相对项目根）
    exist = [m for m in DEFAULT_MEMBERS if os.path.exists(os.path.join(_PROJ, m))]
    if exist:                                                    # 至少有一个存在 → 用这些
        return exist
    fallback = "output/best_model.pth"                            # 全部缺失时回退到主模型
    if os.path.exists(os.path.join(_PROJ, fallback)):             # 主模型存在则采用
        print(f"[提示] 集成成员缺失，回退单模型: {fallback}")
        return [fallback]
    return DEFAULT_MEMBERS                                        # 都没有 → 原样返回，交给下游报错并提示


def _resolve_default_logs():
    """只保留存在的训练日志；全部缺失则返回 None（跳过 fig4）。"""
    # 过滤存在的日志文件（避免 fig4 因文件缺失而报错）
    exist = [p for p in DEFAULT_LOGS if os.path.exists(os.path.join(_PROJ, p))]
    if not exist:                                                # 一个都没有 → 明确告知并跳过 fig4
        print("[提示] 未找到训练日志，跳过 fig4 训练曲线。")
        return None
    return exist                                                 # 返回可用日志列表


# ============================== 统一出图保存 ==============================

def _save(fig, out_dir, name, mat_data=None):
    """统一保存 PNG(600dpi) + SVG + 可选 .mat 数据。

    容错：若目标文件被预览/看图软件占用（Windows 下 open 会报 OSError Errno 22），
    改存为 `<name>_new.<ext>` 并给出提示，不中断其余图件的生成。
    """
    os.makedirs(out_dir, exist_ok=True)                          # 确保输出目录存在（幂等）

    def _try_save(suffix, **kw):
        """内部工具：按扩展名保存，占用时自动换备用文件名。kw 透传 savefig 参数。"""
        path = os.path.join(out_dir, f"{name}.{suffix}")         # 目标绝对路径
        try:
            fig.savefig(path, **kw)                              # 正常路径保存
            return os.path.basename(path)                        # 返回实际落盘文件名
        except OSError as e:                                     # 命中文件被占用（Windows Errno 22 等）
            alt = os.path.join(out_dir, f"{name}_new.{suffix}")   # 备用文件名
            try:
                fig.savefig(alt, **kw)                           # 尝试写备用名
            except OSError:                                      # 备用名也失败 → 放弃该文件，不中断脚本
                print(f"  [失败] {name}.{suffix} 及备用名均写入失败: {e}")
                return None
            # 成功写备用名：打印可操作提示（用户关闭预览后重命名即可）
            print(f"  [警告] {name}.{suffix} 被占用（{e}）→ 已改存 {os.path.basename(alt)}，"
                  f"请关闭预览/看图软件后重命名")
            return os.path.basename(alt)                         # 返回备用文件名

    png = _try_save("png", dpi=FIG_DPI, bbox_inches="tight")      # 位图：600 dpi，裁剪多余白边
    svg = _try_save("svg", bbox_inches="tight")                   # 矢量：文本可编辑（投稿/放大用）
    plt.close(fig)                                                # 关闭图对象，释放内存（批量出图必需）
    if png:                                                       # 只要 PNG 成功就打印一行结果
        print(f"  [图] {png} / {svg}")
    if mat_data:                                                  # 可选：同步导出绘图数据
        try:
            sio.savemat(os.path.join(out_dir, f"{name}.mat"), mat_data)  # 存 .mat 供 MATLAB 复核/重画
        except OSError as e:                                      # .mat 被占用时不致命
            print(f"  [警告] {name}.mat 写入失败（文件被占用？）: {e}")


# ============================== 数据与推理 ==============================

def run_inference(members, te_X, te_M, batch_size=512, max_samples=None):
    """对测试集做集成推理，返回标准化空间预测 (N,10,24)。"""
    # 决定实际评估样本数：未指定上限则全量，否则取较小值
    n = te_X.shape[0] if max_samples is None else min(max_samples, te_X.shape[0])
    outs = []                                                     # 逐批预测结果缓存
    with torch.no_grad():                                         # 推理无需梯度（省显存、加速）
        for i in range(0, n, batch_size):                         # 按 batch_size 逐批遍历
            x = torch.from_numpy(te_X[i:i + batch_size]).float().to(C.DEVICE)   # 本批输入 → float32 → 设备
            m = torch.from_numpy(te_M[i:i + batch_size]).bool().to(C.DEVICE)    # 本批 mask → bool → 设备
            p = predict_batch(members, x, m)                      # 集成推理（标准化空间等权平均）
            outs.append(p.cpu().numpy())                          # 搬回 CPU 并转 numpy 入库
    return np.concatenate(outs, axis=0)                           # 拼成 (n, 10, 24)


def physical_errors(pred_raw, true_raw, masks):
    """
    计算物理空间逐样本/逐步位置误差。
    Returns:
        err3d:   (N,10,max_N) 每目标 3D 位置误差范数 (km)，无效目标为 NaN
        td:      (N,) 末端距离（末步最差目标，km），与训练日志口径一致
        n_arr:   (N,) 各样本目标数
    """
    N = pred_raw.shape[0]                                         # 样本数
    pos = np.array(C.POS_INDICES)                                 # 24 维中的位置分量索引（0,1,2,6,7,8,...）
    P = pred_raw[:, :, pos].reshape(N, C.OUTPUT_STEPS, C.MAX_N, 3)  # 预测 → (N,10,4目标,3坐标)，单位 km
    T = true_raw[:, :, pos].reshape(N, C.OUTPUT_STEPS, C.MAX_N, 3)  # 真值 → 同形状
    valid = masks[:, pos].reshape(N, C.MAX_N, 3).any(-1)          # (N,max_N) 各目标是否有任一坐标有效

    err3d = np.linalg.norm(P - T, axis=-1)                        # (N,10,max_N) 每目标每步 3D 误差范数
    err3d[~np.broadcast_to(valid[:, None, :], err3d.shape)] = np.nan  # 无效（padding）目标置 NaN，避免污染统计
    td = np.nanmax(err3d[:, -1, :], axis=1)                       # 末步取「最差目标」误差 = 末端距离（与训练判据一致）
    n_arr = valid.sum(axis=1)                                     # 各样本真实目标数 N ∈ {2,3,4}
    return err3d, td, n_arr                                       # 返回三元组供 figs 1–3 使用


def cw_and_dv(pred_raw, masks):
    """
    CW 单步递推残差（位置 km / 速度 km/s 分离）与脉冲 Δv（m/s），向量化实现。
    口径与 evaluate.compute_cw_metrics 一致（步长 CW_DT_H=60 s，脉冲=扣除自由演化）。
    """
    Phi = compute_cw_matrix(N_MEAN, C.CW_DT_H).numpy()            # (6,6) 单步 CW 状态转移矩阵（60 s）
    N = pred_raw.shape[0]                                         # 样本数（此处仅用于可读性，循环按目标切分）
    pos_res, vel_res, dv = [], [], []                             # 三个累加容器：位置残差 / 速度残差 / 脉冲 Δv
    for a in range(C.MAX_N):                                      # 逐目标槽位（0..3）处理
        base = a * 6                                              # 该目标 6 维状态在 24 维中的起始下标
        valid = masks[:, base]                                     # 该目标有效样本（N<4 时后置槽位全 False）
        if not valid.any():                                       # 该槽位无任何有效样本 → 跳过
            continue
        S = pred_raw[valid, :, base:base + 6]                      # (M,10,6) 该目标的位置+速度序列
        s_free = S[:, :-1, :] @ Phi.T                              # (M,9,6) 由前一步自由 CW 外推得到的下一步状态
        r = S[:, 1:, :] - s_free                                   # (M,9,6) 实际下一步 − 自由外推 = 偏差（含机动贡献）
        pos_res.append(np.linalg.norm(r[..., :3], axis=-1).ravel())   # 位置残差范数 (km) 压平收集
        vel_res.append(np.linalg.norm(r[..., 3:], axis=-1).ravel())   # 速度残差范数 (km/s) 压平收集
        dv.append(np.linalg.norm(S[:, 1:, 3:] - s_free[..., 3:], axis=-1).ravel())  # 脉冲 Δv = 速度增量范数
    pos_res = np.concatenate(pos_res)                             # 拼接所有目标的样本
    vel_res = np.concatenate(vel_res)                             # 同上
    dv = np.concatenate(dv) * 1000.0                               # km/s → m/s（与 3 m/s 约束同量纲）
    return pos_res, vel_res, dv                                   # 供 fig5 / fig6 使用


# ============================== 图 1：代表性样本轨迹 ==============================

def fig1_trajectory_examples(X_raw, pred_raw, true_raw, err3d, td, n_arr, out_dir,
                             sample_select="best"):
    """N=2/3/4 各取 1 个代表性样本。
    sample_select="best"   → 各组内末端距离最小（预测最佳，展示效果）
    sample_select="median" → 各组内末端距离最接近中位数（统计代表性）
    """
    cols = []                                                     # 选中的样本下标（每列一个样本）
    for n_val in (2, 3, 4):                                       # 依次为 N=2/3/4 各选一个
        idx = np.where(n_arr == n_val)[0]                          # 该 N 组的所有样本下标
        if len(idx) == 0:                                          # 该组为空（理论上不会）→ 跳过
            continue
        if sample_select == "best":                                # 策略一：末端误差最小
            cols.append(idx[np.argmin(td[idx])])                   # 组内 argmin(td) 对应样本
        else:                                                      # 策略二：末端误差最接近中位数
            med = np.median(td[idx])                               # 组内中位数
            cols.append(idx[np.argmin(np.abs(td[idx] - med))])      # 最接近中位数的样本
    if not cols:                                                   # 一列都没选到 → 直接返回
        return

    t_known = np.arange(1, C.INPUT_STEPS + 1) * C.CW_DT_H          # 观测窗绝对时间 (s)：60,120,...,600
    t_fut = np.arange(C.INPUT_STEPS + 1, C.INPUT_STEPS + C.OUTPUT_STEPS + 1) * C.CW_DT_H  # 预测窗：660,...,1200

    fig = plt.figure(figsize=(W_DOUBLE, 4.6))                      # 整图尺寸（双栏宽 × 4.6 in 高）
    labels = ["(a)", "(b)", "(c)"]                                 # 上排面板标签（面板标记按期刊惯例）

    for j, si in enumerate(cols):                                  # j = 列号，si = 样本下标
        n_t = int(n_arr[si])                                       # 该样本的目标数
        # ── 上排：3D 轨迹 ──
        ax = fig.add_subplot(2, len(cols), j + 1, projection="3d")  # 2×cols 网格的第 j 个（上排），3D 投影
        for t in range(n_t):                                       # 逐目标绘制
            b = t * 6                                              # 该目标状态起始下标
            ax.plot(X_raw[si, :, b], X_raw[si, :, b + 1], X_raw[si, :, b + 2],
                    color=TARGET_COLORS[t], ls=":", lw=1.2, alpha=0.75)   # 观测段：点线、半透明
            ax.plot(true_raw[si, :, b], true_raw[si, :, b + 1], true_raw[si, :, b + 2],
                    color=TARGET_COLORS[t], ls="-", lw=1.5)                # 真实未来：实线
            ax.plot(pred_raw[si, :, b], pred_raw[si, :, b + 1], pred_raw[si, :, b + 2],
                    color=TARGET_COLORS[t], ls="--", lw=1.5)               # 预测未来：虚线
            ax.scatter(*X_raw[si, -1, b:b + 3], color=TARGET_COLORS[t], s=12, marker="o")  # 观测末点标记
        # LVLH 原点 = 观测航天器 O（物理情境参考）
        ax.scatter(0, 0, 0, marker="*", s=70, c="k", depthshade=False)     # 原点星标（不随深度变淡）
        ax.plot([], [], color=C_KNOWN, ls=":", label="Observed")           # 图例占位：观测段线型
        ax.plot([], [], color="k", ls="-", label="True")                   # 图例占位：真实未来
        ax.plot([], [], color="k", ls="--", label="Predicted")             # 图例占位：预测未来
        ax.plot([], [], color="k", ls="none", marker="*", ms=8, label="$O$ (origin)")  # 图例占位：原点
        for t in range(n_t):                                               # 图例占位：各目标配色
            ax.plot([], [], color=TARGET_COLORS[t], ls="-", lw=2.5, label=f"$T_{t+1}$")
        ax.set_xlabel("$x$ (km)", labelpad=1)                              # x 轴标签（斜体变量）
        ax.set_ylabel("$y$ (km)", labelpad=1)                              # y 轴标签
        ax.set_zlabel("$z$ (km)", labelpad=1)                              # z 轴标签
        ax.tick_params(pad=0)                                              # 刻度标签贴近轴（省空间）
        ax.set_title(f"{labels[j]} $N={n_t}$", pad=2)                       # 面板标签 + 目标数
        if j == 0:                                                         # 仅首列画图例，避免重复
            ax.legend(loc="upper left", fontsize=6.5, borderpad=0.3, labelspacing=0.3)
        ax.view_init(elev=22, azim=-58)                                    # 固定视角（保证图件可复现）

        # ── 下排：逐步 3D 位置误差 ──
        ax2 = fig.add_subplot(2, len(cols), len(cols) + j + 1)             # 下排对应列（普通 2D 轴）
        for t in range(n_t):                                               # 逐目标误差曲线
            ax2.plot(t_fut, err3d[si, :, t] / 3, color=TARGET_COLORS[t], ls="-",
                     marker="o", ms=2.5, lw=1.2, label=f"$T_{t+1}$")       # ⚠️ 除以 3 为手工加入的压缩显示（见文件末注）
        ax2.plot(t_fut, np.nanmean(err3d[si], axis=1) / 3, color="k", ls="--", lw=1.4,
                 label="Mean")                                             # 同尺度（/3）的目标均值曲线
        ax2.set_xlabel("$t$ (s)")                                          # 横轴：绝对任务时间
        ax2.set_ylabel("Position error (km)")                              # ⚠️ 若保留 /3，此单位应为 km 的 1/3 缩放
        ax2.set_xticks(t_fut[::3])                                         # 刻度稀疏化（每 3 步一个）
        ax2.set_ylim(0, float(np.nanmax(err3d[si])) * 1.15 / 3)   # 底部归零 + 顶部留白（同样 /3 缩放）
        ax2.grid(True, alpha=0.3, lw=0.5)                                  # 淡网格线
        ax2.set_title(f"({chr(ord('d') + j)}) Error growth, $N={n_t}$", pad=2)  # 面板标签 (d)(e)(f)
        ax2.text(0.03, 0.97, f"terminal: {td[si]:.3f} km", transform=ax2.transAxes,
                 fontsize=7, va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5))  # 左上角标注末端距离
        if j == 0:                                                         # 仅首列画图例
            ax2.legend(loc="upper left", fontsize=6.5, borderpad=0.3, labelspacing=0.3,
                       bbox_to_anchor=(0.0, 0.88))                         # 下移一点，避开上方标注框

    fig.tight_layout()                                                     # 自动调整子图间距
    mat = {"t_known": t_known, "t_future": t_fut, "sample_indices": np.array(cols)}  # .mat 数据：时间轴+样本号
    for j, si in enumerate(cols):                                          # 逐样本写入原始误差矩阵
        mat[f"td_sample{j}"] = td[si]                                      # 该样本末端距离
        mat[f"err3d_sample{j}"] = err3d[si]                                # 该样本 (10,max_N) 误差矩阵
    _save(fig, out_dir, "fig1_trajectory_examples", mat)                    # 统一保存三件套


# ============================== 图 2：误差随预测域增长 ==============================

def fig2_error_growth(err3d, n_arr, out_dir):
    # 时间轴与图 1 一致：绝对任务时间（观测窗 60–600 s，预测窗 660–1200 s）
    t_fut = np.arange(C.INPUT_STEPS + 1, C.INPUT_STEPS + C.OUTPUT_STEPS + 1) * C.CW_DT_H
    # 注意：填充目标为 NaN，必须 nanmean（否则 N<4 样本整体失效）
    per_sample = np.nanmean(err3d, axis=2)                        # (N,10) 每样本逐步的平均误差（跨目标取均值）
    step_mean = np.nanmean(per_sample, axis=0)                    # (10,) 逐步总体均值
    step_std = np.nanstd(per_sample, axis=0)                      # (10,) 逐步标准差（做 ±1σ 带）

    fig, ax = plt.subplots(figsize=(W_SINGLE * 1.35, 2.6))        # 单栏略宽的画布
    ax.fill_between(t_fut, np.maximum(step_mean - step_std, 0), step_mean + step_std,
                    color="0.6", alpha=0.25, lw=0, label=r"$\pm 1\sigma$ (all)")  # ±1σ 阴影带（下界截到 0）
    ax.plot(t_fut, step_mean, color=C_TRUE, ls="-", marker="o", ms=3, lw=1.4,
            label="All samples")                                  # 总体均值曲线
    for k, n_val in enumerate((2, 3, 4)):                         # 分 N 曲线
        sel = n_arr == n_val                                      # 该 N 组样本布尔索引
        if sel.any():                                             # 组内有样本才画
            m = np.nanmean(per_sample[sel], axis=0)               # 该组逐步均值
            ax.plot(t_fut, m, color=TARGET_COLORS[k], ls="--", lw=1.1, label=f"$N={n_val}$")
    ax.set_xlabel("$t$ (s)")                                      # 横轴：绝对时间
    ax.set_ylabel("Position error (km)")                          # 纵轴：位置误差 (km)
    ax.set_xticks(t_fut[::3])                                     # 稀疏刻度
    ax.grid(True, alpha=0.3, lw=0.5)                              # 淡网格
    ax.legend(loc="upper left", ncols=2, columnspacing=0.8, borderpad=0.3, labelspacing=0.3)  # 两列图例
    fig.tight_layout()                                            # 紧凑布局
    _save(fig, out_dir, "fig2_error_growth",
          {"t_future": t_fut, "err_mean": step_mean, "err_std": step_std})   # 导出均值/标准差数据


# ============================== 图 3：误差分布 ==============================

def fig3_error_distribution(err3d, td, out_dir):
    sample_mean = np.nanmean(err3d, axis=(1, 2))                  # (N,) 每样本平均位置误差（跨步与目标）

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE * 0.82, 2.5))  # 左右两个直方图
    for ax, data, xlab, lab in (                                  # 统一循环：同一套样式画两幅
        (axes[0], sample_mean, "Mean position error per sample (km)", "(a)"),  # 左：样本均值误差
        (axes[1], td, "Terminal distance (km)", "(b)"),            # 右：末端距离
    ):
        ax.hist(data, bins=60, color="#0072B2", edgecolor="white", lw=0.4, alpha=0.9)  # 直方图
        ax.axvline(np.mean(data), color="#D55E00", ls="--", lw=1.2,
                   label=f"Mean = {np.mean(data):.3f} km")        # 均值参考线
        ax.axvline(np.median(data), color="#009E73", ls=":", lw=1.2,
                   label=f"Median = {np.median(data):.3f} km")    # 中位数参考线
        ax.set_yscale("log")                                      # 对数纵轴（尾部可见）
        ax.set_xlabel(xlab)                                       # 横轴标签
        ax.set_ylabel("Count")                                    # 纵轴：频次
        ax.set_title(lab, loc="left", pad=2)                      # 面板标签
        ax.legend(loc="upper right", borderpad=0.3, labelspacing=0.3)  # 图例
        ax.grid(True, alpha=0.3, lw=0.5)                          # 淡网格
    fig.tight_layout()                                            # 紧凑布局
    _save(fig, out_dir, "fig3_error_distribution",
          {"sample_mean_error": sample_mean, "terminal_distance": td})   # 导出原始分布数据


# ============================== 图 4：训练曲线 ==============================

def _parse_log(log_path):
    """从训练日志解析逐轮指标（与 evaluate.plot_loss_curve 同口径）。"""
    d = {k: [] for k in ("train_pred", "val_pred", "val_td")}     # 三个待填列表
    with open(log_path, "r", encoding="utf-8") as f:              # 逐行读取日志
        for line in f:
            if "Train:" not in line or "pred=" not in line:       # 只处理含指标的行（跳过进度条等）
                continue
            try:
                d["train_pred"].append(float(line.split("Train:")[1].split("pred=")[1].split()[0]))  # 训练预测损失
                d["val_pred"].append(float(line.split("Val:")[1].split("pred=")[1].split()[0]))      # 验证预测损失
                d["val_td"].append(float(line.split("td=")[1].split("km")[0]))                       # 验证末端距离
            except (IndexError, ValueError):                      # 行格式不符（如进度条行）→ 静默跳过
                pass
    return d                                                      # 返回字典（每个键为一个 epoch 一个值）


def fig4_training_curves(log_paths, member_names, out_dir):
    logs = [_parse_log(p) for p in log_paths]                     # 依次解析各成员日志

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE * 0.82, 2.5))  # 左右两幅子图

    # (a) 首成员训练/验证预测损失（标准化空间，对数轴）
    ax = axes[0]                                                  # 左图句柄
    d0 = logs[0]                                                  # 取第一个成员的日志
    ep = np.arange(1, len(d0["train_pred"]) + 1)                  # epoch 轴（1..E）
    ax.plot(ep, d0["train_pred"], color="#0072B2", ls="-", label="Train")        # 训练预测损失
    ax.plot(ep, d0["val_pred"], color="#D55E00", ls="--", label="Validation")    # 验证预测损失
    ax.set_yscale("log")                                          # 对数纵轴（跨量级更清晰）
    ax.set_xlabel("Epoch")                                        # 横轴：轮次
    ax.set_ylabel("Prediction loss (normalized)")                 # 纵轴：标准化空间损失
    ax.set_title("(a)", loc="left", pad=2)                        # 面板标签
    ax.legend(loc="upper right", borderpad=0.3, labelspacing=0.3)  # 图例
    ax.grid(True, alpha=0.3, lw=0.5, which="both")                 # 主/次刻度网格都画

    # (b) 各成员验证末端距离（模型选择判据）
    ax = axes[1]                                                  # 右图句柄
    for k, (d, name) in enumerate(zip(logs, member_names)):       # 逐成员绘制
        ep = np.arange(1, len(d["val_td"]) + 1)                   # 该成员 epoch 轴
        ax.plot(ep, d["val_td"], color=TARGET_COLORS[k % len(TARGET_COLORS)],
                ls=LINE_STYLES[k % 3], lw=1.1, label=name)        # 验证末端距离曲线
        bi = int(np.argmin(d["val_td"]))                          # 该成员最优 epoch
        ax.plot(ep[bi], d["val_td"][bi], marker="*", ms=7,
                color=TARGET_COLORS[k % len(TARGET_COLORS)], ls="none")  # 最优点星标
    ax.set_xlabel("Epoch")                                        # 横轴
    ax.set_ylabel("Val. terminal distance (km)")                  # 纵轴
    ax.set_title("(b)", loc="left", pad=2)                        # 面板标签
    ax.legend(loc="upper right", borderpad=0.3, labelspacing=0.3)  # 图例（成员名）
    ax.grid(True, alpha=0.3, lw=0.5)                              # 淡网格
    fig.tight_layout()                                            # 紧凑布局

    mat = {}                                                      # .mat 数据容器
    for d, name in zip(logs, member_names):                       # 逐成员导出验证末端距离
        safe = name.replace(" ", "_").replace("=", "")            # 变量名合法化（.mat 键名限制）
        mat[f"val_td_{safe}"] = np.array(d["val_td"])             # 写入
    mat["train_pred_first"] = np.array(logs[0]["train_pred"])      # 首成员训练损失
    mat["val_pred_first"] = np.array(logs[0]["val_pred"])          # 首成员验证损失
    _save(fig, out_dir, "fig4_training_curves", mat)               # 统一保存


# ============================== 图 5：脉冲 Δv 分布 ==============================

def fig5_dv_distribution(dv, out_dir):
    over = float(np.mean(dv > C.DELTAV_LIMIT)) * 100              # 超 3 m/s 约束的占比（%）

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE * 0.82, 2.5))  # 左：直方图；右：超越概率

    ax = axes[0]                                                  # 左图句柄
    ax.hist(dv, bins=80, color="#0072B2", edgecolor="white", lw=0.3, alpha=0.9)  # Δv 幅值直方图
    ax.axvline(C.DELTAV_LIMIT, color="#D55E00", ls="--", lw=1.2,
               label=f"Bound {C.DELTAV_LIMIT:.0f} m/s")           # 3 m/s 物理上限参考线
    ax.set_yscale("log")                                          # 对数纵轴（尾部可见）
    ax.set_xlabel(r"Impulsive $\Delta v$ magnitude (m/s)")         # 横轴：脉冲 Δv (m/s)
    ax.set_ylabel("Count")                                        # 纵轴：频次
    ax.set_title("(a)", loc="left", pad=2)                        # 面板标签
    ax.legend(loc="upper right", borderpad=0.3,
              title=f"Over-bound: {over:.2f}%", title_fontsize=7.5, labelspacing=0.3)  # 图例含超限占比
    ax.grid(True, alpha=0.3, lw=0.5)                              # 淡网格

    ax = axes[1]                                                  # 右图句柄
    xs = np.sort(dv)                                              # Δv 升序
    surv = 1.0 - np.arange(1, len(xs) + 1) / len(xs)               # 经验超越概率 (1-ECDF)
    ax.plot(xs, surv, color="#009E73", lw=1.3)                    # 超越概率曲线
    ax.axvline(C.DELTAV_LIMIT, color="#D55E00", ls="--", lw=1.2)   # 上限参考线（无图例，与左图共享语义）
    ax.set_yscale("log")                                          # 对数纵轴
    ax.set_xlabel(r"Impulsive $\Delta v$ magnitude (m/s)")         # 横轴
    ax.set_ylabel(r"$P(\Delta v > x)$")                            # 纵轴：超越概率
    ax.set_title("(b)", loc="left", pad=2)                        # 面板标签
    ax.grid(True, alpha=0.3, lw=0.5, which="both")                 # 主/次网格
    fig.tight_layout()                                            # 紧凑布局
    _save(fig, out_dir, "fig5_dv_distribution",
          {"dv_magnitude_ms": dv, "over_bound_percent": over})     # 导出原始 Δv 数据与超限占比


# ============================== 图 6：CW 残差分布 ==============================

def fig6_cw_residual(pos_res, vel_res, pred_raw, masks, out_dir):
    """(a) CW 位置残差分布；(b) CW 残差随预测步增长（新信息维度，避免与图 5 冗余）。"""
    Phi = compute_cw_matrix(N_MEAN, C.CW_DT_H).numpy()            # (6,6) 单步 CW 转移矩阵
    # 逐步残差：对每个目标分别按步统计
    step_pos, step_vel = [], []                                   # 各步残差均值容器
    for step in range(C.OUTPUT_STEPS - 1):                        # 9 个相邻步对
        p_acc, v_acc = [], []                                     # 本步内各目标的残差收集
        for a in range(C.MAX_N):                                  # 逐目标槽位
            base = a * 6                                          # 状态起始下标
            valid = masks[:, base]                                # 该目标有效样本
            if not valid.any():                                   # 无有效样本 → 跳过
                continue
            S = pred_raw[valid, :, base:base + 6]                 # (M,10,6) 该目标预测状态序列
            s_free = S[:, step, :] @ Phi.T                        # (M,6) 由第 step 步自由外推的第 step+1 步
            r = S[:, step + 1, :] - s_free                        # (M,6) 残差（含机动贡献）
            p_acc.append(np.linalg.norm(r[:, :3], axis=-1))       # 位置残差范数 (km)
            v_acc.append(np.linalg.norm(r[:, 3:], axis=-1))       # 速度残差范数 (km/s)
        step_pos.append(np.concatenate(p_acc).mean())             # 本步位置残差均值
        step_vel.append(np.concatenate(v_acc).mean() * 1000.0)     # → m/s
    t_steps = (np.arange(2, C.OUTPUT_STEPS + 1) + C.INPUT_STEPS) * C.CW_DT_H  # 绝对时间

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE * 0.82, 2.5))  # 左右两幅

    ax = axes[0]                                                  # 左：位置残差分布
    ax.hist(pos_res, bins=80, color="#0072B2", edgecolor="white", lw=0.3, alpha=0.9)  # 直方图
    ax.set_yscale("log")                                          # 对数纵轴
    ax.set_xlabel("CW position residual (km)")                    # 横轴：位置残差 (km)
    ax.set_ylabel("Count")                                        # 纵轴：频次
    ax.set_title(f"(a) Mean = {np.mean(pos_res):.3f} km", loc="left", pad=2)  # 面板标签含均值
    ax.grid(True, alpha=0.3, lw=0.5)                              # 淡网格

    ax = axes[1]                                                  # 右：残差随步增长（双纵轴）
    ax.plot(t_steps, step_pos, color="#0072B2", ls="-", marker="o", ms=3,
            label="Position (km)")                                # 左轴：位置残差曲线
    ax.set_xlabel("$t$ (s)")                                      # 横轴：绝对时间
    ax.set_ylabel("CW position residual (km)", color="#0072B2")    # 左轴标签（蓝，与曲线同色）
    ax.set_xticks(t_steps[::3])                                   # 稀疏刻度
    ax.grid(True, alpha=0.3, lw=0.5)                              # 淡网格
    ax2 = ax.twinx()                                              # 建立共享 x 轴的右侧纵轴
    ax2.plot(t_steps, step_vel, color="#009E73", ls="--", marker="s", ms=3,
             label="Velocity (m/s)")                              # 右轴：速度残差曲线
    ax2.set_ylabel("CW velocity residual (m/s)", color="#009E73")  # 右轴标签（绿）
    ax.set_title("(b) Residual growth over horizon", loc="left", pad=2)  # 面板标签
    h1, l1 = ax.get_legend_handles_labels()                       # 收集左轴图例句柄
    h2, l2 = ax2.get_legend_handles_labels()                      # 收集右轴图例句柄
    ax.legend(h1 + h2, l1 + l2, loc="upper center", ncols=2, borderpad=0.3,
              labelspacing=0.3, columnspacing=0.9)                # 合并图例（顶部居中，避开 U 形曲线）
    fig.tight_layout()                                            # 紧凑布局
    _save(fig, out_dir, "fig6_cw_residual",
          {"cw_pos_residual_km": pos_res, "cw_vel_residual_ms": vel_res * 1000.0,
           "t_steps": t_steps, "step_pos_res_km": np.array(step_pos),
           "step_vel_res_ms": np.array(step_vel)})               # 导出分布与逐步曲线数据


# ============================== 主流程 ==============================

def main():
    ap = argparse.ArgumentParser(
        description="论文仿真图生成（PI-LSTM 轨迹预测）—— 不传参数时用脚本内 DEFAULT_* 默认配置")
    ap.add_argument("--members", nargs="+", default=None,
                    help="集成成员权重路径（1 个 = 单模型）；缺省用 DEFAULT_MEMBERS（3 成员集成）")
    ap.add_argument("--member-names", nargs="+", default=None,
                    help="成员显示名（用于图 4 图例），缺省用 DEFAULT_MEMBER_NAMES")
    ap.add_argument("--logs", nargs="+", default=None,
                    help="各成员训练日志路径（图 4 用）；缺省用 DEFAULT_LOGS（缺失自动跳过）")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="输出目录")
    ap.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
                    help="限制测试样本数（快速自检用；缺省全量）")
    ap.add_argument("--sample-select", choices=["best", "median"], default=DEFAULT_SAMPLE_SELECT,
                    help="图 1 选样策略：best=各 N 组末端距离最小（默认），median=中位数代表")
    args = ap.parse_args()                                        # 解析命令行参数

    # ── 默认值解析（VSCode 直接运行时走这里）──
    if args.members is None:                                      # 未指定成员 → 用默认（含存在性回退）
        args.members = _resolve_default_members()
    if args.logs is None:                                         # 未指定日志 → 用默认（缺失则 None）
        args.logs = _resolve_default_logs()
    names = args.member_names or (                                # 图例显示名：优先用户指定
        DEFAULT_MEMBER_NAMES if list(args.members) == list(DEFAULT_MEMBERS)
        else [os.path.splitext(os.path.basename(p))[0].replace("best_model_", "")  # 否则用文件名派生
              for p in args.members])
    # 日志路径统一转绝对路径（避免 VSCode 运行目录与项目根不一致）
    log_paths = None                                              # 默认无日志
    if args.logs:                                                 # 有日志时逐条转绝对路径
        log_paths = [p if os.path.isabs(p) else os.path.join(_PROJ, p) for p in args.logs]

    print("=" * 70)                                               # 打印本次运行配置（便于排查）
    print("论文仿真图生成 —— 加载数据与模型")
    print(f"  成员 : {args.members}")                              # 生效的成员列表
    print(f"  日志 : {log_paths}")                                 # 生效的日志列表
    print(f"  输出 : {args.out}")                                  # 生效的输出目录
    print("=" * 70)
    (_, _, te_X, _, _, te_Y, _, _, te_M, _) = load_and_split(C.DATA_DIR)  # 只需测试集三件套（忽略其它返回）
    scaler = FeatureScaler()                                      # 新建标准化器实例
    scaler.load(C.SCALER_SAVE_PATH)                                # 载入训练时保存的 mean/std（口径必须一致）

    print(f"\n加载模型（{len(args.members)} 个成员）...")
    members = load_members(args.members, scaler, C.DEVICE)         # 按检查点形状自动推断架构并加载
    if not members:                                                # 全部成员加载失败 → 明确报错退出
        raise RuntimeError("无可用模型成员。")

    print("\n集成推理（测试集）...")
    pred_norm = run_inference(members, te_X, te_M, max_samples=args.max_samples)  # 标准化空间集成预测
    n_eval = pred_norm.shape[0]                                    # 实际评估样本数
    pred_raw = scaler.inverse_transform(pred_norm)                 # 反归一化 → 物理量纲
    true_raw = scaler.inverse_transform(te_Y[:n_eval])             # 真值同步反归一化
    X_raw = scaler.inverse_transform(te_X[:n_eval])                # 观测窗（fig1 画已知轨迹用）
    masks = te_M[:n_eval]                                          # 有效维度掩码

    # ── 指标（与 _tools/eval_ensemble.py 同口径）──
    err3d, td, n_arr = physical_errors(pred_raw, true_raw, masks)   # 逐目标误差 / 末端距离 / 目标数
    pos = np.array(C.POS_INDICES)                                  # 位置分量索引
    v12 = np.repeat(masks[:, pos].reshape(n_eval, C.MAX_N, 3).any(-1), 3, axis=1)  # 展开回 12 列有效掩码
    pe = (pred_raw[:, :, pos] - true_raw[:, :, pos])               # (N,10,12) 位置误差
    m = np.broadcast_to(v12[:, None, :], pe.shape)                 # 广播成与误差同形状的掩码
    pos_rmse = float(np.sqrt(np.mean(pe[m] ** 2)))                 # 分量级位置 RMSE (km)
    pos_mae = float(np.mean(np.abs(pe[m])))                         # 分量级位置 MAE (km)
    print(f"\n【测试集指标】样本 {n_eval} | 位置 RMSE {pos_rmse:.4f} km | "
          f"MAE {pos_mae:.4f} km | 末端距离 {td.mean():.4f} km")     # 控制台报告

    print("\n计算 CW 残差与脉冲 Δv ...")
    pos_res, vel_res, dv = cw_and_dv(pred_raw, masks)               # 物理一致性指标
    print(f"  CW 位置残差均值 {pos_res.mean():.4e} km | 速度残差均值 {vel_res.mean():.4e} km/s")
    print(f"  脉冲 Δv 均值 {dv.mean():.4f} m/s | 最大 {dv.max():.4f} m/s | "
          f"超 {C.DELTAV_LIMIT:.0f} m/s 占比 {np.mean(dv > C.DELTAV_LIMIT) * 100:.2f}%")

    metrics = {                                                    # 指标汇总（写 JSON 便于引用/核对）
        "n_eval": int(n_eval), "members": args.members,            # 样本数与成员
        "pos_rmse_km": pos_rmse, "pos_mae_km": pos_mae,            # 位置精度
        "terminal_distance_km": float(td.mean()),                  # 末端距离
        "cw_pos_res_mean_km": float(pos_res.mean()),               # CW 位置残差
        "cw_vel_res_mean_kms": float(vel_res.mean()),              # CW 速度残差
        "dv_mean_ms": float(dv.mean()), "dv_max_ms": float(dv.max()),      # Δv 统计
        "dv_over_bound_pct": float(np.mean(dv > C.DELTAV_LIMIT) * 100),    # Δv 超限占比
    }
    os.makedirs(args.out, exist_ok=True)                           # 确保输出目录存在
    with open(os.path.join(args.out, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)         # 写汇总文件（中文不转义，缩进 2）

    print("\n生成图件...")                                          # 依次出图（顺序与论文叙述一致）
    fig1_trajectory_examples(X_raw, pred_raw, true_raw, err3d, td, n_arr, args.out,
                             sample_select=args.sample_select)      # 图 1：代表性轨迹
    fig2_error_growth(err3d, n_arr, args.out)                      # 图 2：误差随预测域增长
    fig3_error_distribution(err3d, td, args.out)                   # 图 3：误差 / 末端距离分布
    if log_paths:                                                  # 图 4：仅当有训练日志时
        fig4_training_curves(log_paths, names[:len(log_paths)], args.out)
    fig5_dv_distribution(dv, args.out)                             # 图 5：脉冲 Δv 分布
    fig6_cw_residual(pos_res, vel_res, pred_raw, masks, args.out)   # 图 6：CW 残差
    print(f"\n全部完成，输出目录: {args.out}")                       # 收尾提示


if __name__ == "__main__":                                         # 作为脚本直接运行（含 VSCode 运行按钮）时
    main()                                                         # 进入主流程

# ============================== 附注（手工改动提醒） ==============================
# ⚠️ fig1 下排误差子图中有 3 处「/ 3」缩放（err3d[...] / 3、nanmean(...) / 3、ylim(...) * 1.15 / 3），
#    为 2026-09-22 20:03 手工加入的压缩显示，非物理量纲换算。若保留：
#      · 纵轴标签 "Position error (km)" 已与实际不符（实为 km 的 1/3 缩放）；
#      · 左上角 "terminal: X km" 标注未缩放（真实 km），二者数值不自洽。
#    要恢复严格物理口径：删除上述 3 处「/ 3」即可。
