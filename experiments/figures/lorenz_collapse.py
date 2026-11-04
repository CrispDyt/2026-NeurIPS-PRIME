#!/usr/bin/env python3
"""Portable extraction of the original notebook plotting code; see docs/figures.md."""
import argparse
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def main():
    import os
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/figures/lorenz_collapse')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--checkpoint', type=lambda s: Path(s).resolve(), required=True, help='Original notebook checkpoint path is missing; select a checkpoint explicitly.')
    args = p.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import pandas  # initialize before legacy NumPy pickle aliases
    import numpy as np
    import sys
    import torch
    torch.set_num_threads(1)
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; --device cpu is a numerical smoke test.')
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    os.chdir(args.output_dir)
    args.checkpoint = args.checkpoint.resolve()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    import io
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MaxAbsScaler

    # ============================================================
    # 1. 加载数据与基础设置
    # ============================================================
    train_data = np.load(str(ROOT/'data/lorenz63/train_data.npy'))
    test_data = np.load(str(ROOT/'data/lorenz63/test_data.npy'))

    dt_obs  = 0.1 # Observation 
    dt_base = 0.1 # Intergration step
    sub = int(round(dt_obs/dt_base))

    device = args.device
    print(f"Using device: {device}")

    # ============================================================
    # 2. 模型加载与重构依赖函数
    # ============================================================
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
            return super().find_class(module, name)

    def load_p_cpu(path):
        with open(path, "rb") as f:
            return CPU_Unpickler(f).load()

    def smart_load(path):
        last_err = None
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            last_err = e
        try:
            return load_p_cpu(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {path}\n")

    def infer_mlp_dims_from_state_dict(sd: dict):
        wkeys = []
        for k in sd.keys():
            if k.endswith(".weight"):
                p = k.split(".")[0]
                if p.isdigit():
                    wkeys.append((int(p), k))
        wkeys = sorted(wkeys, key=lambda x: x[0])
        if len(wkeys) == 0:
            raise ValueError("Could not infer MLP dims from state_dict.")
        W0 = sd[wkeys[0][1]]
        Wlast = sd[wkeys[-1][1]]
        width = int(W0.shape[0])
        d_in  = int(W0.shape[1])
        d_out = int(Wlast.shape[0])
        return d_in, width, d_out

    def build_net(d, width=100):
        return nn.Sequential(
            nn.Linear(d, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, d),
        )

    def load_model_ckpt(path):
        obj = smart_load(path)
        sd = None
        scaler = None

        if isinstance(obj, dict):
            if "model" in obj and isinstance(obj["model"], dict):
                sd = obj["model"].get("state_dict", None)
                scaler = obj["model"].get("transformer", obj["model"].get("scaler", None))
            if sd is None and "state_dict" in obj:
                sd = obj["state_dict"]
            if sd is None and "model_state_dict" in obj:
                sd = obj["model_state_dict"]
            if scaler is None and "transformer" in obj:
                scaler = obj["transformer"]
            if scaler is None and "scaler" in obj:
                scaler = obj["scaler"]
        else:
            sd = obj

        if sd is None:
            raise RuntimeError(f"Could not find usable state_dict in checkpoint: {path}")
        if scaler is None:
            scaler = MaxAbsScaler().fit(np.asarray(train_data, dtype=np.float64))

        return obj, sd, scaler

    @torch.no_grad()
    def rollout_from_x0(net, scaler, x0_raw, K, dt_obs, device):
        x0_s = scaler.transform(np.asarray(x0_raw).reshape(1, -1)).astype(np.float32)
        x = torch.tensor(x0_s, dtype=torch.float32, device=device)

        pred_s = np.zeros((K + 1, x.shape[1]), dtype=np.float32)
        for k in range(K + 1):
            pred_s[k] = x.detach().cpu().numpy()[0]
            x = x + float(dt_obs) * net(x)

        pred = scaler.inverse_transform(pred_s).astype(np.float64)
        return pred

    # ============================================================
    # 3. 加载 NODE 模型
    # ============================================================
    NODE_CKPT_PATH = str(args.checkpoint) # 确保该文件在你的工作目录下

    print(f"Loading NODE model from {NODE_CKPT_PATH}...")
    _, sd_node, scaler_node = load_model_ckpt(NODE_CKPT_PATH)
    d_in_n, width_n, d_out_n = infer_mlp_dims_from_state_dict(sd_node)

    net_node = build_net(d_in_n, width=width_n).to(device)
    net_node.load_state_dict(sd_node)
    net_node.eval()
    print("Model loaded successfully!")

    # ============================================================
    # 4. 随机选取 30 个初始点并进行 Rollout
    # ============================================================
    # 设定随机种子以保证结果可复现
    np.random.seed(13532) 

    # 从前 300 个点中随机抽取 30 个索引 (不重复) ---> 修改点 1
    pool_size = min(300, len(test_data)) 
    selected_indices = np.random.choice(pool_size, 30, replace=False)

    K_STEPS = 1000  # 定义你想 Rollout 推演的步数

    print(f"Starting rollout for {len(selected_indices)} initial conditions for {K_STEPS} steps...")

    # 将列表命名为 res，用于保存 30 条轨迹 ---> 修改点 2
    res = []

    plt.figure(figsize=(12, 6), dpi=120)

    for idx in selected_indices:
        x0 = test_data[idx]
    
        # 执行单步自回归推演
        pred = rollout_from_x0(net_node, scaler_node, x0, K_STEPS, dt_obs, device)
    
        # 保存轨迹到 res 列表中
        res.append(pred)
    
        # 绘制第一维度 (Channel 0) 的轨迹
        time_axis = np.arange(K_STEPS + 1) * dt_obs
        plt.plot(time_axis, pred[:, 0], alpha=0.6, linewidth=1.0)

    # 可视化设置 ---> 修改点 3 (更新了标题)
    plt.title(f"NODE Model: 30 Rollouts from Random Init Points (Indices 0-300)")
    plt.xlabel("Time (t)")
    plt.ylabel("State Value (Channel 0)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'figure_{len(list(args.output_dir.glob("figure_*.png"))):02d}.png', dpi=180, bbox_inches='tight'); plt.close()

    print("Rollout completed.")
    print(f"Total trajectories saved in 'res': {len(res)}")
    print(f"You can access them using res[0] to res[{len(res)-1}]")

    res = np.array(res)

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D  # 用于创建干干净净的自定义图例

    # --- 学术级字体与排版设置 ---
    plt.rcParams.update({
        "text.usetex": False,        
        "mathtext.fontset": "cm",    
        "font.family": "serif",
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "font.size": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": False,
        "grid.alpha": 0.8,
        "grid.linestyle": "--"
    })

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # ==========================================
    # Subplot A: Phase Portrait (拓扑相图)
    # ==========================================
    ax = axes[0]

    # 1. Ground Truth - 底层背景，颜色调柔和
    ax.plot(test_data[:, 0], test_data[:, 1], color='#b0b0b0', alpha=0.3, linewidth=0.8, zorder=1)

    # 2. 严格指定只绘制 5, 7, 10
    indices_to_plot = [5, 7, 10]
    colors = ['#0072B2', '#009E73', '#D55E00'] 

    # 绘制轨迹
    for i, idx in enumerate(indices_to_plot):
        ax.plot(res[idx][:, 0], res[idx][:, 1], color=colors[i], alpha=0.85, linewidth=1.5, zorder=2)

    # 新增/修改：绘制初始点 (t=0) - 去掉黑边，纯实心圆 ('o')
    for i, idx in enumerate(indices_to_plot):
        # 去掉了 edgecolor 和 linewidths，保持纯色实心点
        ax.scatter(res[idx][0, 0], res[idx][0, 1], color=colors[i], marker='o', s=60, zorder=4)

    # 3. 高质感伪吸引子标记 (Spurious Attractor)
    # 使用干净的叉号 ('X')，稍微放大一点以保证辨识度
    attractor_color = '#1a1a1a' # 用极深的颜色代替原来的红色，更显干净
    ax.scatter([8.485], [27], color=attractor_color, marker='x', s=50, zorder=5)
    # 如果左边也有收缩点，替换为：
    # ax.scatter([-8.485, 8.485], [27, 27], color=attractor_color, marker='X', s=100, zorder=5)

    ax.set_title(r"(a) Topological Collapse in Phase Space", pad=15)
    ax.set_xlabel(r"$u_{0}$")
    ax.set_ylabel(r"$u_{2}$")

    legend_elements = [
        Line2D([0], [0], color='#b0b0b0', lw=1.5, alpha=0.6, label='Ground Truth'),
        Line2D([0], [0], marker='x', color='w', markeredgecolor=attractor_color, 
               markersize=9, markeredgewidth=2.0, label='Spurious Attractor'),
        # 将对应的线和实心圆点组合在一起，直观表达“该颜色的轨迹与其初始点”
        Line2D([0], [0], color=colors[0], lw=1.5, marker='o', markersize=6, label='Rollout 5'),
        Line2D([0], [0], color=colors[1], lw=1.5, marker='o', markersize=6, label='Rollout 7'),
        Line2D([0], [0], color=colors[2], lw=1.5, marker='o', markersize=6, label='Rollout 10')
    ]

    # 优化排版：启用双列 (ncol=2)，放置在上方中间 (upper center) 的空白区
    ax.legend(handles=legend_elements, loc='upper center', 
              ncol=2,                 # 分成两列，极大压缩高度
              framealpha=0.9, edgecolor='gray', 
              fontsize=10,            # 略微缩小字体
              labelspacing=0.3,       # 压缩行间距
              handletextpad=0.5,      # 压缩图标和文字的间距
              columnspacing=1.2,      # 两列之间的横向间距
              borderpad=0.4)          # 压缩图例外边距

    ax = axes[1]
    time_axis = np.arange(len(res[0])) * dt_obs

    res_arr = np.array(res)
    max_steps = res_arr.shape[1]
    vol_pred = np.zeros(max_steps)

    for t in range(max_steps):
        pts = res_arr[:, t, :]
        cov_matrix = np.cov(pts, rowvar=False)
        vol_pred[t] = np.linalg.det(cov_matrix)

    vol_pred_safe = np.maximum(vol_pred, 1e-10)

    vol_series = pd.Series(vol_pred_safe)
    window_size = 20 
    vol_smoothed = vol_series.rolling(window=window_size, min_periods=1, center=True).mean()

    theme_color = '#b30000' 
    ax.plot(time_axis, vol_pred_safe, color=theme_color, alpha=0.2, linewidth=1.0, label='Raw Determinant')
    ax.plot(time_axis, vol_smoothed, color=theme_color, alpha=1.0, linewidth=2.5, label='Smoothed Trend')

    ax.set_title(r"(b) Phase Space Volume Contraction", pad=15)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"$\det(\Sigma_t)$ (Log Scale)")
    ax.set_yscale('log')

    ax.legend(loc='best', framealpha=0.95, edgecolor='gray')

    plt.tight_layout()
    plt.savefig(f'figure_{len(list(args.output_dir.glob("figure_*.png"))):02d}.png', dpi=180, bbox_inches='tight'); plt.close()

if __name__ == '__main__':
    main()
