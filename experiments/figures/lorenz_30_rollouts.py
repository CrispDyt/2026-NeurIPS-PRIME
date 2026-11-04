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
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/figures/lorenz_30_rollouts')
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
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MaxAbsScaler
    from matplotlib.lines import Line2D

    # ============================================================
    # 1. Load data
    # ============================================================
    train_data = np.load(str(ROOT/'data/lorenz63/train_data.npy'))
    test_data  = np.load(str(ROOT/'data/lorenz63/test_data.npy'))

    train_data = np.asarray(train_data, dtype=np.float64)
    test_data  = np.asarray(test_data, dtype=np.float64)

    device = args.device
    print("device:", device)
    print("train:", train_data.shape)
    print("test :", test_data.shape)

    # ============================================================
    # 2. Checkpoint path
    # ============================================================
    # 你说的是 NODE.p，所以优先读 NODE.p；如果没有，再 fallback 到 NODE_BEST.p
    NODE_CKPT_PATH = str(args.checkpoint)

    if not os.path.exists(NODE_CKPT_PATH):
        if os.path.exists(str(args.checkpoint)):
            NODE_CKPT_PATH = str(args.checkpoint)
        else:
            raise FileNotFoundError(
                "Cannot find NODE checkpoint. Expected NODE.p or NODE_BEST.p in current directory."
            )

    print("Using NODE checkpoint:", NODE_CKPT_PATH)

    # ============================================================
    # 3. CPU-safe loader
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
            raise RuntimeError(
                f"Failed to load checkpoint: {path}\n"
                f"torch.load error: {last_err}\n"
                f"pickle error: {e}"
            )


    # ============================================================
    # 4. Config / model helpers
    # ============================================================
    def strip_module_prefix(sd):
        if not isinstance(sd, dict):
            return sd
        if all(k.startswith("module.") for k in sd.keys()):
            return {k.replace("module.", "", 1): v for k, v in sd.items()}
        return sd


    def extract_config(obj):
        """
        Robustly extract config from checkpoint.
        """
        cfg = {}

        if isinstance(obj, dict):
            for key in ["config", "cfg", "args", "hyperparams", "params"]:
                if key in obj and isinstance(obj[key], dict):
                    cfg.update(obj[key])

            for key in [
                "dt_obs", "dt_base", "sub",
                "rollout_len", "K_STEPS", "KMAX", "T_eval",
                "seed", "eval_seed", "rollout_seed",
                "n_rollouts", "N_ROLLOUTS",
                "pool_size", "start_pool_size",
            ]:
                if key in obj and key not in cfg:
                    cfg[key] = obj[key]

        return cfg


    def get_cfg(cfg, keys, default):
        for k in keys:
            if k in cfg and cfg[k] is not None:
                return cfg[k]
        return default


    def infer_mlp_dims_from_state_dict(sd: dict):
        sd = strip_module_prefix(sd)

        wkeys = []
        for k in sd.keys():
            if k.endswith(".weight"):
                prefix = k.split(".")[0]
                if prefix.isdigit():
                    wkeys.append((int(prefix), k))

        wkeys = sorted(wkeys, key=lambda x: x[0])

        if len(wkeys) == 0:
            raise ValueError("Could not infer MLP dims from state_dict. Check checkpoint format.")

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

            if sd is None and "net_state_dict" in obj:
                sd = obj["net_state_dict"]

            if scaler is None and "transformer" in obj:
                scaler = obj["transformer"]

            if scaler is None and "scaler" in obj:
                scaler = obj["scaler"]
        else:
            sd = obj

        if sd is None:
            raise RuntimeError(f"Could not find usable state_dict in checkpoint: {path}")

        sd = strip_module_prefix(sd)

        if scaler is None:
            print("[warn] checkpoint has no scaler; fitting MaxAbsScaler on train_data.")
            scaler = MaxAbsScaler().fit(train_data)

        cfg = extract_config(obj)

        return obj, sd, scaler, cfg


    @torch.no_grad()
    def rollout_from_x0_config(net, scaler, x0_raw, K, dt_base, sub, device):
        """
        One observed step:
            repeat sub micro-steps:
                x <- x + dt_base * f_theta(x)

        If sub=1 and dt_base=dt_obs, this is:
            x <- x + dt_obs * f_theta(x)
        """
        x0_s = scaler.transform(np.asarray(x0_raw).reshape(1, -1)).astype(np.float32)
        x = torch.tensor(x0_s, dtype=torch.float32, device=device)

        pred_s = np.zeros((K + 1, x.shape[1]), dtype=np.float32)

        for k in range(K + 1):
            pred_s[k] = x.detach().cpu().numpy()[0]

            if k == K:
                break

            for _ in range(int(sub)):
                x = x + float(dt_base) * net(x)

        pred = scaler.inverse_transform(pred_s).astype(np.float64)
        return pred


    # ============================================================
    # 5. Load NODE model and config
    # ============================================================
    obj_node, sd_node, scaler_node, cfg = load_model_ckpt(NODE_CKPT_PATH)

    print("\n==================== CHECKPOINT CONFIG ====================")
    if len(cfg) == 0:
        print("[warn] No config found in checkpoint. Using fallback defaults.")
    else:
        for k in sorted(cfg.keys()):
            print(f"{k}: {cfg[k]}")
    print("===========================================================")

    d_in, width, d_out = infer_mlp_dims_from_state_dict(sd_node)

    if d_in != d_out:
        raise ValueError(f"Model is not square: d_in={d_in}, d_out={d_out}")

    net_node = build_net(d_in, width=width).to(device)
    net_node.load_state_dict(sd_node)
    net_node.eval()

    print("\nModel loaded successfully.")
    print(f"d_in={d_in}, width={width}, d_out={d_out}")

    # ============================================================
    # 6. Rollout settings from config
    # ============================================================
    dt_obs_cfg  = float(get_cfg(cfg, ["dt_obs", "dt", "dt_data"], 0.1))
    dt_base_cfg = float(get_cfg(cfg, ["dt_base", "dt_integrator"], dt_obs_cfg))

    sub_cfg = int(get_cfg(cfg, ["sub"], int(round(dt_obs_cfg / dt_base_cfg))))
    if sub_cfg < 1:
        sub_cfg = 1

    N_ROLLOUTS = int(get_cfg(cfg, ["n_rollouts", "N_ROLLOUTS"], 30))
    K_STEPS    = int(get_cfg(cfg, ["K_STEPS", "rollout_len", "KMAX", "T_eval"], 1000))

    START_SEED = int(get_cfg(cfg, ["rollout_seed", "eval_seed", "seed"], 13532))

    START_POOL_SIZE = int(get_cfg(cfg, ["start_pool_size", "pool_size"], 300))
    START_POOL_SIZE = min(START_POOL_SIZE, len(test_data))

    print("\n==================== ROLLOUT CONFIG USED ====================")
    print("dt_obs        :", dt_obs_cfg)
    print("dt_base       :", dt_base_cfg)
    print("sub           :", sub_cfg)
    print("N_ROLLOUTS    :", N_ROLLOUTS)
    print("K_STEPS       :", K_STEPS)
    print("START_SEED    :", START_SEED)
    print("START_POOL    :", START_POOL_SIZE)
    print("=============================================================")

    # ============================================================
    # 7. Select 30 initial conditions
    # ============================================================
    rng = np.random.default_rng(START_SEED)

    if START_POOL_SIZE < N_ROLLOUTS:
        raise ValueError(
            f"START_POOL_SIZE={START_POOL_SIZE} < N_ROLLOUTS={N_ROLLOUTS}. "
            f"Increase START_POOL_SIZE or reduce N_ROLLOUTS."
        )

    selected_indices = rng.choice(START_POOL_SIZE, size=N_ROLLOUTS, replace=False)
    selected_indices = np.asarray(selected_indices, dtype=int)

    # 为了图中顺序稳定，按 test index 排序
    selected_indices = np.sort(selected_indices)

    print("\nSelected test indices:")
    print(selected_indices)

    # ============================================================
    # 8. Generate rollouts
    # ============================================================
    res = []

    print(f"\nStarting {N_ROLLOUTS} rollouts, each K={K_STEPS} steps...")

    for j, idx in enumerate(selected_indices):
        x0 = test_data[idx]

        pred = rollout_from_x0_config(
            net=net_node,
            scaler=scaler_node,
            x0_raw=x0,
            K=K_STEPS,
            dt_base=dt_base_cfg,
            sub=sub_cfg,
            device=device,
        )

        res.append(pred)

        print(f"  rollout {j:02d} | start index = {idx:4d} | pred shape = {pred.shape}")

    res = np.asarray(res, dtype=np.float64)

    print("\nRollout completed.")
    print("res shape:", res.shape)

    # ============================================================
    # 9. Plot style
    # ============================================================
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "font.size": 12,
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": False,
        "grid.alpha": 0.8,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    GT_COLOR = "#b0b0b0"

    COLOR_POOL = [
        "#0072B2", "#D55E00",
        "#009E73", "#CC79A7",
        "#56B4E9", "#E69F00",
        "#332288", "#AA4499",
        "#44AA99", "#882255",
        "#117733", "#999933",
        "#DDCC77", "#88CCEE",
        "#CC6677", "#6699CC",
        "#661100", "#669900",
        "#AA4466", "#4477AA",
        "#228833", "#EE6677",
        "#AA3377", "#11CBD1",
        "#000000", "#994455",
        "#004488", "#DDAA33",
        "#BB5566", "#33BBEE",
    ]

    # ============================================================
    # 10. Plot config
    # ============================================================
    X_IDX = 0
    Y_IDX = 1

    X_LABEL = r"$u_{0}$"
    Y_LABEL = r"$u_{2}$"

    # Rollout legend label uses rollout order.
    # If you want zero-based labels, set this to 0.
    ROLLOUT_LABEL_BASE = 1

    # full GT background
    gt_bg = test_data

    # ============================================================
    # 11. Shared axis limits
    # ============================================================
    all_xy_parts = [gt_bg[:, [X_IDX, Y_IDX]]]

    for i in range(len(res)):
        # IMPORTANT:
        # Use res[i][:, [X_IDX, Y_IDX]]
        # Do NOT use res[i, :, [X_IDX, Y_IDX]]
        all_xy_parts.append(res[i][:, [X_IDX, Y_IDX]])

    all_xy = np.concatenate(all_xy_parts, axis=0)

    xmin, ymin = np.min(all_xy, axis=0)
    xmax, ymax = np.max(all_xy, axis=0)

    xpad = 0.06 * max(1e-8, xmax - xmin)
    ypad = 0.06 * max(1e-8, ymax - ymin)

    xlim = (xmin - xpad, xmax + xpad)
    ylim = (ymin - ypad, ymax + ypad)

    # 如果你想固定 Lorenz x-z 范围，取消下面两行注释：
    # xlim = (-22, 22)
    # ylim = (0, 52)

    # ============================================================
    # 12. Draw helper
    # ============================================================
    def rollout_label(ridx):
        return f"Rollout {ridx + ROLLOUT_LABEL_BASE}"


    def draw_two_rollouts(ax, idx_a, idx_b):
        # Ground truth background
        gt_line, = ax.plot(
            gt_bg[:, X_IDX],
            gt_bg[:, Y_IDX],
            color=GT_COLOR,
            alpha=0.30,
            linewidth=0.8,
            zorder=1,
            label="Ground Truth",
        )

        handles = [
            Line2D(
                [0], [0],
                color=GT_COLOR,
                lw=1.4,
                alpha=0.55,
                label="Ground Truth",
            )
        ]

        pair = [idx_a, idx_b]

        for ridx in pair:
            if ridx is None or ridx >= len(res):
                continue

            color = COLOR_POOL[ridx % len(COLOR_POOL)]
            traj = res[ridx]

            line, = ax.plot(
                traj[:, X_IDX],
                traj[:, Y_IDX],
                color=color,
                alpha=0.85,
                linewidth=1.45,
                zorder=2,
                label=rollout_label(ridx),
            )

            # Initial point: pure solid circle
            ax.scatter(
                traj[0, X_IDX],
                traj[0, Y_IDX],
                color=color,
                marker="o",
                s=35,
                zorder=4,
            )

            handles.append(
                Line2D(
                    [0], [0],
                    color=color,
                    lw=1.45,
                    marker="o",
                    markersize=5,
                    label=rollout_label(ridx),
                )
            )

        # 不再在 title 或 legend 中标注 fixed point / limit cycle
        ax.set_title(
            f"{rollout_label(idx_a)} & {rollout_label(idx_b)}",
            pad=7,
            fontsize=11,
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        ax.tick_params(axis="both", which="major", labelsize=8.5)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

        # 每个小图自己的 legend
        ax.legend(
            handles=handles,
            loc="upper right",
            framealpha=0.92,
            edgecolor="gray",
            fontsize=7.5,
            labelspacing=0.25,
            handletextpad=0.45,
            borderpad=0.35,
        )


    # ============================================================
    # 13. Figure: 15 panels = 3 rows x 5 columns
    # each panel contains 2 rollouts
    # ============================================================
    n_panels = int(np.ceil(len(res) / 2))

    n_rows = 3
    n_cols = 5

    if n_panels > n_rows * n_cols:
        raise ValueError(
            f"Need {n_panels} panels but only have {n_rows*n_cols}. "
            f"Increase subplot grid."
        )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(22.0, 12.0),
        dpi=300,
        sharex=True,
        sharey=True,
    )

    axes = axes.ravel()

    for p in range(n_rows * n_cols):
        ax = axes[p]

        idx_a = 2 * p
        idx_b = 2 * p + 1

        if idx_a < len(res):
            draw_two_rollouts(ax, idx_a, idx_b)
        else:
            ax.axis("off")
    # outer axis labels
    for r in range(n_rows):
        axes[r * n_cols].set_ylabel(Y_LABEL, fontsize=12)

    for c in range(n_cols):
        axes[(n_rows - 1) * n_cols + c].set_xlabel(X_LABEL, fontsize=12)

    fig.suptitle(
        "",
        y=1.015,
        fontsize=16,
    )

    plt.tight_layout(rect=[0.015, 0.02, 0.985, 0.975])

    # ============================================================
    # 14. Save
    # ============================================================
    out_png = "NODE_30_rollouts_3x5_two_per_panel.png"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    print("\nSaved:")
    print(" ", out_png)

    plt.savefig(f'figure_{len(list(args.output_dir.glob("figure_*.png"))):02d}.png', dpi=180, bbox_inches='tight'); plt.close()

if __name__ == '__main__':
    main()
