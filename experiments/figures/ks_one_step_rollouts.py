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
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/figures/ks_one_step_rollouts')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--checkpoint', type=lambda s: Path(s).resolve(), default=ROOT/'checkpoints/ks/partial_seed_13531/node_BEST.pt')
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

    import io
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from matplotlib.lines import Line2D

    # ============================================================
    # 1. Fixed user settings
    # ============================================================
    CKPT_PATH = str(args.checkpoint)
    TEST_PATH = str(ROOT/'data/ks/test_full.npy')

    N_ROLLOUTS = 30
    ROLLOUT_STEPS = 2000
    START_SEED = 13531

    # phase portrait channels
    # 你如果想改成别的二维投影，改这里即可
    X_IDX = 0
    Y_IDX = 1
    X_LABEL = r"$u_{0}$"
    Y_LABEL = r"$u_{1}$"

    OUT_PNG = "KS_NODE_rollout_30starts_len1000.png"
    OUT_PDF = "KS_NODE_rollout_30starts_len1000.pdf"

    device = args.device
    print("device:", device)


    # ============================================================
    # 2. Load test data only
    # ============================================================
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Cannot find test data: {TEST_PATH}")

    test_data = np.load(TEST_PATH).astype(np.float64)
    print(f"[OK] loaded test data: {TEST_PATH}, shape={test_data.shape}")

    if test_data.ndim != 2:
        raise ValueError(f"test_full.npy must be 2D, got shape={test_data.shape}")

    T_eval, d_data = test_data.shape
    print("T_eval =", T_eval)
    print("obs_dim =", d_data)

    if T_eval != 2500:
        print(f"[warn] test length is {T_eval}, not 2500. Script will still use actual length.")


    # ============================================================
    # 3. Robust checkpoint loader
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


    def strip_module_prefix(sd):
        if not isinstance(sd, dict):
            return sd
        if len(sd) > 0 and all(k.startswith("module.") for k in sd.keys()):
            return {k.replace("module.", "", 1): v for k, v in sd.items()}
        return sd


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
            raise ValueError("Could not infer MLP dims from state_dict.")

        W0 = sd[wkeys[0][1]]
        Wlast = sd[wkeys[-1][1]]

        d_in = int(W0.shape[1])
        width = int(W0.shape[0])
        d_out = int(Wlast.shape[0])

        return d_in, width, d_out


    def build_net(d, width=100):
        return nn.Sequential(
            nn.Linear(d, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, d),
        )


    def load_node_checkpoint(path):
        obj = smart_load(path)

        sd = None
        scaler = None

        if isinstance(obj, dict):
            if "net_state_dict" in obj:
                sd = obj["net_state_dict"]
            elif "state_dict" in obj:
                sd = obj["state_dict"]
            elif "model_state_dict" in obj:
                sd = obj["model_state_dict"]

            if "scaler" in obj:
                scaler = obj["scaler"]
            elif "transformer" in obj:
                scaler = obj["transformer"]
        else:
            sd = obj

        if sd is None:
            raise RuntimeError(f"Could not find state_dict in checkpoint: {path}")

        sd = strip_module_prefix(sd)

        if scaler is None:
            raise RuntimeError(
                f"Checkpoint {path} does not contain a scaler. "
                "This plotting script does not use train data."
            )

        cfg = obj if isinstance(obj, dict) else {}
        return obj, sd, scaler, cfg


    # ============================================================
    # 4. Load model
    # ============================================================
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Cannot find checkpoint: {CKPT_PATH}")

    obj_node, sd_node, scaler_node, cfg = load_node_checkpoint(CKPT_PATH)

    d_in, width, d_out = infer_mlp_dims_from_state_dict(sd_node)

    if d_in != d_out:
        raise ValueError(f"Model is not square: d_in={d_in}, d_out={d_out}")
    if d_in != d_data:
        raise ValueError(f"Model/data dimension mismatch: model={d_in}, data={d_data}")

    net_node = build_net(d_in, width=width).to(device)
    net_node.load_state_dict(sd_node)
    net_node.eval()

    print("\n==================== CHECKPOINT CONFIG ====================")
    if isinstance(cfg, dict):
        for k in ["seed", "dt_base", "stride", "dt_obs", "Kmax", "width", "obs_dim", "iter", "best_train_loss"]:
            if k in cfg:
                print(f"{k}: {cfg[k]}")
    print("===========================================================")

    dt_base_cfg = float(cfg.get("dt_base", 1.0)) if isinstance(cfg, dict) else 1.0
    stride_cfg = int(cfg.get("stride", 1)) if isinstance(cfg, dict) else 1
    dt_obs_cfg = float(cfg.get("dt_obs", dt_base_cfg * stride_cfg)) if isinstance(cfg, dict) else dt_base_cfg * stride_cfg

    print("\nUsing rollout config:")
    print("dt_base :", dt_base_cfg)
    print("stride  :", stride_cfg)
    print("dt_obs  :", dt_obs_cfg)
    print("rollout length =", ROLLOUT_STEPS)


    # ============================================================
    # 5. Rollout function
    # ============================================================
    @torch.no_grad()
    def rollout_from_x0_ks(net, scaler, x0_raw, K, dt_base, stride, device):
        """
        Same NODE rollout logic:
            for each observation step:
                repeat stride micro-steps:
                    x <- x + dt_base * f_theta(x)
        """
        x0_s = scaler.transform(np.asarray(x0_raw).reshape(1, -1)).astype(np.float32)
        x = torch.tensor(x0_s, dtype=torch.float32, device=device)

        pred_s = np.zeros((K + 1, x.shape[1]), dtype=np.float32)
        pred_s[0] = x.detach().cpu().numpy()[0]

        for k in range(1, K + 1):
            for _ in range(int(stride)):
                x = x + float(dt_base) * net(x)
            pred_s[k] = x.detach().cpu().numpy()[0]

        pred = scaler.inverse_transform(pred_s).astype(np.float64)
        return pred


    # ============================================================
    # 6. Sample 30 random starts
    # ============================================================
    max_start = T_eval - ROLLOUT_STEPS - 1
    if max_start <= 0:
        raise ValueError(
            f"test data too short for rollout length {ROLLOUT_STEPS}. "
            f"T_eval={T_eval}, max_start={max_start}"
        )

    rng = np.random.default_rng(START_SEED)
    starts = np.sort(rng.choice(np.arange(max_start + 1), size=N_ROLLOUTS, replace=False))

    print("\nRandom rollout starts:")
    print(starts)
    print(f"Start range: 0 ~ {max_start}")


    # ============================================================
    # 7. Generate 30 rollouts
    # ============================================================
    res = []

    print("\nGenerating rollouts...")
    for i, s in enumerate(starts):
        pred = rollout_from_x0_ks(
            net=net_node,
            scaler=scaler_node,
            x0_raw=test_data[s],
            K=ROLLOUT_STEPS,
            dt_base=dt_base_cfg,
            stride=stride_cfg,
            device=device,
        )
        res.append(pred)
        print(f"rollout {i+1:02d} | start={s:4d} | shape={pred.shape}")

    res = np.asarray(res, dtype=np.float64)
    print("\nres.shape =", res.shape)


    # ============================================================
    # 8. Plot style (same spirit as Lorenz)
    # ============================================================
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "font.size": 12,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": False,
        "grid.alpha": 0.8,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    GT_COLOR = "#d0d0d0"

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
        "#AA3377", "#004488",
        "#DDAA33", "#BB5566",
        "#33BBEE", "#EE7733",
        "#33A02C", "#6A3D9A",
    ]


    def rollout_label(ridx):
        return f"Rollout {ridx + 1}"


    def robust_limits(arr_xy, pad_ratio=0.08):
        arr_xy = arr_xy[np.isfinite(arr_xy).all(axis=1)]
        if len(arr_xy) == 0:
            return (-1, 1), (-1, 1)

        # 用分位数，避免单个爆炸点把图撑得完全看不见
        x_low, y_low = np.percentile(arr_xy, 1, axis=0)
        x_high, y_high = np.percentile(arr_xy, 99, axis=0)

        if not np.isfinite(x_low) or not np.isfinite(x_high) or x_low == x_high:
            x_low, y_low = np.min(arr_xy, axis=0)
            x_high, y_high = np.max(arr_xy, axis=0)

        if not np.isfinite(y_low) or not np.isfinite(y_high) or y_low == y_high:
            x_low, y_low = np.min(arr_xy, axis=0)
            x_high, y_high = np.max(arr_xy, axis=0)

        xpad = pad_ratio * max(1e-8, x_high - x_low)
        ypad = pad_ratio * max(1e-8, y_high - y_low)

        return (x_low - xpad, x_high + xpad), (y_low - ypad, y_high + ypad)


    def draw_two_rollouts(ax, ridx_a, ridx_b):
        handles = [
            Line2D([0], [0], color=GT_COLOR, lw=1.4, alpha=0.8, label="Ground Truth")
        ]

        # 灰色背景：整个 test trajectory
        ax.plot(
            test_data[:, X_IDX],
            test_data[:, Y_IDX],
            color=GT_COLOR,
            alpha=0.35,
            linewidth=0.7,
            zorder=1
        )

        panel_xy = [test_data[:, [X_IDX, Y_IDX]]]

        for ridx in [ridx_a, ridx_b]:
            color = COLOR_POOL[ridx % len(COLOR_POOL)]
            traj = res[ridx]

            ax.plot(
                traj[:, X_IDX],
                traj[:, Y_IDX],
                color=color,
                alpha=0.95,
                linewidth=1.4,
                zorder=2
            )
            ax.scatter(
                traj[0, X_IDX],
                traj[0, Y_IDX],
                color=color,
                s=34,
                marker='o',
                zorder=4
            )

            handles.append(
                Line2D([0], [0], color=color, lw=1.4, marker='o',
                       markersize=5, label=rollout_label(ridx))
            )

            # 为了看出是否爆炸，panel limit 同时考虑 rollout
            finite_mask = np.isfinite(traj[:, [X_IDX, Y_IDX]]).all(axis=1)
            if np.any(finite_mask):
                panel_xy.append(traj[finite_mask][:, [X_IDX, Y_IDX]])

        panel_xy = np.concatenate(panel_xy, axis=0)
        xlim, ylim = robust_limits(panel_xy)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        ax.set_title(f"{rollout_label(ridx_a)} & {rollout_label(ridx_b)}", pad=6, fontsize=11)

        ax.legend(
            handles=handles,
            loc="upper right",
            framealpha=0.92,
            edgecolor='gray',
            fontsize=7.3,
            labelspacing=0.25,
            handletextpad=0.45,
            borderpad=0.35
        )


    # ============================================================
    # 9. Draw 3 x 5 figure
    # ============================================================
    fig, axes = plt.subplots(
        3, 5,
        figsize=(22, 12),
        dpi=300,
    )

    axes = axes.ravel()

    for p in range(15):
        ax = axes[p]
        ridx_a = 2 * p
        ridx_b = 2 * p + 1
        draw_two_rollouts(ax, ridx_a, ridx_b)

    for r in range(3):
        axes[r * 5].set_ylabel(Y_LABEL, fontsize=12)

    for c in range(5):
        axes[10 + c].set_xlabel(X_LABEL, fontsize=12)

    plt.tight_layout()

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")

    print("\nSaved:")
    print(" ", OUT_PNG)
    print(" ", OUT_PDF)

    plt.savefig(f'figure_{len(list(args.output_dir.glob("figure_*.png"))):02d}.png', dpi=180, bbox_inches='tight'); plt.close()

if __name__ == '__main__':
    main()
