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
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/figures/ks_multistep_rollouts')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--checkpoint', type=lambda s: Path(s).resolve(), default=ROOT/'checkpoints/figures/ks_mse20.pt')
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
    # 1. Paths / config
    # ============================================================
    CKPT_PATH = str(args.checkpoint)
    TEST_PATH = str(ROOT/'data/ks/test_full.npy')

    N_ROLLOUTS = 30
    K_STEPS = 2000

    # 你明确说 multi-step 的 stride 是 10
    FORCE_STRIDE = 10

    START_SEED = 13531

    # KS phase projection
    X_IDX = 0
    Y_IDX = 1
    X_LABEL = r"$u_{0}$"
    Y_LABEL = r"$u_{1}$"

    # 如果想看 u0-u2，改成：
    # X_IDX = 0
    # Y_IDX = 2
    # X_LABEL = r"$u_{0}$"
    # Y_LABEL = r"$u_{2}$"

    OUT_PNG = "KS_NODE_MULTI_30_rollouts_len1000_randomstarts.png"
    OUT_PDF = "KS_NODE_MULTI_30_rollouts_len1000_randomstarts.pdf"

    device = args.device
    print("device:", device)


    # ============================================================
    # 2. Load test data only
    # ============================================================
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Cannot find test data: {TEST_PATH}")

    test_data = np.load(TEST_PATH).astype(np.float64)

    T_eval, d_data = test_data.shape
    print(f"[OK] loaded test data: {TEST_PATH}, shape={test_data.shape}")

    if T_eval <= K_STEPS + 1:
        raise ValueError(
            f"test data too short for K_STEPS={K_STEPS}. "
            f"Need T_eval > {K_STEPS + 1}, got T_eval={T_eval}"
        )


    # ============================================================
    # 3. CPU-safe / torch-safe loader
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
            elif "net" in obj and isinstance(obj["net"], dict):
                sd = obj["net"]
            elif "model" in obj and isinstance(obj["model"], dict):
                sd = obj["model"].get("state_dict", None)

            if "scaler" in obj:
                scaler = obj["scaler"]
            elif "transformer" in obj:
                scaler = obj["transformer"]
            elif "model" in obj and isinstance(obj["model"], dict):
                scaler = obj["model"].get("transformer", obj["model"].get("scaler", None))
        else:
            sd = obj

        if sd is None:
            raise RuntimeError(f"Could not find state_dict in checkpoint: {path}")

        sd = strip_module_prefix(sd)

        if scaler is None:
            raise RuntimeError(
                f"Checkpoint {path} does not contain a scaler. "
                "This plotting script does not use train data; please use a checkpoint saved with scaler."
            )

        cfg = obj if isinstance(obj, dict) else {}
        return obj, sd, scaler, cfg


    # ============================================================
    # 4. Load multi-step NODE model
    # ============================================================
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Cannot find checkpoint: {CKPT_PATH}")

    obj_node, sd_node, scaler_node, cfg = load_node_checkpoint(CKPT_PATH)

    print("\n==================== CHECKPOINT CONFIG ====================")
    if isinstance(cfg, dict):
        for k in [
            "seed", "dt_base", "stride", "dt_obs", "tau_pair", "micro_steps",
            "Kmax", "width", "obs_dim", "best_train_loss", "iter"
        ]:
            if k in cfg:
                print(f"{k}: {cfg[k]}")
    print("===========================================================")

    d_in, width, d_out = infer_mlp_dims_from_state_dict(sd_node)

    if d_in != d_out:
        raise ValueError(f"Model is not square: d_in={d_in}, d_out={d_out}")
    if d_in != d_data:
        raise ValueError(f"Model/data dimension mismatch: model={d_in}, data={d_data}")

    net_node = build_net(d_in, width=width).to(device)
    net_node.load_state_dict(sd_node)
    net_node.eval()

    print("\nMulti-step NODE loaded successfully.")
    print(f"d_in={d_in}, width={width}, d_out={d_out}")


    # ============================================================
    # 5. Rollout settings
    # ============================================================
    dt_base_cfg = float(cfg.get("dt_base", 1.0)) if isinstance(cfg, dict) else 1.0

    # 强制用你说的 multi-step stride=10
    stride_cfg = int(FORCE_STRIDE)

    dt_obs_cfg = float(cfg.get("dt_obs", dt_base_cfg * stride_cfg)) if isinstance(cfg, dict) else dt_base_cfg * stride_cfg

    print("\n==================== ROLLOUT CONFIG USED ====================")
    print("checkpoint     :", CKPT_PATH)
    print("dt_base        :", dt_base_cfg)
    print("stride         :", stride_cfg)
    print("dt_obs         :", dt_obs_cfg)
    print("N_ROLLOUTS     :", N_ROLLOUTS)
    print("K_STEPS        :", K_STEPS)
    print("START_SEED     :", START_SEED)
    print("=============================================================")


    @torch.no_grad()
    def rollout_from_x0_ks(net, scaler, x0_raw, K, dt_base, stride, device):
        """
        Multi-step NODE rollout:
            for each observed step:
                repeat stride times:
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
    # 6. Select 30 random starts from full valid range
    # ============================================================
    max_start = T_eval - K_STEPS - 1
    if max_start < 0:
        raise ValueError(f"Invalid max_start={max_start}. Check T_eval and K_STEPS.")

    valid_pool = np.arange(max_start + 1, dtype=int)

    print(f"\nValid start range: 0 ~ {max_start} (total {len(valid_pool)})")

    rng = np.random.default_rng(START_SEED)
    selected_indices = rng.choice(valid_pool, size=N_ROLLOUTS, replace=False)
    selected_indices = np.sort(selected_indices)

    print("\nSelected 30 random start indices:")
    print(selected_indices)


    # ============================================================
    # 7. Generate rollouts
    # ============================================================
    res = []
    gt_segments = []
    diagnostics = []

    print(f"\nStarting {N_ROLLOUTS} KS multi-step NODE rollouts, each K={K_STEPS} steps...")

    for j, idx in enumerate(selected_indices):
        gt_seg = test_data[idx: idx + K_STEPS + 1]

        pred = rollout_from_x0_ks(
            net=net_node,
            scaler=scaler_node,
            x0_raw=test_data[idx],
            K=K_STEPS,
            dt_base=dt_base_cfg,
            stride=stride_cfg,
            device=device,
        )

        res.append(pred)
        gt_segments.append(gt_seg)

        finite_mask = np.isfinite(pred).all()
        pred_norm_end = np.linalg.norm(pred[-1]) if finite_mask else np.inf
        pred_amp = np.nanmax(np.linalg.norm(pred, axis=1)) if finite_mask else np.inf
        gt_amp = np.nanmax(np.linalg.norm(gt_seg, axis=1))
        amp_ratio = pred_amp / max(gt_amp, 1e-12)

        diagnostics.append({
            "rollout_id": j + 1,
            "start_idx": int(idx),
            "finite": bool(finite_mask),
            "pred_end_norm": float(pred_norm_end),
            "pred_max_norm": float(pred_amp),
            "gt_max_norm": float(gt_amp),
            "amp_ratio_vs_gt": float(amp_ratio),
        })

        print(
            f"  rollout {j+1:02d} | start={idx:4d} | finite={finite_mask} "
            f"| end_norm={pred_norm_end:.4g} | amp_ratio={amp_ratio:.4g}"
        )

    res = np.asarray(res, dtype=np.float64)
    gt_segments = np.asarray(gt_segments, dtype=np.float64)

    print("\nRollout completed.")
    print("res shape        :", res.shape)
    print("gt_segments shape:", gt_segments.shape)

    print("\n==================== SIMPLE DIAGNOSTICS ====================")
    for d in diagnostics:
        print(
            f"Rollout {d['rollout_id']:02d} | start={d['start_idx']:4d} | "
            f"finite={d['finite']} | pred_end_norm={d['pred_end_norm']:.4g} | "
            f"pred_max_norm={d['pred_max_norm']:.4g} | gt_max_norm={d['gt_max_norm']:.4g} | "
            f"amp_ratio_vs_gt={d['amp_ratio_vs_gt']:.4g}"
        )
    print("============================================================")


    # ============================================================
    # 8. Plot style
    # ============================================================
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.labelsize": 13,
        "axes.titlesize": 11,
        "font.size": 11,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    GT_BG_COLOR = "#c8c8c8"
    GT_SEG_COLOR = "#6e6e6e"

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

    ROLLOUT_LABEL_BASE = 1

    def rollout_label(ridx):
        return f"Rollout {ridx + ROLLOUT_LABEL_BASE}"


    def safe_xy(arr):
        xy = arr[:, [X_IDX, Y_IDX]]
        xy = xy[np.isfinite(xy).all(axis=1)]
        return xy


    def classify_rollout(pred, gt_seg):
        if not np.isfinite(pred).all():
            return "non-finite"

        pred_norm = np.linalg.norm(pred, axis=1)
        gt_norm = np.linalg.norm(gt_seg, axis=1)

        pred_tail_std = np.std(pred[-200:], axis=0).mean() if len(pred) >= 200 else np.std(pred, axis=0).mean()
        gt_tail_std = np.std(gt_seg[-200:], axis=0).mean() if len(gt_seg) >= 200 else np.std(gt_seg, axis=0).mean()

        pred_max = np.max(pred_norm)
        gt_max = np.max(gt_norm)

        if pred_tail_std < 0.02 * max(gt_tail_std, 1e-12):
            return "collapsed"
        elif pred_max > 5.0 * max(gt_max, 1e-12):
            return "blow-up"
        else:
            return "active"


    def draw_two_rollouts(ax, idx_a, idx_b):
        ax.plot(
            test_data[:, X_IDX],
            test_data[:, Y_IDX],
            color=GT_BG_COLOR,
            alpha=0.26,
            linewidth=0.8,
            zorder=1,
        )

        handles = [
            Line2D([0], [0], color=GT_BG_COLOR, lw=1.2, alpha=0.8, label="GT background"),
            Line2D([0], [0], color=GT_SEG_COLOR, lw=1.6, alpha=0.95, label="GT local"),
        ]

        title_parts = []
        xy_parts = []

        for ridx in [idx_a, idx_b]:
            if ridx is None or ridx >= len(res):
                continue

            color = COLOR_POOL[ridx % len(COLOR_POOL)]
            pred = res[ridx]
            gt_seg = gt_segments[ridx]

            ax.plot(
                gt_seg[:, X_IDX],
                gt_seg[:, Y_IDX],
                color=GT_SEG_COLOR,
                alpha=0.95,
                linewidth=1.55,
                zorder=2,
            )

            ax.plot(
                pred[:, X_IDX],
                pred[:, Y_IDX],
                color=color,
                alpha=0.95,
                linewidth=1.45,
                zorder=3,
            )

            ax.scatter(
                pred[0, X_IDX],
                pred[0, Y_IDX],
                color=color,
                marker="o",
                s=34,
                zorder=5,
            )

            handles.append(
                Line2D(
                    [0], [0],
                    color=color,
                    lw=1.45,
                    marker='o',
                    markersize=4.8,
                    label=rollout_label(ridx),
                )
            )

            cls = classify_rollout(pred, gt_seg)
            title_parts.append(f"{ridx+1}:{cls}")

            pxy = safe_xy(pred)
            gxy = safe_xy(gt_seg)
            if len(pxy) > 0:
                xy_parts.append(pxy)
            if len(gxy) > 0:
                xy_parts.append(gxy)

        if len(xy_parts) > 0:
            xy = np.concatenate(xy_parts, axis=0)
            xmin, ymin = np.min(xy, axis=0)
            xmax, ymax = np.max(xy, axis=0)
        else:
            xmin = ymin = -1
            xmax = ymax = 1

        bg_xy = safe_xy(test_data)
        if len(bg_xy) > 0:
            bgxmin, bgymin = np.percentile(bg_xy, 2, axis=0)
            bgxmax, bgymax = np.percentile(bg_xy, 98, axis=0)
            xmin = min(xmin, bgxmin)
            ymin = min(ymin, bgymin)
            xmax = max(xmax, bgxmax)
            ymax = max(ymax, bgymax)

        dx = max(xmax - xmin, 1e-8)
        dy = max(ymax - ymin, 1e-8)
        ax.set_xlim(xmin - 0.08 * dx, xmax + 0.08 * dx)
        ax.set_ylim(ymin - 0.08 * dy, ymax + 0.08 * dy)

        ax.tick_params(axis="both", which="major", labelsize=8.2)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

        ax.legend(
            handles=handles,
            loc="upper right",
            framealpha=0.92,
            edgecolor="gray",
            fontsize=7.0,
            labelspacing=0.22,
            handletextpad=0.4,
            borderpad=0.3,
        )


    # ============================================================
    # 9. Figure: 15 panels = 3 rows x 5 columns
    # ============================================================
    n_panels = int(np.ceil(len(res) / 2))
    n_rows = 3
    n_cols = 5

    if n_panels > n_rows * n_cols:
        raise ValueError(f"Need {n_panels} panels but only have {n_rows*n_cols}.")

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(22.0, 12.0),
        dpi=300,
        sharex=False,
        sharey=False,
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

    for r in range(n_rows):
        axes[r * n_cols].set_ylabel(Y_LABEL, fontsize=12)

    for c in range(n_cols):
        axes[(n_rows - 1) * n_cols + c].set_xlabel(X_LABEL, fontsize=12)

    plt.tight_layout(rect=[0.015, 0.02, 0.985, 0.98])

    print("\nSaved:")

    plt.savefig(f'figure_{len(list(args.output_dir.glob("figure_*.png"))):02d}.png', dpi=180, bbox_inches='tight'); plt.close()

if __name__ == '__main__':
    main()
