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
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/figures/lorenz_phase_return')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--prime', type=lambda s: Path(s).resolve(), required=True)
    p.add_argument('--dim', type=lambda s: Path(s).resolve(), required=True)
    p.add_argument('--dim-mse', type=lambda s: Path(s).resolve(), required=True)
    p.add_argument('--mse', type=lambda s: Path(s).resolve(), required=True)
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
    train_data = np.load(ROOT/'data/lorenz63/train_data.npy')
    test_data = np.load(ROOT/'data/lorenz63/test_data.npy')
    import io
    import os
    import pickle
    import re
    from collections import OrderedDict

    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import torch
    import torch.nn as nn
    from scipy.signal import find_peaks
    from sklearn.preprocessing import MaxAbsScaler


    # ============================================================
    # 0. 全局风格（paper-like）
    # ============================================================
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


    # ============================================================
    # 1. 基础设置
    # ============================================================
    PRIME_CKPT_PATH = str(args.prime)
    DIM_CKPT_PATH = str(args.dim)
    MIXED_CKPT_PATH = str(args.dim_mse)
    MSE_CKPT_PATH = str(args.mse)

    dt_obs = 0.1
    DEVICE = args.device
    print(f"Device: {DEVICE}")

    # 假设 train_data 和 test_data 已在全局环境中
    assert train_data.ndim == 2, "请确保 train_data 已加载"
    assert test_data.ndim == 2, "请确保 test_data 已加载"

    train_arr = np.asarray(train_data, dtype=np.float64)
    test_arr  = np.asarray(test_data, dtype=np.float64)

    T_eval = int(len(test_arr))
    ref = test_arr[:T_eval].astype(np.float64)
    KMAX = T_eval - 1

    # ------------------------------------------------------------
    # 坐标索引
    # 如果数据是二维观测 (u0, u2)，则 X_IDX=0, Z_IDX=1
    # 如果数据是完整 Lorenz (x,y,z)，且要画 x-z，则设 X_IDX=0, Z_IDX=2
    # ------------------------------------------------------------
    X_IDX = 0
    Z_IDX = 1

    # A图（unimodal map）的坐标标签
    RETURN_XLABEL = r"$u_{2,n}$"
    RETURN_YLABEL = r"$u_{2,n+1}$"

    # B图（phase portrait）的坐标标签
    PHASE_XLABEL = r"$u_0$"
    PHASE_YLABEL = r"$u_2$"


    # ============================================================
    # 2. 颜色与绘图参数
    # ============================================================
    # --- 你要求的第3点：GT更深，prediction更透明 ---
    GT_COLOR = "#8D97A6"       # 比之前更深一点的灰蓝色
    PRED_COLOR = "#F28E2B"     # 橙色

    GT_ALPHA_RETURN = 0.62
    PRED_ALPHA_RETURN = 0.58

    GT_ALPHA_PHASE = 0.50
    PRED_ALPHA_PHASE = 0.60

    GT_MS_RETURN = 4.5
    PRED_MS_RETURN = 5.2

    GT_MS_PHASE = 4.3
    PRED_MS_PHASE = 4.5

    # explicit section 线
    SECTION_COLOR = "#7A869A"
    SECTION_ALPHA = 0.50
    SECTION_LW = 0.85
    SECTION_LS = (0, (4, 3))

    # A图 y=x 参考线
    DIAG_COLOR = "#F8F4F4"
    DIAG_ALPHA = 0.42
    DIAG_LW = 0.75
    DIAG_LS = "--"

    # 是否保存图
    SAVE_FIG = False
    OUT_PDF = "combined_return_phase_plot.pdf"
    OUT_PNG = "combined_return_phase_plot.png"
    DPI = 300

    # 图大小
    FIGSIZE = (13.8, 6.2)

    # B图 section 位置（你要求的第1点）
    SECTION_X_VALUES = [10.0]
    SECTION_Z_VALUES = [40.0]

    # B图展示前多少个点
    PLOT_N_PHASE = 1500

    # A图 peak detection
    PEAK_PROMINENCE = 1.0
    PEAK_DISTANCE = None


    # ============================================================
    # 3. 安全加载工具
    # ============================================================
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: _torch_load_cpu(io.BytesIO(b))
            return super().find_class(module, name)


    def _torch_load_cpu(path_or_buffer):
        try:
            return torch.load(path_or_buffer, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path_or_buffer, map_location="cpu")


    def smart_load(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            return _torch_load_cpu(path)
        except Exception as torch_err:
            try:
                with open(path, "rb") as f:
                    return CPU_Unpickler(f).load()
            except Exception as pickle_err:
                raise RuntimeError(
                    f"Failed to load checkpoint: {path}\n"
                    f"torch.load error: {torch_err}\n"
                    f"pickle error: {pickle_err}"
                )


    def _is_state_dict_like(obj):
        return isinstance(obj, dict) and any(
            isinstance(v, (torch.Tensor, np.ndarray)) for v in obj.values()
        )


    def extract_state_dict_and_scaler(obj):
        sd = None
        scaler = None

        if isinstance(obj, dict):
            # nested format
            if "model" in obj and isinstance(obj["model"], dict):
                sd = obj["model"].get("state_dict", None)
                scaler = obj["model"].get("transformer", obj["model"].get("scaler", None))

            # flat format
            if sd is None:
                sd = obj.get("state_dict", obj.get("model_state_dict", None))

            if scaler is None:
                scaler = obj.get("transformer", obj.get("scaler", None))

            # maybe checkpoint itself is state dict
            if sd is None and _is_state_dict_like(obj):
                sd = obj
        else:
            sd = obj

        if sd is None:
            raise RuntimeError("Could not extract a valid state_dict from checkpoint.")

        if scaler is None:
            scaler = MaxAbsScaler().fit(train_arr)

        return sd, scaler


    def infer_mlp_dims_from_state_dict(sd):
        wkeys = []
        for k, v in sd.items():
            if k.endswith(".weight"):
                parts = k.split(".")
                if parts[0].isdigit() and isinstance(v, torch.Tensor) and v.ndim == 2:
                    wkeys.append((int(parts[0]), k))
        wkeys = sorted(wkeys, key=lambda x: x[0])

        if len(wkeys) < 2:
            raise RuntimeError("Cannot infer MLP dimensions from state dict.")

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


    def load_model_ckpt(path):
        obj = smart_load(path)
        sd, scaler = extract_state_dict_and_scaler(obj)
        return sd, scaler


    @torch.no_grad()
    def rollout_from_x0(net, scaler, x0_raw, K, dt_obs, device):
        x0_raw = np.asarray(x0_raw, dtype=np.float64).reshape(1, -1)
        x0_s = scaler.transform(x0_raw).astype(np.float32)
        x = torch.tensor(x0_s, dtype=torch.float32, device=device)

        pred_s = np.zeros((K + 1, x.shape[1]), dtype=np.float32)
        for k in range(K + 1):
            pred_s[k] = x.detach().cpu().numpy()[0]
            x = x + float(dt_obs) * net(x)

        return scaler.inverse_transform(pred_s).astype(np.float64)


    # ============================================================
    # 4. 读取模型并 rollout
    # ============================================================
    CKPT_PATHS = OrderedDict([
        ("DIM", DIM_CKPT_PATH),
        ("MSE", MSE_CKPT_PATH),
        ("DIM + MSE", MIXED_CKPT_PATH),
        ("PRIME", PRIME_CKPT_PATH),
    ])

    MODEL_ORDER = ["DIM", "MSE", "DIM + MSE", "PRIME"]

    models_pred = OrderedDict()

    print("\n--- Rolling out models ---")
    for name, path in CKPT_PATHS.items():
        print(f"Loading and rolling out: {name} ...")
        try:
            sd, scaler = load_model_ckpt(path)
            d_in, width, d_out = infer_mlp_dims_from_state_dict(sd)

            if d_in != ref.shape[1] or d_out != ref.shape[1]:
                raise RuntimeError(
                    f"[{name}] checkpoint dim mismatch: "
                    f"d_in={d_in}, d_out={d_out}, data_dim={ref.shape[1]}"
                )

            net = build_net(d_in, width=width).to(DEVICE)
            net.load_state_dict(sd)
            net.eval()

            pred = rollout_from_x0(net, scaler, ref[0], KMAX, dt_obs, DEVICE)
            models_pred[name] = pred
        except Exception as e:
            print(f"[Warning] Could not process {name}: {e}")

    if len(models_pred) == 0:
        raise RuntimeError("No models were successfully loaded and rolled out.")


    # ============================================================
    # 5. Unimodal map 提取
    # ============================================================
    def extract_unimodal_map(trajectory, z_idx=1, prominence=1.0, distance=None):
        z_series = np.asarray(trajectory[:, z_idx], dtype=np.float64)

        kwargs = {"prominence": prominence}
        if distance is not None:
            kwargs["distance"] = int(distance)

        peaks, _ = find_peaks(z_series, **kwargs)
        z_peaks = z_series[peaks]

        if len(z_peaks) >= 2:
            return z_peaks[:-1], z_peaks[1:]
        else:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)


    zn_gt, zn1_gt = extract_unimodal_map(
        ref, z_idx=Z_IDX, prominence=PEAK_PROMINENCE, distance=PEAK_DISTANCE
    )

    return_maps = OrderedDict()
    for name in MODEL_ORDER:
        if name not in models_pred:
            continue
        zn_pred, zn1_pred = extract_unimodal_map(
            models_pred[name],
            z_idx=Z_IDX,
            prominence=PEAK_PROMINENCE,
            distance=PEAK_DISTANCE
        )
        return_maps[name] = (zn_pred, zn1_pred)


    # ============================================================
    # 6. 共享坐标范围
    # ============================================================
    def finite_xy_limits(arrays, pad_frac=0.06, min_pad=1.0):
        valid = []
        for arr in arrays:
            arr = np.asarray(arr, dtype=np.float64)
            if arr.size == 0:
                continue
            arr = arr[np.all(np.isfinite(arr), axis=1)]
            if len(arr):
                valid.append(arr)

        if not valid:
            return (-1.0, 1.0), (-1.0, 1.0)

        all_xy = np.concatenate(valid, axis=0)
        mn = np.min(all_xy, axis=0)
        mx = np.max(all_xy, axis=0)
        span = np.maximum(mx - mn, min_pad)
        pad = pad_frac * span
        return (mn[0] - pad[0], mx[0] + pad[0]), (mn[1] - pad[1], mx[1] + pad[1])


    def square_limits(xlim, ylim):
        xmid = 0.5 * (xlim[0] + xlim[1])
        ymid = 0.5 * (ylim[0] + ylim[1])
        half = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        return (xmid - half, xmid + half), (ymid - half, ymid + half)


    # A图 limits
    return_arrays = [np.column_stack([zn_gt, zn1_gt])]
    for name in return_maps:
        znp, zn1p = return_maps[name]
        return_arrays.append(np.column_stack([znp, zn1p]))

    return_xlim, return_ylim = finite_xy_limits(return_arrays, pad_frac=0.07)
    return_xlim, return_ylim = square_limits(return_xlim, return_ylim)

    # B图 limits
    N_phase = min([PLOT_N_PHASE, len(ref)] + [len(models_pred[m]) for m in models_pred])
    gt_phase = ref[:N_phase]

    phase_arrays = [gt_phase[:, [X_IDX, Z_IDX]]]
    for m in models_pred:
        phase_arrays.append(models_pred[m][:N_phase, [X_IDX, Z_IDX]])

    phase_xlim, phase_ylim = finite_xy_limits(phase_arrays, pad_frac=0.06)


    # ============================================================
    # 7. 绘图辅助
    # ============================================================
    def style_axes(ax, use_grid=False):
        ax.spines["left"].set_color("#222222")
        ax.spines["bottom"].set_color("#222222")
        ax.spines["left"].set_linewidth(0.75)
        ax.spines["bottom"].set_linewidth(0.75)

        ax.tick_params(
            axis="both",
            which="both",
            direction="out",
            colors="#222222",
            length=2.6,
            pad=2.0,
        )

        if use_grid:
            ax.grid(
                True,
                linestyle=(0, (2, 2)),
                linewidth=0.45,
                color="#D6DADE",
                alpha=0.65,
                zorder=0,
            )
        else:
            ax.grid(False)


    def draw_sections(ax):
        # 你要求的第1点：B图加 explicit section 虚线
        for xv in SECTION_X_VALUES:
            ax.axvline(
                xv,
                color=SECTION_COLOR,
                linestyle=SECTION_LS,
                linewidth=SECTION_LW,
                alpha=SECTION_ALPHA,
                zorder=0
            )

        for zv in SECTION_Z_VALUES:
            ax.axhline(
                zv,
                color=SECTION_COLOR,
                linestyle=SECTION_LS,
                linewidth=SECTION_LW,
                alpha=SECTION_ALPHA,
                zorder=0
            )


    def draw_return_panel(ax, model_name):
        zn_pred, zn1_pred = return_maps[model_name]

        # Ground truth
        ax.scatter(
            zn_gt, zn1_gt,
            s=GT_MS_RETURN,
            c=GT_COLOR,
            alpha=GT_ALPHA_RETURN,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

        # Prediction
        ax.scatter(
            zn_pred, zn1_pred,
            s=PRED_MS_RETURN,
            c=PRED_COLOR,
            alpha=PRED_ALPHA_RETURN,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

        # 你要求的第2点：A图加 y=x 参考线
        lo = min(return_xlim[0], return_ylim[0])
        hi = max(return_xlim[1], return_ylim[1])
        ax.plot(
            [lo, hi], [lo, hi],
            color=DIAG_COLOR,
            lw=DIAG_LW,
            ls=DIAG_LS,
            alpha=DIAG_ALPHA,
            zorder=0,
        )

        ax.set_title(model_name, pad=3.0)
        ax.set_xlim(*return_xlim)
        ax.set_ylim(*return_ylim)
        ax.set_aspect("equal", adjustable="box")
        style_axes(ax, use_grid=True)


    def draw_phase_panel(ax, model_name):
        pred = models_pred[model_name][:N_phase]

        draw_sections(ax)

        ax.scatter(
            gt_phase[:, X_IDX], gt_phase[:, Z_IDX],
            s=GT_MS_PHASE,
            c=GT_COLOR,
            alpha=GT_ALPHA_PHASE,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

        ax.scatter(
            pred[:, X_IDX], pred[:, Z_IDX],
            s=PRED_MS_PHASE,
            c=PRED_COLOR,
            alpha=PRED_ALPHA_PHASE,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

        ax.set_title(model_name, pad=3.0)
        ax.set_xlim(*phase_xlim)
        ax.set_ylim(*phase_ylim)
        style_axes(ax, use_grid=False)


    def add_group_axis_labels(fig, axes_group, xlabel, ylabel):
        # 你要求的第4点：把 A/B 的 group axis labels 居中
        boxes = [ax.get_position() for ax in axes_group if ax.get_visible()]
        x0 = min(b.x0 for b in boxes)
        x1 = max(b.x1 for b in boxes)
        y0 = min(b.y0 for b in boxes)
        y1 = max(b.y1 for b in boxes)

        fig.text(
            0.5 * (x0 + x1),
            y0 - 0.055,
            xlabel,
            ha="center",
            va="top",
            fontsize=9.0,
        )

        fig.text(
            x0 - 0.045,
            0.5 * (y0 + y1),
            ylabel,
            ha="center",
            va="center",
            rotation=90,
            fontsize=9.0,
        )

    def add_group_title_below(fig, axes_group, title, dy=0.105):
        """
        Put group title below each 2x2 block.
        dy controls vertical distance below the bottom of the axes block.
        Larger dy means lower title.
        """
        boxes = [ax.get_position() for ax in axes_group if ax.get_visible()]
        x0 = min(b.x0 for b in boxes)
        x1 = max(b.x1 for b in boxes)
        y0 = min(b.y0 for b in boxes)

        fig.text(
            0.5 * (x0 + x1),
            y0 - dy,
            title,
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="bold",
        )

    # ============================================================
    # 8. 组合绘图
    # ============================================================
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)

    outer = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.0, 1.28],
        left=0.055,
        right=0.992,
        bottom=0.165,
        top=0.885,
        wspace=0.17,
    )

    # A: unimodal map (2x2)
    gs_a = outer[0].subgridspec(2, 2, wspace=0.14, hspace=0.22)

    # B: phase portrait (2x2)
    gs_b = outer[1].subgridspec(2, 2, wspace=0.09, hspace=0.14)

    axes_a = []
    axes_b = []

    for i, model_name in enumerate(MODEL_ORDER):
        r, c = divmod(i, 2)

        ax_a = fig.add_subplot(gs_a[r, c])
        ax_b = fig.add_subplot(gs_b[r, c])

        axes_a.append(ax_a)
        axes_b.append(ax_b)

        if model_name in return_maps:
            draw_return_panel(ax_a, model_name)
        else:
            ax_a.set_visible(False)

        if model_name in models_pred:
            draw_phase_panel(ax_b, model_name)
        else:
            ax_b.set_visible(False)

    # 隐藏内部重复 tick label
    for axes_group in [axes_a, axes_b]:
        for idx, ax in enumerate(axes_group):
            if not ax.get_visible():
                continue
            r, c = divmod(idx, 2)
            if r == 0:
                ax.tick_params(labelbottom=False)
            if c == 1:
                ax.tick_params(labelleft=False)

    # 组标签（居中）
    add_group_axis_labels(fig, axes_a, RETURN_XLABEL, RETURN_YLABEL)
    add_group_axis_labels(fig, axes_b, PHASE_XLABEL, PHASE_YLABEL)

    # panel 标题放到每个大图下面
    add_group_title_below(
        fig,
        axes_a,
        "(a) Unimodal return map",
        dy=0.105,
    )

    add_group_title_below(
        fig,
        axes_b,
        "(b) Unrolled trajectory",
        dy=0.105,
    )

    # 全局 legend
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o", linestyle="None",
            markersize=4.5,
            markerfacecolor=GT_COLOR,
            markeredgewidth=0,
            alpha=0.80,
            label="Ground truth",
        ),
        Line2D(
            [0], [0],
            marker="o", linestyle="None",
            markersize=4.5,
            markerfacecolor=PRED_COLOR,
            markeredgewidth=0,
            alpha=0.75,
            label="Prediction",
        ),
        Line2D(
            [0], [0],
            color=SECTION_COLOR,
            linestyle=SECTION_LS,
            linewidth=1.0,
            alpha=SECTION_ALPHA,
            label="Explicit sections",
        ),
        Line2D(
            [0], [0],
            color=DIAG_COLOR,
            linestyle=DIAG_LS,
            linewidth=1.0,
            alpha=DIAG_ALPHA,
            label=r"",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.57, 0.988),
        ncol=4,
        frameon=False,
        handlelength=1.9,
        columnspacing=1.3,
    )

    if SAVE_FIG:
        fig.savefig(OUT_PDF, bbox_inches="tight")
        fig.savefig(OUT_PNG, bbox_inches="tight", dpi=DPI)
        print(f"Saved: {OUT_PDF}")
        print(f"Saved: {OUT_PNG}")

    plt.savefig(f'figure_{len(list(args.output_dir.glob("figure_*.png"))):02d}.png', dpi=180, bbox_inches='tight'); plt.close()

if __name__ == '__main__':
    main()
