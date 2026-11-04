#!/usr/bin/env python3
"""Cylinder forecast/sensor panels from Comparison of Cylinder CFD.ipynb.

The latest cells17/18 use Seed100, probe0, a start-zero 100-time-unit rollout,
with the final20 units displayed. Exact correspondence to the published Figure3
image is unverified; selected settings and file hashes are written to a manifest.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SENSORS = np.array([[62, 12], [52, 44], [77, 15], [43, 35], [74, 35], [109, 40], [125, 15], [80, 35], [100, 20]])
COLORS = {"Ground Truth": "#A9A9A9", "PRIME": "#0000FF", "DIM+MSE": "#FFB74D", "DIM": "#4DBBD5", "MSE": "#8FBC8F"}
FILES = {"PRIME": "PRIME.p", "MSE": "MSE.p", "DIM": "DIM.p", "DIM+MSE": "DIM_MSE.p"}


def plot_forecast(ax, times, trajectories, channel, window, zoom):
    selected = (times >= window[0]) & (times <= window[1])
    zoomed = (times >= zoom[0]) & (times <= zoom[1])
    if selected.sum() < 2 or zoomed.sum() < 2:
        raise ValueError("Display and inset intervals must lie inside the rollout and contain at least two points")
    for name in ["Ground Truth", "MSE", "DIM", "DIM+MSE", "PRIME"]:
        ax.plot(times[selected], trajectories[name][selected, channel],
                color=COLORS[name], ls="-" if name == "Ground Truth" else "--",
                lw=2 if name == "Ground Truth" else 1.2,
                alpha=.75 if name == "Ground Truth" else 1., label=name)
    ax.set(xlim=window, xlabel="Time", ylabel=f"Probe {channel} amplitude")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.18), ncol=5, frameon=False, fontsize=8)
    inner = inset_axes(ax, width="34%", height="48%", loc="upper left", borderpad=1.2)
    for name in ["Ground Truth", "MSE", "DIM", "DIM+MSE", "PRIME"]:
        inner.plot(times[zoomed], trajectories[name][zoomed, channel], color=COLORS[name],
                   ls="-" if name == "Ground Truth" else "--", lw=2 if name == "Ground Truth" else 1.1)
    values = np.concatenate([x[zoomed, channel] for x in trajectories.values()])
    low, high = values.min(), values.max()
    pad = .08 * (high - low + 1e-12)
    inner.set(xlim=zoom, ylim=(low - pad, high + pad), xticks=[], yticks=[])


def plot_flow(ax, raw, frame, fig):
    frames = raw[0]
    if not 0 <= frame < len(frames):
        raise ValueError(f"frame {frame} outside raw sequence of length {len(frames)}")
    speed = np.asarray(frames[frame], dtype=np.float64) * 100.
    if speed.shape != (150, 50):
        raise ValueError(f"Expected raw speed shape(150,50); got{speed.shape}")
    x, y = np.arange(150) * .1, np.arange(50) * .1
    im = ax.contourf(x, y, speed.T, cmap="RdBu_r", vmin=0., vmax=7., levels=100)
    cx, cy, radius = float(raw[1]), float(raw[2]), float(raw[3])
    ax.add_patch(Circle((cx * .1, cy * .1), radius * .1, facecolor="white"))
    for i, (sx, sy) in enumerate(SENSORS * .1):
        ax.add_patch(Circle((sx, sy), .15, color="grey", alpha=.75))
        ax.scatter(sx, sy, marker=".", s=3, c="black")
        ax.text(sx + .08, sy + .08, str(i), fontsize=7)
    ax.set(xlabel="$x$", ylabel="$y$", title="Flow past a cylinder", aspect="equal")
    fig.colorbar(im, ax=ax, label="Flow speed", shrink=.75, pad=.02)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/cylinder")
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/cylinder")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--horizon", type=int, default=1000, help="Common observation steps, dt=0.1.")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--plot-window", nargs=2, type=float, default=[80., 100.])
    p.add_argument("--zoom-window", nargs=2, type=float, default=[86., 90.])
    p.add_argument("--raw-file", type=Path, help="Optional trusted flow_full70.p for the left flow snapshot; not bundled.")
    p.add_argument("--raw-frame", type=int, default=4000, help="Raw index4000 reproduces V[1000] after the3000-frame transient.")
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs/figures/cylinder")
    args = p.parse_args()
    if not 0 <= args.channel < 9 or args.start_index < 0 or args.horizon < 1:
        p.error("channel must be0..8, start nonnegative, and horizon positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly choose --device cpu for a CPU plot")
    spec = importlib.util.spec_from_file_location("cylinder_evaluator", ROOT / "experiments/cylinder/evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    data_path = args.data_dir / "flow_sensors.p"
    data = ev.load_p(data_path)
    train, test = np.asarray(data[0], dtype=np.float64), np.asarray(data[1], dtype=np.float64)
    dt_raw, dt = float(data[4]), float(data[6])
    common = np.arange(0., (len(test) - 1) * dt_raw, dt)
    ref = ev.interp_traj_to_times(test, dt_raw, common)
    if args.start_index + args.horizon >= len(ref):
        p.error("Requested rollout exceeds the available test sequence")
    ref = ref[args.start_index:args.start_index + args.horizon + 1]
    times = np.arange(len(ref), dtype=np.float64) * dt
    trajectories = {"Ground Truth": ref}
    provenance = []
    for model, filename in FILES.items():
        path = args.checkpoint_dir / f"Seed{args.seed}" / filename
        net, scaler, transformer, dt_model = ev.load_net_and_scaler(path, train, dt, args.device)
        trajectories[model] = ev.perform_rollout(net, scaler, transformer, dt_model, ref[0], times, args.device)
        provenance.append(dict(model=model, file=f"Seed{args.seed}/{filename}", sha256=hashlib.sha256(path.read_bytes()).hexdigest(), dt_model=dt_model))
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 10})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9., 4.))
    plot_forecast(ax, times, trajectories, args.channel, args.plot_window, args.zoom_window)
    fig.subplots_adjust(top=.8, bottom=.16)
    for extension in ["pdf", "png"]:
        fig.savefig(args.output_dir / f"cylinder_forecast.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    if args.raw_file:
        raw = ev.load_p(args.raw_file)
        fig, axes = plt.subplots(1, 2, figsize=(13., 4.3), gridspec_kw={"width_ratios": [1.05, 1.]})
        plot_flow(axes[0], raw, args.raw_frame, fig)
        plot_forecast(axes[1], times, trajectories, args.channel, args.plot_window, args.zoom_window)
        fig.subplots_adjust(wspace=.36, top=.8, bottom=.16)
        for extension in ["pdf", "png"]:
            fig.savefig(args.output_dir / f"cylinder_flow_and_forecast.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    np.savez_compressed(args.output_dir / "cylinder_forecast_arrays.npz", time=times,
                        reference=ref, **{k.replace("+", "_"): v for k, v in trajectories.items() if k != "Ground Truth"})
    manifest = dict(seed=args.seed, channel=args.channel, start_index=args.start_index, horizon=args.horizon,
                    plot_window=args.plot_window, zoom_window=args.zoom_window, device=args.device, dt=dt,
                    torch=torch.__version__, data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    checkpoints=provenance, raw_file=str(args.raw_file) if args.raw_file else None,
                    raw_frame=args.raw_frame if args.raw_file else None,
                    source="Comparison of Cylinder CFD.ipynb cells17/18; Flow_data_preparison.py for snapshot/probes",
                    exact_published_figure_selection_verified=False)
    (args.output_dir / "cylinder_forecast_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved cylinder forecast panels, plotted arrays and manifest to {args.output_dir}")


if __name__ == "__main__":
    main()
