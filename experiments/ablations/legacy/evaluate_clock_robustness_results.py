#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Lorenz63 PRIME recurrence-clock robustness results.

This script is meant to be run AFTER recurrence_clock_ablation.py --all has finished.

It reads:
    lorenz_clock_robustness/clockrobust_summary_seed1000.csv

and evaluates the selected checkpoints for:
    1) NMSE over 30 rollout starts: mean ± std
    2) Attractor_W2 over the same 30 rollout starts: mean only
    3) PSD_logL1 over the same 30 rollout starts: mean only

This is aligned with your evaluate_implicit_ablation_results.py script.
"""

import io
import os
import csv
import json
import pickle
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.preprocessing import MaxAbsScaler


# ============================================================
# optional metric dependencies
# ============================================================
try:
    from geomloss import SamplesLoss
    HAS_GEOMLOSS = True
except Exception as e:
    HAS_GEOMLOSS = False
    GEOMLOSS_ERR = repr(e)

try:
    from scipy.signal import welch
    HAS_WELCH = True
except Exception as e:
    HAS_WELCH = False
    WELCH_ERR = repr(e)


# ============================================================
# args
# ============================================================
def parse_list_int(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--train_path", type=str, default="train_data.npy")
    p.add_argument("--test_path", type=str, default="test_data.npy")
    p.add_argument("--dt_obs", type=float, default=0.1)
    p.add_argument("--dt_base", type=float, default=0.1)

    p.add_argument("--run_root", type=str, default="./lorenz_clock_robustness")
    p.add_argument("--summary_csv", type=str, default=None)

    p.add_argument("--n_rollouts", type=int, default=30)
    p.add_argument("--windows", type=str, default="10,20,50,100,200,300,400,500")
    p.add_argument("--struct_len", type=int, default=500)
    p.add_argument("--start_mode", type=str, default="linspace", choices=["linspace", "random"])
    p.add_argument("--start_seed", type=int, default=42)

    p.add_argument("--psd_nfft", type=int, default=2048)
    p.add_argument("--psd_channels", type=str, default="all")

    p.add_argument("--w2_blur", type=float, default=0.05)
    p.add_argument("--w2_max_points", type=int, default=1500)
    p.add_argument("--w2_fallback_proj", type=int, default=256)

    p.add_argument("--output_dir", type=str, default=None)

    return p.parse_args()


# ============================================================
# IO helpers
# ============================================================
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_p_cpu(path: str):
    with open(path, "rb") as f:
        return CPU_Unpickler(f).load()


def smart_load(path: str):
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


def read_clock_summary(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))

    if len(rows) == 0:
        raise RuntimeError(f"Empty summary csv: {path}")

    return rows


def resolve_checkpoint_path(ckpt: str, run_root: str, summary_csv: str) -> str:
    if os.path.exists(ckpt):
        return ckpt

    p1 = os.path.join(run_root, ckpt)
    if os.path.exists(p1):
        return p1

    summary_dir = os.path.dirname(os.path.abspath(summary_csv))
    p2 = os.path.join(summary_dir, ckpt)
    if os.path.exists(p2):
        return p2

    # Common case: ckpt already starts with lorenz_clock_robustness/...
    # but script is executed from a different parent folder.
    base = os.path.basename(ckpt)
    matches = list(Path(run_root).glob(f"**/{base}"))
    if len(matches) > 0:
        return str(matches[0])

    raise FileNotFoundError(f"Cannot find checkpoint: {ckpt}")


def strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(sd, dict):
        return sd
    if len(sd) > 0 and all(str(k).startswith("module.") for k in sd.keys()):
        return {str(k).replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def infer_mlp_dims_from_state_dict(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    sd = strip_module_prefix(sd)
    wkeys = []
    for k in sd.keys():
        if str(k).endswith(".weight"):
            p = str(k).split(".")[0]
            if p.isdigit():
                wkeys.append((int(p), k))

    wkeys = sorted(wkeys, key=lambda x: x[0])
    if len(wkeys) == 0:
        raise ValueError("Could not infer MLP dims from state_dict.")

    W0 = sd[wkeys[0][1]]
    Wlast = sd[wkeys[-1][1]]
    width = int(W0.shape[0])
    d_in = int(W0.shape[1])
    d_out = int(Wlast.shape[0])
    return d_in, width, d_out


def build_net(d: int, width: int = 100) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, d),
    )


def load_model_ckpt(path: str, train_data: np.ndarray):
    obj = smart_load(path)

    sd = None
    scaler = None
    config = {}

    if isinstance(obj, dict):
        config = obj.get("config", {}) if isinstance(obj.get("config", {}), dict) else {}

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
        print(f"[warn] no scaler found in {path}; fitting MaxAbsScaler on train_data")
        scaler = MaxAbsScaler().fit(np.asarray(train_data, dtype=np.float64))

    return obj, sd, scaler, config


# ============================================================
# rollout
# ============================================================
@torch.no_grad()
def rollout_from_x0(
    net: nn.Module,
    scaler,
    x0_raw: np.ndarray,
    K_obs: int,
    dt_base: float,
    sub: int,
    device: str,
) -> np.ndarray:
    """
    Same rollout convention as your implicit ablation evaluation script:
        x <- x + dt_base * f_theta(x)
    repeated sub times per observed step.
    """
    x0_s = scaler.transform(np.asarray(x0_raw).reshape(1, -1)).astype(np.float32)
    x = torch.tensor(x0_s, dtype=torch.float32, device=device)

    pred_s = np.zeros((K_obs + 1, x.shape[1]), dtype=np.float32)
    for k in range(K_obs + 1):
        pred_s[k] = x.detach().cpu().numpy()[0]
        if k == K_obs:
            break
        for _ in range(int(sub)):
            x = x + float(dt_base) * net(x)

    return scaler.inverse_transform(pred_s).astype(np.float64)


# ============================================================
# metrics, aligned with your second script
# ============================================================
def nmse_value(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    L = min(len(gt), len(pred))
    gt = gt[:L]
    pred = pred[:L]
    mse = np.mean((gt - pred) ** 2)
    denom = np.mean(gt ** 2) + 1e-12
    return float(mse / denom)


def psd_logl1_distance(ref: np.ndarray, pred: np.ndarray, dt: float, nfft: int = 2048, channels=None) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    L = min(len(ref), len(pred))
    ref = ref[:L]
    pred = pred[:L]

    if channels is None:
        channels = list(range(ref.shape[1]))

    fs = 1.0 / float(dt)
    eps = 1e-12
    dists = []

    for ch in channels:
        x = ref[:, ch] - np.mean(ref[:, ch])
        y = pred[:, ch] - np.mean(pred[:, ch])

        n = min(int(nfft), L)
        x = x[:n]
        y = y[:n]

        if len(x) < 8 or len(y) < 8:
            dists.append(np.nan)
            continue

        if HAS_WELCH:
            _, Pxx = welch(x, fs=fs, nperseg=min(512, n), noverlap=min(256, n // 2))
            _, Pyy = welch(y, fs=fs, nperseg=min(512, n), noverlap=min(256, n // 2))
        else:
            X = np.fft.rfft(x)
            Y = np.fft.rfft(y)
            Pxx = (np.abs(X) ** 2) / max(1, len(x))
            Pyy = (np.abs(Y) ** 2) / max(1, len(y))

        mlen = min(len(Pxx), len(Pyy))
        if mlen == 0:
            dists.append(np.nan)
            continue

        Pxx = Pxx[:mlen] + eps
        Pyy = Pyy[:mlen] + eps
        dists.append(float(np.mean(np.abs(np.log10(Pxx) - np.log10(Pyy)))))

    dists = np.asarray(dists, dtype=np.float64)
    dists = dists[np.isfinite(dists)]
    return float(np.mean(dists)) if len(dists) else float("nan")


def sliced_w2_cloud(A: np.ndarray, B: np.ndarray, n_proj: int = 256, seed: int = 123) -> float:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if len(A) == 0 or len(B) == 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    d = A.shape[1]
    dirs = rng.normal(size=(int(n_proj), d))
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)

    n = min(len(A), len(B))
    vals = []
    for v in dirs:
        a = np.sort(A @ v)[:n]
        b = np.sort(B @ v)[:n]
        vals.append(np.mean((a - b) ** 2))

    return float(np.mean(vals))


def sinkhorn_w2_cloud(
    A: np.ndarray,
    B: np.ndarray,
    blur: float = 0.05,
    device: str = "cpu",
    max_points: int = 1500,
    fallback_proj: int = 256,
) -> float:
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if len(A) == 0 or len(B) == 0:
        return float("nan")

    if len(A) > int(max_points):
        idx = np.linspace(0, len(A) - 1, int(max_points)).astype(int)
        A = A[idx]
    if len(B) > int(max_points):
        idx = np.linspace(0, len(B) - 1, int(max_points)).astype(int)
        B = B[idx]

    if not HAS_GEOMLOSS:
        return sliced_w2_cloud(A, B, n_proj=fallback_proj, seed=123)

    XA = torch.tensor(A, dtype=torch.float32, device=device)
    XB = torch.tensor(B, dtype=torch.float32, device=device)
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=float(blur), backend="tensorized")
    return float(loss(XA, XB).detach().cpu())


# ============================================================
# evaluation
# ============================================================
def choose_starts(T: int, horizon: int, n_rollouts: int, mode: str, seed: int) -> np.ndarray:
    valid_count = T - int(horizon) - 1
    if valid_count <= 0:
        raise ValueError(f"test_data too short: T={T}, required horizon={horizon}")

    n = min(int(n_rollouts), valid_count)
    if mode == "linspace":
        starts = np.linspace(0, valid_count - 1, n).astype(int)
    else:
        rng = np.random.default_rng(seed)
        starts = np.sort(rng.choice(valid_count, size=n, replace=False))

    return np.unique(starts)


def mode_sort_key(mode: str) -> int:
    order = {
        "dim_only": 0,
        "exact_k": 1,
        "jitter_k_10": 2,
        "jitter_k_20": 3,
    }
    return order.get(mode, 999)


def fmt_float(x, nd=6) -> str:
    try:
        x = float(x)
    except Exception:
        return str(x)
    return "nan" if not np.isfinite(x) else f"{x:.{nd}g}"


def evaluate_one_checkpoint(
    summary_row: Dict[str, Any],
    ckpt_path: str,
    train_data: np.ndarray,
    test_data: np.ndarray,
    starts: np.ndarray,
    windows: List[int],
    struct_len: int,
    dt_obs: float,
    dt_base_default: float,
    sub_default: int,
    psd_nfft: int,
    psd_channels,
    w2_blur: float,
    w2_max_points: int,
    w2_fallback_proj: int,
    device: str,
) -> Dict[str, Any]:
    obj, sd, scaler, config = load_model_ckpt(ckpt_path, train_data=train_data)
    d_in, width, d_out = infer_mlp_dims_from_state_dict(sd)

    if d_in != d_out:
        raise ValueError(f"Model is not square: d_in={d_in}, d_out={d_out}, path={ckpt_path}")

    net = build_net(d_in, width=width).to(device)
    net.load_state_dict(sd)
    net.eval()

    dt_base = float(config.get("dt_base", dt_base_default))
    sub = int(config.get("sub", sub_default))
    if sub <= 0:
        sub = int(round(float(dt_obs) / float(dt_base)))

    row: Dict[str, Any] = {}
    row["mode"] = summary_row.get("mode", "")
    row["clock_jitter_frac"] = summary_row.get("clock_jitter_frac", "")
    row["mean_abs_relative_K_error"] = summary_row.get("mean_abs_relative_K_error", "")
    row["max_abs_relative_K_error"] = summary_row.get("max_abs_relative_K_error", "")
    row["final_cumulative_K_drift"] = summary_row.get("final_cumulative_K_drift", "")
    row["max_abs_cumulative_K_drift"] = summary_row.get("max_abs_cumulative_K_drift", "")
    row["N_hits"] = summary_row.get("N_hits", "")
    row["selected_source"] = summary_row.get("selected_source", "")
    row["checkpoint"] = ckpt_path
    row["width"] = width
    row["d"] = d_in
    row["n_rollouts"] = int(len(starts))
    row["struct_len"] = int(struct_len)

    nmse_by_window: Dict[int, List[float]] = {k: [] for k in windows}
    w2_vals: List[float] = []
    psd_vals: List[float] = []

    for s in starts:
        s = int(s)
        x0 = test_data[s]
        gt = test_data[s:s + struct_len + 1]

        pred = rollout_from_x0(
            net=net,
            scaler=scaler,
            x0_raw=x0,
            K_obs=int(struct_len),
            dt_base=dt_base,
            sub=sub,
            device=device,
        )

        for k in windows:
            gt_k = gt[:k + 1]
            pr_k = pred[:k + 1]
            nmse_by_window[k].append(nmse_value(gt_k, pr_k))

        w2_vals.append(
            sinkhorn_w2_cloud(
                gt,
                pred,
                blur=w2_blur,
                device=device,
                max_points=w2_max_points,
                fallback_proj=w2_fallback_proj,
            )
        )

        psd_vals.append(
            psd_logl1_distance(
                gt,
                pred,
                dt=float(dt_obs),
                nfft=psd_nfft,
                channels=psd_channels,
            )
        )

    for k in windows:
        vals = np.asarray(nmse_by_window[k], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            mu, sdv = np.nan, np.nan
        else:
            mu, sdv = float(np.mean(vals)), float(np.std(vals))
        row[f"NMSE_{k}_mean"] = mu
        row[f"NMSE_{k}_std"] = sdv
        row[f"NMSE_{k}"] = "nan" if not np.isfinite(mu) else f"{mu:.4g} ± {sdv:.4g}"

    w2_vals = np.asarray(w2_vals, dtype=np.float64)
    w2_vals = w2_vals[np.isfinite(w2_vals)]

    psd_vals = np.asarray(psd_vals, dtype=np.float64)
    psd_vals = psd_vals[np.isfinite(psd_vals)]

    row["Attractor_W2_mean"] = float(np.mean(w2_vals)) if len(w2_vals) else np.nan
    row["PSD_logL1_mean"] = float(np.mean(psd_vals)) if len(psd_vals) else np.nan

    return row


# ============================================================
# output
# ============================================================
def sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: mode_sort_key(str(r.get("mode", ""))))


def print_tables(rows: List[Dict[str, Any]], windows: List[int]) -> None:
    rows = sort_rows(rows)
    colw = 20

    print("\n==================== CLOCK ROBUSTNESS: NMSE over rollout starts: mean ± std ====================")
    nmse_cols = [
        "mode", "clock_jitter_frac", "mean_abs_relative_K_error", "N_hits"
    ] + [f"NMSE_{k}" for k in windows]

    header = " | ".join([f"{c:>{colw}s}" for c in nmse_cols])
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join([f"{str(r.get(c, '')):>{colw}s}" for c in nmse_cols]))

    print("\n==================== CLOCK ROBUSTNESS: STRUCTURE METRICS over rollout starts: mean only ====================")
    struct_cols = [
        "mode", "clock_jitter_frac", "mean_abs_relative_K_error", "N_hits",
        "Attractor_W2_mean", "PSD_logL1_mean",
    ]

    header = " | ".join([f"{c:>{colw}s}" for c in struct_cols])
    print(header)
    print("-" * len(header))
    for r in rows:
        vals = []
        for c in struct_cols:
            if c in ["clock_jitter_frac", "mean_abs_relative_K_error", "Attractor_W2_mean", "PSD_logL1_mean"]:
                vals.append(fmt_float(r.get(c, "")))
            else:
                vals.append(str(r.get(c, "")))
        print(" | ".join([f"{v:>{colw}s}" for v in vals]))


def write_csv(rows: List[Dict[str, Any]], windows: List[int], path: str) -> None:
    meta_cols = [
        "mode", "clock_jitter_frac", "mean_abs_relative_K_error", "max_abs_relative_K_error",
        "final_cumulative_K_drift", "max_abs_cumulative_K_drift", "N_hits",
        "selected_source", "checkpoint", "n_rollouts", "struct_len", "width", "d",
    ]

    metric_cols = []
    for k in windows:
        metric_cols += [f"NMSE_{k}_mean", f"NMSE_{k}_std"]
    metric_cols += ["Attractor_W2_mean", "PSD_logL1_mean"]

    cols = meta_cols + metric_cols

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in sort_rows(rows):
            writer.writerow({c: r.get(c, "") for c in cols})


def write_markdown(rows: List[Dict[str, Any]], windows: List[int], path: str) -> None:
    rows = sort_rows(rows)
    show_windows = [k for k in windows if k in (50, 100, 200, 400, 500)]
    if len(show_windows) == 0:
        show_windows = windows

    cols = [
        "mode", "clock_jitter_frac", "mean_abs_relative_K_error", "N_hits"
    ] + [f"NMSE_{k}" for k in show_windows] + ["Attractor_W2_mean", "PSD_logL1_mean"]

    with open(path, "w") as f:
        f.write("# Lorenz63 Recurrence-Clock Robustness Evaluation\n\n")
        f.write(
            "NMSE is reported as mean ± std over rollout starts. "
            "Attractor_W2 and PSD_logL1 are reported as mean only over the same starts.\n\n"
        )
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")

        for r in rows:
            vals = []
            for c in cols:
                if c in [
                    "clock_jitter_frac", "mean_abs_relative_K_error",
                    "Attractor_W2_mean", "PSD_logL1_mean"
                ]:
                    vals.append(fmt_float(r.get(c, "")))
                else:
                    vals.append(str(r.get(c, "")))
            f.write("| " + " | ".join(vals) + " |\n")


def plot_summary(rows: List[Dict[str, Any]], windows: List[int], out_dir: str) -> None:
    rows = sort_rows(rows)

    x = []
    labels = []
    w2 = []
    psd = []
    for r in rows:
        labels.append(str(r.get("mode", "")))
        try:
            x.append(float(r.get("mean_abs_relative_K_error", 0.0)))
        except Exception:
            x.append(np.nan)
        w2.append(float(r.get("Attractor_W2_mean", np.nan)))
        psd.append(float(r.get("PSD_logL1_mean", np.nan)))

    x = np.asarray(x, dtype=np.float64)

    def simple_plot(y, ylabel, fname):
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return

        plt.figure(figsize=(6.5, 4), dpi=170)
        plt.plot(x[mask], y[mask], marker="o")
        for xi, yi, lab in zip(x[mask], y[mask], np.asarray(labels)[mask]):
            plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)

        plt.xlabel("mean absolute relative K error")
        plt.ylabel(ylabel)
        plt.title(ylabel + " vs recurrence-clock perturbation")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()

    simple_plot(w2, "Attractor_W2_mean", "clock_error_vs_Attractor_W2_mean.png")
    simple_plot(psd, "PSD_logL1_mean", "clock_error_vs_PSD_logL1_mean.png")

    for k in windows:
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(f"NMSE_{k}_mean", np.nan)))
            except Exception:
                vals.append(np.nan)
        simple_plot(vals, f"NMSE_{k}_mean", f"clock_error_vs_NMSE_{k}_mean.png")


# ============================================================
# main
# ============================================================
def main() -> None:
    args = parse_args()

    train_data = np.load(args.train_path)
    test_data = np.load(args.test_path)

    train_data = np.asarray(train_data, dtype=np.float64)
    test_data = np.asarray(test_data, dtype=np.float64)

    if train_data.ndim != 2 or test_data.ndim != 2:
        raise ValueError(f"Expected 2D arrays. train={train_data.shape}, test={test_data.shape}")

    windows = parse_list_int(args.windows)
    max_nmse = max(windows)

    sub_default = int(round(float(args.dt_obs) / float(args.dt_base)))
    if sub_default <= 0:
        raise ValueError(f"Invalid sub_default={sub_default}")

    if args.psd_channels.strip().lower() == "all":
        psd_channels = None
    else:
        psd_channels = parse_list_int(args.psd_channels)

    run_root = os.path.abspath(args.run_root)
    summary_csv = args.summary_csv
    if summary_csv is None:
        summary_csv = os.path.join(run_root, "clockrobust_summary_seed1000.csv")
    summary_csv = os.path.abspath(summary_csv)

    clock_rows = read_clock_summary(summary_csv)

    struct_len = int(args.struct_len)
    if struct_len < max_nmse:
        struct_len = max_nmse

    if len(test_data) <= struct_len + 1:
        max_possible = len(test_data) - 2
        if max_possible < max_nmse:
            raise ValueError(
                f"test_data too short for max NMSE window {max_nmse}: T={len(test_data)}"
            )
        print(f"[warn] struct_len={struct_len} too long for test_data; shrinking to {max_possible}")
        struct_len = max_possible

    starts = choose_starts(
        T=len(test_data),
        horizon=struct_len,
        n_rollouts=args.n_rollouts,
        mode=args.start_mode,
        seed=args.start_seed,
    )

    if args.output_dir is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(run_root, f"eval_30start_metrics_{stamp}")
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("============================================================")
    print("Lorenz63 recurrence-clock robustness evaluation")
    print("============================================================")
    print("device:", device)
    print("run_root:", run_root)
    print("summary_csv:", summary_csv)
    print("output_dir:", out_dir)
    print("train:", train_data.shape, args.train_path)
    print("test :", test_data.shape, args.test_path)
    print("dt_obs:", args.dt_obs, "dt_base:", args.dt_base, "sub_default:", sub_default)
    print("windows:", windows)
    print("struct_len:", struct_len)
    print("starts:", starts)
    if not HAS_GEOMLOSS:
        print("[warn] geomloss unavailable; Attractor_W2 uses sliced-W2 fallback:", GEOMLOSS_ERR)
    if not HAS_WELCH:
        print("[warn] scipy.signal.welch unavailable; PSD uses FFT fallback:", WELCH_ERR)
    print("============================================================")

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    clock_rows = sorted(clock_rows, key=lambda r: mode_sort_key(str(r.get("mode", ""))))

    for i, sr in enumerate(clock_rows, 1):
        mode = sr.get("mode", "")
        ckpt_raw = sr.get("selected_checkpoint", "")

        if ckpt_raw == "":
            errors.append({"mode": mode, "checkpoint": "", "error": "empty selected_checkpoint"})
            continue

        try:
            ckpt_path = resolve_checkpoint_path(ckpt_raw, run_root=run_root, summary_csv=summary_csv)
            print(f"\n[{i}/{len(clock_rows)}] Evaluating mode={mode}")
            print("checkpoint:", ckpt_path)

            row = evaluate_one_checkpoint(
                summary_row=sr,
                ckpt_path=ckpt_path,
                train_data=train_data,
                test_data=test_data,
                starts=starts,
                windows=windows,
                struct_len=struct_len,
                dt_obs=float(args.dt_obs),
                dt_base_default=float(args.dt_base),
                sub_default=sub_default,
                psd_nfft=int(args.psd_nfft),
                psd_channels=psd_channels,
                w2_blur=float(args.w2_blur),
                w2_max_points=int(args.w2_max_points),
                w2_fallback_proj=int(args.w2_fallback_proj),
                device=device,
            )

            rows.append(row)

            print(
                f"  ok | mode={row.get('mode')} | "
                f"Kerr={fmt_float(row.get('mean_abs_relative_K_error'))} | "
                f"N_hits={row.get('N_hits')} | "
                f"W2={fmt_float(row.get('Attractor_W2_mean'))} | "
                f"PSD={fmt_float(row.get('PSD_logL1_mean'))}"
            )

        except Exception as e:
            msg = repr(e)
            errors.append({"mode": mode, "checkpoint": ckpt_raw, "error": msg})
            print("  [ERROR]", msg)

    if len(rows) == 0:
        raise RuntimeError("No checkpoint was successfully evaluated. See errors above.")

    print_tables(rows, windows)

    csv_path = os.path.join(out_dir, "clock_robustness_eval_30starts.csv")
    md_path = os.path.join(out_dir, "clock_robustness_eval_30starts.md")
    json_path = os.path.join(out_dir, "clock_robustness_eval_30starts.json")
    err_path = os.path.join(out_dir, "clock_robustness_eval_errors.json")

    write_csv(rows, windows, csv_path)
    write_markdown(rows, windows, md_path)
    with open(json_path, "w") as f:
        json.dump(sort_rows(rows), f, indent=2)
    with open(err_path, "w") as f:
        json.dump(errors, f, indent=2)

    plot_summary(rows, windows, out_dir)

    print("\nSaved:")
    print("  CSV :", csv_path)
    print("  MD  :", md_path)
    print("  JSON:", json_path)
    print("  ERR :", err_path)
    print("Done.")


if __name__ == "__main__":
    main()