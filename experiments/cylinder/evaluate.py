#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CFD 5-seed evaluation script.

Protocol:

A) Prediction / structure / temporal metrics
   - 5 seed folders × 32 rollout starts = 160 rollouts per model
   - Metrics:
       NMSE_20
       NMSE_100
       NMSE_400
       Attractor_W2
       PSD_logL1
       TCM

B) Poincare / DRTD metrics
   - For each seed model, rollout once from test[0]
   - Default horizon = 1000 common-time steps
   - Sections:
       Sigma1 = {u6 = 4}, Sigma2 = {u4 = 4}, Sigma3 = {u5 = 4}
       (zero-based channels; both crossing directions)
   - DRTD = tau_dist_W1: legacy sorted/truncated return-time comparison
"""

from pathlib import Path
import sys
import hashlib
import time
import argparse
import csv
import glob
import io
import json
import os
import pickle
import warnings

import numpy as np
# Read NumPy 2 pickles in the validated NumPy 1.26 runtime.
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

try:
    from scipy.signal import welch
    HAS_WELCH = True
except Exception as e:
    HAS_WELCH = False
    print("[warn] scipy.signal.welch import failed; paper evaluation will refuse to run.")
    print("[warn]", repr(e))

try:
    from geomloss import SamplesLoss
    HAS_GEOMLOSS = True
except Exception as e:
    HAS_GEOMLOSS = False
    print("[warn] geomloss import failed; paper evaluation will refuse to run.")
    print("[warn]", repr(e))


# ============================================================
# Defaults
# ============================================================
DEFAULT_SEED_FOLDERS = [
    "Seed100",
    "Seed200",
    "Seed13532",
    "Seed123456",
    "Seed123123123",
]

MODEL_ORDER = ["PRIME", "MSE", "DIM", "DIM+MSE"]

MODEL_FILE_CANDIDATES = {
    "PRIME": [
        "PRIME.p", "PRIME", "prime.p",
    ],
    "MSE": [
        "MSE.p", "MSE", "MESE.p", "MESE", "pointwise.p", "POINTWISE.p",
    ],
    "DIM": [
        "DIM.p", "DIM",
    ],
    "DIM+MSE": [
        "DIM_MSE.p", "DIM_MSE", "DIM+MSE.p", "mixed.p", "MIXED.p",
    ],
}

NMSE_WINDOWS_DEFAULT = [20, 100, 400]


# ============================================================
# Formatting / aggregation
# ============================================================
def as_float_or_nan(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def finite_array(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return vals[np.isfinite(vals)]


def finite_mean_std(vals):
    vals = finite_array(vals)
    if len(vals) == 0:
        return np.nan, np.nan, 0
    return float(np.mean(vals)), float(np.std(vals, ddof=0)), int(len(vals))


def fmt_num(x, nd=2):
    if not np.isfinite(x):
        return "nan"
    ax = abs(float(x))
    if ax != 0 and (ax < 1e-2 or ax >= 1e4):
        return f"{x:.{nd}e}"
    return f"{x:.{nd}f}"


def fmt_mean_std(mu, sd, nd=2):
    if not np.isfinite(mu):
        return "nan"
    if not np.isfinite(sd):
        return f"{fmt_num(mu, nd)} ± nan"
    return f"{fmt_num(mu, nd)} ± {fmt_num(sd, nd)}"


def print_table(cols, rows):
    widths = [len(str(c)) for c in cols]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    print("  ".join(str(c).rjust(widths[i]) for i, c in enumerate(cols)))
    for row in rows:
        print("  ".join(str(cell).rjust(widths[i]) for i, cell in enumerate(row)))


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# IO / interpolation
# ============================================================
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_p(path):
    with open(path, "rb") as f:
        return CPU_Unpickler(f).load()


def smart_load(path):
    last_err = None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        last_err = e

    try:
        return load_p(path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint: {path}\n"
            f"torch.load error: {last_err}\n"
            f"pickle error: {e}"
        )


def interp_traj_to_times(traj_data, dt_data, times_query):
    traj_data = np.asarray(traj_data, dtype=np.float64)
    times_query = np.asarray(times_query, dtype=np.float64)

    t_data = np.arange(len(traj_data), dtype=np.float64) * float(dt_data)
    tq = np.clip(times_query, t_data[0], t_data[-1])

    out = np.zeros((len(tq), traj_data.shape[1]), dtype=np.float64)
    for j in range(traj_data.shape[1]):
        out[:, j] = np.interp(tq, t_data, traj_data[:, j])
    return out


# ============================================================
# Model loading
# ============================================================
def strip_module_prefix(sd):
    if not isinstance(sd, dict):
        return sd
    if len(sd) > 0 and all(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def build_mlp_from_sd(sd_local):
    sd_local = strip_module_prefix(sd_local)

    weight_keys = [k for k in sd_local.keys() if k.endswith("weight")]
    if not weight_keys:
        raise RuntimeError("No weight keys found in state_dict.")

    if any(k.startswith("L1.") for k in weight_keys):
        ordered = sorted(weight_keys, key=lambda k: int(k.split(".")[0][1:]))
    elif all(k.split(".")[0].isdigit() for k in weight_keys):
        ordered = sorted(weight_keys, key=lambda k: int(k.split(".")[0]))
    else:
        # fallback: preserve stable lexical order
        ordered = sorted(weight_keys)

    seq = []
    for i, wk in enumerate(ordered):
        bk = wk.replace("weight", "bias")
        if bk not in sd_local:
            raise RuntimeError(f"Missing bias key for {wk}: expected {bk}")

        W = sd_local[wk]
        b = sd_local[bk]

        lin = nn.Linear(W.shape[1], W.shape[0])
        lin.weight.data.copy_(W)
        lin.bias.data.copy_(b)
        seq.append(lin)

        if i != len(ordered) - 1:
            seq.append(nn.Tanh())

    return nn.Sequential(*seq)


def extract_state_dict_and_scalers(obj):
    """
    Supports dict checkpoints used in your CFD scripts:
      obj["model"]["state_dict"]
      obj["state_dict"]
      obj["model_state_dict"]
      obj itself as state_dict
    Also supports scaler / transformer stored either top-level or inside model block.
    """
    if isinstance(obj, (list, tuple)):
        raise TypeError(
            "Checkpoint appears to contain only prediction trajectories, not model weights."
        )

    if not isinstance(obj, dict):
        return obj, None, None, {}

    cfg = obj.get("config", {})

    mdl_blk = obj.get("model", obj)

    sd = None
    if isinstance(mdl_blk, dict):
        sd = mdl_blk.get("state_dict", None)
    if sd is None:
        sd = obj.get("state_dict", None)
    if sd is None:
        sd = obj.get("model_state_dict", None)
    if sd is None:
        sd = obj.get("net_state_dict", None)
    if sd is None:
        sd = obj

    scaler = None
    transformer = None

    if isinstance(mdl_blk, dict):
        scaler = mdl_blk.get("scaler", None)
        transformer = mdl_blk.get("transformer", None)

    if scaler is None:
        scaler = obj.get("scaler", None)
    if transformer is None:
        transformer = obj.get("transformer", None)

    return sd, scaler, transformer, cfg


def load_net_and_scaler(path, training_data, fallback_dt, device):
    obj = smart_load(path)
    sd, scaler, transformer, cfg = extract_state_dict_and_scalers(obj)

    if not isinstance(sd, dict):
        raise RuntimeError(f"Could not find usable state_dict in {path}")

    # ------------------------------------------------------------
    # Robust dt extraction.
    # Some checkpoints store dt inside obj["config"];
    # MSE.p stores dt directly as obj["dt"].
    # ------------------------------------------------------------
    dt_model = None

    if isinstance(cfg, dict):
        for key in ["dt_fine", "dt_model", "dt", "dt_data_sub"]:
            if key in cfg and cfg[key] is not None:
                dt_model = float(cfg[key])
                break

    if dt_model is None and isinstance(obj, dict):
        for key in ["dt_fine", "dt_model", "dt", "dt_data_sub"]:
            if key in obj and obj[key] is not None:
                dt_model = float(obj[key])
                break

    if dt_model is None and isinstance(obj, dict) and isinstance(obj.get("model", None), dict):
        mdl_blk = obj["model"]
        for key in ["dt_fine", "dt_model", "dt", "dt_data_sub"]:
            if key in mdl_blk and mdl_blk[key] is not None:
                dt_model = float(mdl_blk[key])
                break

    if dt_model is None:
        dt_model = float(fallback_dt)
        print(f"[warn] {path} has no dt field; fallback dt_model={dt_model}")

    if scaler is None and transformer is None:
        print(f"[warn] {path} has no scaler/transformer; fitting StandardScaler on training data.")
        scaler = StandardScaler().fit(training_data)

    net = build_mlp_from_sd(sd).to(device)
    net.eval()

    return net, scaler, transformer, float(dt_model)


def scale_X(X, scaler, transformer):
    if transformer is not None:
        return transformer.transform(X)
    return scaler.transform(X)


def inv_scale_X(Xs, scaler, transformer):
    if transformer is not None:
        return transformer.inverse_transform(Xs)
    return scaler.inverse_transform(Xs)


@torch.no_grad()
def perform_rollout(net, scaler, transformer, dt_model, x0_phys, t_common_local, device):
    """
    Roll out model in its dt_model time step, then interpolate prediction
    back onto the common observation grid t_common_local.
    """
    t_end = float(t_common_local[-1])
    n_steps = int(np.ceil(t_end / float(dt_model))) + 1

    x = torch.tensor(
        scale_X(np.asarray(x0_phys)[None, :], scaler, transformer),
        dtype=torch.float32,
        device=device,
    )

    xs = [x.detach().cpu().numpy()[0]]

    with torch.no_grad():
        for _ in range(n_steps):
            x = x + float(dt_model) * net(x)
            xs.append(x.detach().cpu().numpy()[0])

    pred_phys = inv_scale_X(np.asarray(xs, dtype=np.float64), scaler, transformer)
    pred_on_common = interp_traj_to_times(pred_phys, dt_model, t_common_local)
    return pred_on_common


def resolve_model_path(seed_folder, model_name):
    candidates = MODEL_FILE_CANDIDATES[model_name]
    for cand in candidates:
        p = os.path.join(seed_folder, cand)
        if os.path.exists(p):
            return p
    return None


def load_all_models(seed_folders, training, dim, device, fallback_dt, models=None):
    loaded = []
    errors = []

    for sf in seed_folders:
        for model_name in (models or MODEL_ORDER):
            ckpt_path = resolve_model_path(sf, model_name)

            if ckpt_path is None:
                errors.append({
                    "stage": "resolve_checkpoint",
                    "seed_folder": sf,
                    "model": model_name,
                    "error": "checkpoint not found",
                })
                print(f"[warn] missing {model_name} checkpoint in {sf}")
                continue

            try:
                net, scaler, transformer, dt_model = load_net_and_scaler(
                    ckpt_path,
                    training_data=training,
                    fallback_dt=fallback_dt,
                    device=device,
                )

                loaded.append({
                    "seed_folder": sf,
                    "model": model_name,
                    "checkpoint": ckpt_path,
                    "net": net,
                    "scaler": scaler,
                    "transformer": transformer,
                    "dt_model": dt_model,
                })

                print(f"[OK] loaded {model_name:7s} | {sf} | {ckpt_path} | dt_model={dt_model}")

            except Exception as e:
                errors.append({
                    "stage": "load_model",
                    "seed_folder": sf,
                    "model": model_name,
                    "checkpoint": ckpt_path,
                    "error": repr(e),
                })
                print(f"[warn] failed to load {model_name} from {ckpt_path}: {e}")

    return loaded, errors


# ============================================================
# Metrics: NMSE / W2 / PSD / TCM
# ============================================================
def nmse_window_steps(gt, pred, k, eps=1e-12):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    L = min(len(gt), len(pred))
    if L <= 0:
        return np.nan

    k = int(max(0, min(k, L - 1)))
    err = gt[:k + 1] - pred[:k + 1]

    num = np.mean(err ** 2)
    den = np.mean(gt[:k + 1] ** 2) + eps
    return float(num / den)


def sliced_w2_cloud(A, B, n_proj=256, seed=123):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    A = A[np.isfinite(A).all(axis=1)]
    B = B[np.isfinite(B).all(axis=1)]

    if len(A) == 0 or len(B) == 0:
        return np.nan

    n = min(len(A), len(B))

    rng = np.random.default_rng(seed)
    d = A.shape[1]
    dirs = rng.normal(size=(n_proj, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

    vals = []
    for v in dirs:
        a = np.sort(A @ v)[:n]
        b = np.sort(B @ v)[:n]
        vals.append(np.mean((a - b) ** 2))

    return float(np.mean(vals))


def sinkhorn_w2_cloud(A, B, blur=0.05, device="cpu", max_points=1500):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)

    A = A[np.isfinite(A).all(axis=1)]
    B = B[np.isfinite(B).all(axis=1)]

    if len(A) == 0 or len(B) == 0:
        return np.nan

    if np.nanmax(np.abs(A)) > 1e6 or np.nanmax(np.abs(B)) > 1e6:
        return np.nan

    if len(A) > max_points:
        idx = np.linspace(0, len(A) - 1, max_points).astype(int)
        A = A[idx]

    if len(B) > max_points:
        idx = np.linspace(0, len(B) - 1, max_points).astype(int)
        B = B[idx]

    if not HAS_GEOMLOSS:
        raise RuntimeError("GeomLoss is required; a sliced metric cannot replace the paper SD.")
    XA = torch.tensor(A, dtype=torch.float32, device=device)
    XB = torch.tensor(B, dtype=torch.float32, device=device)
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=float(blur), backend="tensorized")
    return float(loss(XA, XB).detach().cpu())


def psd_logl1_distance(ref, pred, dt, nfft=2048, channels=None, eps=1e-12):
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    L = min(len(ref), len(pred))
    ref = ref[:L]
    pred = pred[:L]

    if channels is None:
        channels = list(range(ref.shape[1]))

    fs = 1.0 / float(dt)
    dists = []

    for ch in channels:
        x = ref[:, ch] - np.mean(ref[:, ch])
        y = pred[:, ch] - np.mean(pred[:, ch])

        n = min(int(nfft), L)
        x = x[:n]
        y = y[:n]

        if len(x) < 8 or len(y) < 8:
            continue

        if HAS_WELCH:
            _, Pxx = welch(x, fs=fs, nperseg=min(512, n), noverlap=min(256, n // 2))
            _, Pyy = welch(y, fs=fs, nperseg=min(512, n), noverlap=min(256, n // 2))
        else:
            Pxx = (np.abs(np.fft.rfft(x)) ** 2) / max(1, len(x))
            Pyy = (np.abs(np.fft.rfft(y)) ** 2) / max(1, len(y))

        m = min(len(Pxx), len(Pyy))
        if m > 0:
            dists.append(float(np.mean(np.abs(np.log10(Pxx[:m] + eps) - np.log10(Pyy[:m] + eps)))))

    dists = finite_array(dists)
    return float(np.mean(dists)) if len(dists) else np.nan


def autocorr_1d(x, max_lag, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) < 8:
        return np.zeros((0,), dtype=np.float64)

    x = x - np.mean(x)
    n = len(x)
    c0 = float(np.dot(x, x) / n)

    if c0 < eps:
        return np.zeros((int(min(max_lag, n - 2)) + 1,), dtype=np.float64)

    max_lag = int(min(max_lag, n - 2))
    rho = np.empty(max_lag + 1, dtype=np.float64)
    rho[0] = 1.0

    for lag in range(1, max_lag + 1):
        rho[lag] = float(np.dot(x[lag:], x[:-lag]) / (n - lag) / c0)

    return rho


def integrated_autocorr_time_1d(x, dt, max_lag=200, stop_first_nonpositive=False):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) < 8:
        return np.nan

    max_lag = min(int(max_lag), len(x) // 2)
    rho = autocorr_1d(x, max_lag=max_lag)

    if len(rho) == 0:
        return np.nan
    if np.allclose(rho, 0.0):
        return 0.0

    tail = rho[1:]

    if stop_first_nonpositive:
        bad = np.where(tail <= 0.0)[0]
        if len(bad):
            tail = tail[:bad[0]]

    tau = float(dt) * (1.0 + 2.0 * float(np.sum(tail)))
    return float(max(tau, 0.0)) if np.isfinite(tau) else np.nan


def tcm_distance(ref, pred, dt, max_lag=200, channels=None, stop_first_nonpositive=False):
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    L = min(len(ref), len(pred))
    ref = ref[:L]
    pred = pred[:L]

    if ref.ndim == 1:
        ref = ref[:, None]
    if pred.ndim == 1:
        pred = pred[:, None]

    if channels is None:
        channels = list(range(ref.shape[1]))

    tr_list, tp_list = [], []

    for ch in channels:
        tr = integrated_autocorr_time_1d(
            ref[:, ch],
            dt=dt,
            max_lag=max_lag,
            stop_first_nonpositive=stop_first_nonpositive,
        )

        tp = integrated_autocorr_time_1d(
            pred[:, ch],
            dt=dt,
            max_lag=max_lag,
            stop_first_nonpositive=stop_first_nonpositive,
        )

        if np.isfinite(tr) and np.isfinite(tp):
            tr_list.append(tr)
            tp_list.append(tp)

    if len(tr_list) == 0:
        return np.nan

    return float(abs(np.mean(tr_list) - np.mean(tp_list)))


# ============================================================
# Poincare / DRTD helpers
# ============================================================
def crossing_happened(a, b, direction):
    if direction > 0:
        return (a < 0.0) and (b >= 0.0)
    if direction < 0:
        return (a > 0.0) and (b <= 0.0)
    return (a == 0.0) or (b == 0.0) or (a * b < 0.0)


def extract_poincare_hits(traj, dt, section_dim=0, section_value=0.0, direction=0):
    X = np.asarray(traj, dtype=np.float64)
    X = X[np.isfinite(X).all(axis=1)]

    if len(X) < 2:
        return np.array([], dtype=np.float64), np.zeros((0, X.shape[1] if X.ndim == 2 else 0), dtype=np.float64), np.array([], dtype=np.float64)

    T, d = X.shape
    hit_times = []
    hit_points = []

    for k in range(T - 1):
        a = X[k, section_dim] - section_value
        b = X[k + 1, section_dim] - section_value

        if not crossing_happened(a, b, direction):
            continue

        denom = b - a
        alpha = 0.0 if abs(denom) < 1e-14 else -a / denom

        if alpha < 0.0 or alpha > 1.0:
            continue

        t_hit = (k + alpha) * float(dt)
        x_hit = X[k] + alpha * (X[k + 1] - X[k])

        hit_times.append(t_hit)
        hit_points.append(x_hit)

    hit_times = np.asarray(hit_times, dtype=np.float64)
    hit_points = np.asarray(hit_points, dtype=np.float64) if hit_points else np.zeros((0, d), dtype=np.float64)
    tau = np.diff(hit_times) if len(hit_times) >= 2 else np.array([], dtype=np.float64)

    return hit_times, hit_points, tau


def wasserstein_1d_empirical(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    n = min(len(x), len(y))
    xs = np.sort(x)[:n]
    ys = np.sort(y)[:n]

    return float(np.mean(np.abs(xs - ys)))


def build_fixed_sections(ref_common, direction=1):
    """
    Fixed CFD Poincare sections:

        Sigma1 = {u6 = 4}
        Sigma2 = {u4 = 4}
        Sigma3 = {u5 = 4}

    direction:
        0  = both directions
        +1 = upward crossing
        -1 = downward crossing
    """
    sections = [
        {
            "section_name": "Sigma1_u0_4",
            "section_label": "Sigma1: u6=4",
            "section_dim": 6,
            "section_value": 4.0,
            "section_direction": int(direction),
        },
        {
            "section_name": "Sigma2_u3_4",
            "section_label": "Sigma2: u4=4",
            "section_dim": 4,
            "section_value": 4.0,
            "section_direction": int(direction),
        },
        {
            "section_name": "Sigma3_u2_4",
            "section_label": "Sigma3: u5=4",
            "section_dim": 5,
            "section_value": 4.0,
            "section_direction": int(direction),
        },
    ]

    d = ref_common.shape[1]
    for sec in sections:
        if sec["section_dim"] < 0 or sec["section_dim"] >= d:
            raise ValueError(
                f"section dim {sec['section_dim']} out of range for data dim={d}"
            )

    return sections


def tau_dist_w1_on_section(ref_traj, pred_traj, dt, section):
    hit_ref, _, tau_ref = extract_poincare_hits(
        ref_traj,
        dt=dt,
        section_dim=section["section_dim"],
        section_value=section["section_value"],
        direction=section["section_direction"],
    )

    hit_pred, _, tau_pred = extract_poincare_hits(
        pred_traj,
        dt=dt,
        section_dim=section["section_dim"],
        section_value=section["section_value"],
        direction=section["section_direction"],
    )

    return {
        "gt_hits": int(len(hit_ref)),
        "pred_hits": int(len(hit_pred)),
        "tau_dist_W1": wasserstein_1d_empirical(tau_ref, tau_pred),
        "valid": int(np.isfinite(wasserstein_1d_empirical(tau_ref, tau_pred))),
        "gt_mean_tau": float(np.mean(tau_ref)) if len(tau_ref) else np.nan,
        "pred_mean_tau": float(np.mean(tau_pred)) if len(tau_pred) else np.nan,
    }


# ============================================================
# Evaluation routines
# ============================================================
def compute_start_indices(T_eval, horizon, n_starts):
    valid_max_idx = T_eval - horizon - 1

    if valid_max_idx <= 0:
        raise ValueError(f"trajectory too short: T_eval={T_eval}, horizon={horizon}")

    n = min(int(n_starts), valid_max_idx + 1)
    starts = np.linspace(0, valid_max_idx, n, dtype=int)
    return np.unique(starts)


def evaluate_prediction_metrics(loaded_models, ref_common, dt_common, args, device):
    starts = compute_start_indices(len(ref_common), args.pred_horizon, args.n_starts)

    print("\n============================================================")
    print("Prediction metrics: NMSE / W2 / PSD / TCM")
    print("============================================================")
    print("starts:", starts)
    print("pred_horizon:", args.pred_horizon)
    print("nmse_windows:", args.nmse_windows)

    rows, errors = [], []

    for item in loaded_models:
        model = item["model"]
        sf = item["seed_folder"]
        net = item["net"]
        scaler = item["scaler"]
        transformer = item["transformer"]
        dt_model = item["dt_model"]
        ckpt = item["checkpoint"]

        print(f"\nEvaluating prediction metrics: {model}, {sf}")

        for start_idx in starts:
            try:
                start_idx = int(start_idx)

                ref_traj = ref_common[start_idx: start_idx + args.pred_horizon + 1]
                t_local = np.arange(len(ref_traj), dtype=np.float64) * float(dt_common)
                x0_phys = ref_common[start_idx]

                pred_traj = perform_rollout(
                    net=net,
                    scaler=scaler,
                    transformer=transformer,
                    dt_model=dt_model,
                    x0_phys=x0_phys,
                    t_common_local=t_local,
                    device=device,
                )

                row = {
                    "model": model,
                    "seed_folder": sf,
                    "checkpoint": ckpt,
                    "start_idx": int(start_idx),
                    "pred_horizon": int(args.pred_horizon),
                }

                for k in args.nmse_windows:
                    row[f"NMSE_{k}"] = nmse_window_steps(ref_traj, pred_traj, k)

                row["Attractor_W2"] = sinkhorn_w2_cloud(
                    ref_traj,
                    pred_traj,
                    blur=args.w2_blur,
                    device=device,
                    max_points=args.w2_max_points,
                )

                row["PSD_logL1"] = psd_logl1_distance(
                    ref_traj,
                    pred_traj,
                    dt=dt_common,
                    nfft=args.psd_nfft,
                    channels=None,
                )

                row["TCM"] = tcm_distance(
                    ref_traj,
                    pred_traj,
                    dt=dt_common,
                    max_lag=args.tcm_max_lag,
                    channels=None,
                    stop_first_nonpositive=args.tcm_truncate_first_nonpositive,
                )

                rows.append(row)

            except Exception as e:
                errors.append({
                    "stage": "prediction_metrics",
                    "model": model,
                    "seed_folder": sf,
                    "checkpoint": ckpt,
                    "start_idx": int(start_idx),
                    "error": repr(e),
                })
                print(f"[warn] prediction eval failed: {model}, {sf}, start={start_idx}: {e}")

    return rows, errors


def evaluate_full_horizon_poincare(loaded_models, ref_common, dt_common, args, device):
    K = min(int(args.poincare_horizon), len(ref_common) - 1)
    ref_long = ref_common[:K + 1]
    t_local = np.arange(len(ref_long), dtype=np.float64) * float(dt_common)
    x0_phys = ref_common[0]

    sections = build_fixed_sections(
        ref_common=ref_common,
        direction=args.section_direction,
    )

    print("\n============================================================")
    print("Full-horizon Poincare / DRTD from test[0]")
    print("============================================================")
    print("poincare_horizon:", K)
    print("section_direction:", args.section_direction)
    for sec in sections:
        print(sec["section_label"], f"direction={sec['section_direction']:+d}")

    rows, errors = [], []

    for item in loaded_models:
        model = item["model"]
        sf = item["seed_folder"]
        net = item["net"]
        scaler = item["scaler"]
        transformer = item["transformer"]
        dt_model = item["dt_model"]
        ckpt = item["checkpoint"]

        print(f"\nEvaluating DRTD: {model}, {sf}")

        try:
            pred_long = perform_rollout(
                net=net,
                scaler=scaler,
                transformer=transformer,
                dt_model=dt_model,
                x0_phys=x0_phys,
                t_common_local=t_local,
                device=device,
            )

            for sec in sections:
                out = tau_dist_w1_on_section(
                    ref_traj=ref_long,
                    pred_traj=pred_long,
                    dt=dt_common,
                    section=sec,
                )

                row = {
                    "model": model,
                    "seed_folder": sf,
                    "checkpoint": ckpt,
                    "start_idx": 0,
                    "poincare_horizon": int(K),
                    **sec,
                    **out,
                }

                rows.append(row)

        except Exception as e:
            errors.append({
                "stage": "poincare_metrics",
                "model": model,
                "seed_folder": sf,
                "checkpoint": ckpt,
                "error": repr(e),
            })
            print(f"[warn] DRTD eval failed: {model}, {sf}: {e}")

    return rows, errors


# ============================================================
# Summaries
# ============================================================
def summarize_prediction(rows, nmse_windows):
    out = []

    for model in MODEL_ORDER:
        group = [r for r in rows if r["model"] == model]
        if not group:
            continue

        row = {
            "model": model,
            "n_prediction_rollouts": int(len(group)),
            "n_seed_folders_prediction": int(len(set(r["seed_folder"] for r in group))),
        }

        for metric in [f"NMSE_{k}" for k in nmse_windows] + ["Attractor_W2", "PSD_logL1", "TCM"]:
            vals = [as_float_or_nan(g.get(metric, np.nan)) for g in group]
            mu, sd, n = finite_mean_std(vals)
            row[f"{metric}_mean"] = mu
            row[f"{metric}_std"] = sd
            row[f"{metric}_n_valid"] = n
            row[metric] = fmt_mean_std(mu, sd)

        out.append(row)

    return out


def summarize_poincare(rows):
    out = []

    section_order = []
    for r in rows:
        name = r["section_name"]
        if name not in section_order:
            section_order.append(name)

    for sec_name in section_order:
        sec_rows = [r for r in rows if r["section_name"] == sec_name]
        sec_label = sec_rows[0]["section_label"]

        for model in MODEL_ORDER:
            group = [r for r in sec_rows if r["model"] == model]
            if not group:
                continue

            row = {
                "section_name": sec_name,
                "section_label": sec_label,
                "model": model,
                "records": int(len(group)),
                "n_seed_folders_poincare": int(len(set(r["seed_folder"] for r in group))),
            }

            for metric in ["gt_hits", "pred_hits", "gt_mean_tau", "pred_mean_tau", "tau_dist_W1"]:
                vals = [as_float_or_nan(g.get(metric, np.nan)) for g in group]
                mu, sd, n = finite_mean_std(vals)
                row[f"{metric}_mean"] = mu
                row[f"{metric}_std"] = sd
                row[f"{metric}_n_valid"] = n
                row[metric] = fmt_mean_std(mu, sd)

            row["valid"] = row["tau_dist_W1_n_valid"]

            out.append(row)

    return out


def write_markdown_summary(path, pred_summary, poi_summary, nmse_windows):
    with open(path, "w") as f:
        f.write("# CFD evaluation summary\n\n")
        f.write("Prediction metrics use 5 seed folders × 32 rollout starts by default. ")
        f.write("Poincare DRTD metrics use one rollout from test[0] per seed model. ")
        f.write("All values are mean ± std. Lower is better.\n\n")

        f.write("## Prediction / structure / temporal metrics\n\n")
        f.write("| Model | n | " + " | ".join([f"NMSE_{k}" for k in nmse_windows]) + " | Attractor W2 | PSD logL1 | TCM |\n")
        f.write("|---|---:|" + "---:|" * (len(nmse_windows) + 3) + "\n")

        for r in pred_summary:
            f.write(
                f"| {r['model']} | {r['n_prediction_rollouts']} | "
                + " | ".join([r.get(f"NMSE_{k}", "nan") for k in nmse_windows])
                + f" | {r.get('Attractor_W2', 'nan')} | {r.get('PSD_logL1', 'nan')} | {r.get('TCM', 'nan')} |\n"
            )

        f.write("\n## Poincare DRTD from test[0]\n\n")

        sections = []
        for r in poi_summary:
            if r["section_name"] not in sections:
                sections.append(r["section_name"])

        for sec_name in sections:
            label = next(r["section_label"] for r in poi_summary if r["section_name"] == sec_name)
            f.write(f"### {label}\n\n")
            f.write("| Model | records | GT hits | Pred hits | DRTD / tau_dist_W1 | valid |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")

            for model in MODEL_ORDER:
                r = next((x for x in poi_summary if x["section_name"] == sec_name and x["model"] == model), None)
                if r is None:
                    continue
                f.write(
                    f"| {model} | {r['records']} | {r['gt_hits']} | {r['pred_hits']} | {r['tau_dist_W1']} | {r['valid']} |\n"
                )

            f.write("\n")


# ============================================================
# CLI / main
# ============================================================
ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(description="CFD paper evaluation: original metric and rollout numerics.")
    p.add_argument("--model", choices=MODEL_ORDER, help="Omit to evaluate all four models.")
    p.add_argument("--seeds", default="100,200,13532,123456,123123123")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/cylinder")
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/cylinder")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/cylinder")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--n-starts", type=int, default=32)
    p.add_argument("--pred-horizon", type=int, default=500)
    p.add_argument("--poincare-horizon", type=int, default=1000)
    p.add_argument("--nmse-windows", default="20,100,400")
    p.add_argument("--w2-blur", type=float, default=0.05)
    p.add_argument("--w2-max-points", type=int, default=1500)
    p.add_argument("--psd-nfft", type=int, default=2048)
    p.add_argument("--tcm-max-lag", type=int, default=200)
    p.add_argument("--tcm-truncate-first-nonpositive", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if any((args.output_dir / name).exists() for name in (
        "evaluation_manifest.json", "per_rollout_prediction_metrics_CFD.csv", "per_seed_poincare_DRTD_CFD.csv"
    )):
        raise FileExistsError("Output directory already contains evaluation results; choose a new --output-dir")
    args.nmse_windows = [int(x) for x in args.nmse_windows.split(",")]
    args.section_direction = 0
    if args.n_starts < 1 or args.pred_horizon < 1 or args.poincare_horizon < 1:
        raise ValueError("Rollout counts and horizons must be positive.")
    if not HAS_WELCH or not HAS_GEOMLOSS:
        raise RuntimeError("Paper evaluation requires scipy.signal.welch and geomloss; no metric fallback is allowed.")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    torch.set_num_threads(1)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_file = args.data_dir / "flow_sensors.p"
    data = load_p(data_file)
    training = np.asarray(data[0], dtype=np.float64)
    testing_true = np.asarray(data[1], dtype=np.float64)
    dt_data, dt_common = float(data[4]), float(data[6])
    t_ref = np.arange(len(testing_true), dtype=np.float64) * dt_data
    t_common = np.arange(0.0, t_ref[-1], dt_common)
    ref_common = interp_traj_to_times(testing_true, dt_data, t_common)
    seeds = [int(s) for s in args.seeds.split(",")]
    seed_folders = [str(args.checkpoint_dir / f"Seed{s}") for s in seeds]
    loaded, load_errors = load_all_models(seed_folders, training, ref_common.shape[1], device, dt_common,
                                          models=[args.model] if args.model else MODEL_ORDER)
    pred_rows, pred_errors = evaluate_prediction_metrics(loaded, ref_common, dt_common, args, device)
    poi_rows, poi_errors = evaluate_full_horizon_poincare(loaded, ref_common, dt_common, args, device)
    pred_summary = summarize_prediction(pred_rows, args.nmse_windows)
    poi_summary = summarize_poincare(poi_rows)
    for name, rows in [("per_rollout_prediction_metrics_CFD.csv", pred_rows),
                       ("per_seed_poincare_DRTD_CFD.csv", poi_rows),
                       ("summary_prediction_metrics_CFD.csv", pred_summary),
                       ("summary_poincare_DRTD_CFD.csv", poi_summary)]:
        write_csv(args.output_dir / name, rows)
    errors = load_errors + pred_errors + poi_errors
    (args.output_dir / "eval_errors_CFD.json").write_text(json.dumps(errors, indent=2) + "\n")
    write_markdown_summary(args.output_dir / "summary_CFD.md", pred_summary, poi_summary, args.nmse_windows)
    manifest = {"arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                "device": device, "torch": torch.__version__, "numpy": np.__version__,
                "elapsed_seconds": time.perf_counter() - started,
                "data_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
                "checkpoints": {x["checkpoint"]: hashlib.sha256(Path(x["checkpoint"]).read_bytes()).hexdigest() for x in loaded},
                "prediction_rows": len(pred_rows), "poincare_rows": len(poi_rows),
                "section_dims": [6,4,5], "section_value": 4.0, "section_direction": 0,
                "error_count": len(errors)}
    (args.output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(pred_rows)} prediction rows and {len(poi_rows)} DRTD rows to {args.output_dir}")
    if errors:
        raise SystemExit(f"Evaluation had {len(errors)} errors; inspect eval_errors_CFD.json.")


if __name__ == "__main__":
    main()
