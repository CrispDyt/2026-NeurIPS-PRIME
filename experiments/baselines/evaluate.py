#!/usr/bin/env python3
"""
Evaluate saved heterogeneous Lorenz baselines together with a saved PRIME implicit model
from the SAME 30 rollout start times.

Expected directory layout, if using the previous training script:

  lorenz_heterogeneous_baselines/
  ├── checkpoints/
  │   ├── scaler.pkl
  │   ├── NGRC-NVAR_seed1000.p
  │   ├── Reservoir-ESN_seed1000.p
  │   ├── LSTM_seed1000.p
  │   ├── Koopman-AE_seed1000.p
  │   ├── DySLIM-style_seed1000.p
  │   └── Implicit_BEST.p              # copy your PRIME implicit checkpoint here
  └── run_config.json

Typical use:

  python eval_lorenz_heterogeneous_plus_prime.py \
      --test_path test_data.npy \
      --out_dir lorenz_heterogeneous_baselines \
      --seed 1000 \
      --eval_starts 30 \
      --horizon 500

If your test_data.npy is full Lorenz (T,3), the script uses [X,Z] by default.
If your test_data.npy is already partial (T,2), it uses both columns.

Evaluation protocol:
  - A shared list of 30 first-predicted time indices t_start is sampled once.
  - For history-based baselines, the history window ends at t_start-1.
  - For PRIME, only y_{t_start-1} is used, then the model is stepped forward.
  - Metrics are computed against y_{t_start:t_start+horizon}.

This means all models are evaluated from the same 30 forecast start times.  LSTM and
Koopman-AE still use their history windows by design, so this is a common-start-time
comparison, not a same-information comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import time
import csv
import io
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sys
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler

# The heterogeneous model classes and metric helpers live in the previous training script.
# Keep this script in the same directory as lorenz_heterogeneous_baselines.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import lorenz_heterogeneous_baselines as hb
except Exception as e:
    raise ImportError(
        "Could not import lorenz_heterogeneous_baselines.py. Put this evaluator in the "
        "same directory as lorenz_heterogeneous_baselines.py, or add that directory to PYTHONPATH.\n"
        f"Original import error: {e}"
    )


ROOT = Path(__file__).resolve().parents[2]

# -----------------------------------------------------------------------------
# Generic loading helpers
# -----------------------------------------------------------------------------

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_pickle_cpu(path: Path):
    with open(path, "rb") as f:
        return CPU_Unpickler(f).load()


def smart_load(path: Path, map_location="cpu"):
    path = Path(path)
    last_err = None
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e:
        last_err = e
    try:
        return load_pickle_cpu(path)
    except Exception as e:
        # Some classical baselines are ordinary pickle files.
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load checkpoint {path}\n"
                f"torch.load error: {last_err}\n"
                f"CPU pickle error: {e}\n"
                f"normal pickle error: {e2}"
            )


def load_array(path: str) -> np.ndarray:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    suf = p.suffix.lower()
    if suf == ".npy":
        arr = np.load(p, allow_pickle=True)
    elif suf == ".npz":
        obj = np.load(p, allow_pickle=True)
        for key in ["test_data", "train_data", "test", "train", "data", "X", "Y"]:
            if key in obj:
                arr = obj[key]
                break
        else:
            keys = list(obj.keys())
            if len(keys) != 1:
                raise KeyError(f"Could not infer array key in {p}; keys={keys}")
            arr = obj[keys[0]]
    elif suf in {".p", ".pkl", ".pickle"}:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            for key in ["test_data", "train_data", "test", "train", "data", "X", "Y"]:
                if key in obj:
                    arr = obj[key]
                    break
            else:
                raise KeyError(f"Could not infer array key in {p}; keys={list(obj.keys())}")
        else:
            arr = obj
    else:
        raise ValueError(f"Unsupported data file extension: {suf}")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (T,d), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Data contains non-finite values: {p}")
    return arr


def parse_obs_cols(obs_cols: str, d: int) -> Optional[List[int]]:
    obs_cols = str(obs_cols or "").strip()
    if obs_cols == "":
        return None
    cols = [int(s.strip()) for s in obs_cols.split(",") if s.strip()]
    for c in cols:
        if c < 0 or c >= d:
            raise ValueError(f"obs col {c} out of bounds for data dimension {d}")
    return cols


def select_obs(arr: np.ndarray, obs_cols: Optional[List[int]]) -> np.ndarray:
    if obs_cols is not None:
        return arr[:, obs_cols]
    if arr.shape[1] == 2:
        return arr
    if arr.shape[1] >= 3:
        return arr[:, [0, 2]]  # default Lorenz partial observation [X,Z]
    return arr


# -----------------------------------------------------------------------------
# Heterogeneous baseline checkpoint loading
# -----------------------------------------------------------------------------

def infer_lstm_dims(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    hidden = int(sd["lstm.weight_hh_l0"].shape[1])
    layers = len([k for k in sd.keys() if k.startswith("lstm.weight_ih_l")])
    return hidden, layers


def load_heterogeneous_model(path: Path, device: torch.device):
    obj = smart_load(path, map_location="cpu")
    if not isinstance(obj, dict) or "type" not in obj:
        raise ValueError(f"Not a heterogeneous-baseline checkpoint: {path}")

    typ = obj["type"]

    if typ == "nvar":
        return hb.NVARModel(W=obj["W"], seq_len=obj["seq_len"], dim=obj["dim"], order=obj["order"])

    if typ == "esn":
        return hb.ESNModel(
            Win=obj["Win"], Wres=obj["Wres"], bias=obj["bias"], Wout=obj["Wout"],
            leak=obj["leak"], seq_len=obj["seq_len"], dim=obj["dim"]
        )

    if typ == "lstm":
        sd = obj["state_dict"]
        dim = int(obj["dim"])
        hidden, layers = infer_lstm_dims(sd)
        net = hb.LSTMForecaster(dim=dim, hidden=hidden, layers=layers).to(device)
        net.load_state_dict(sd)
        net.eval()
        return hb.LSTMRolloutModel(net=net, seq_len=int(obj["seq_len"]), dim=dim, device=device)

    if typ == "koopman":
        sd = obj["state_dict"]
        seq_len = int(obj["seq_len"])
        dim = int(obj["dim"])
        in_dim = int(obj.get("in_dim", seq_len * dim))
        hidden = int(sd["encoder.0.weight"].shape[0])
        latent = int(sd["encoder.4.weight"].shape[0])
        net = hb.KoopmanAE(in_dim=in_dim, obs_dim=dim, latent=latent, hidden=hidden).to(device)
        net.load_state_dict(sd)
        net.eval()
        return hb.KoopmanRolloutModel(net=net, seq_len=seq_len, dim=dim, device=device)

    if typ == "dyslim":
        sd = obj["state_dict"]
        dim = int(obj["dim"])
        hidden = int(sd["net.0.weight"].shape[0])
        net = hb.MLPMap(dim=dim, hidden=hidden).to(device)
        net.load_state_dict(sd)
        net.eval()
        return hb.DySLIMRolloutModel(net=net, seq_len=int(obj["seq_len"]), dim=dim, device=device)

    raise ValueError(f"Unknown checkpoint type {typ} in {path}")


# -----------------------------------------------------------------------------
# PRIME implicit checkpoint loading
# -----------------------------------------------------------------------------

def infer_mlp_dims_from_state_dict(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    # PRIME MLP is usually nn.Sequential: 0 Linear, 2 Linear, 4 Linear, 6 Linear.
    wkeys = []
    for k in sd.keys():
        if k.endswith(".weight"):
            p = k.split(".")[0]
            if p.isdigit():
                wkeys.append((int(p), k))
    wkeys = sorted(wkeys, key=lambda x: x[0])
    if not wkeys:
        raise ValueError("Could not infer MLP architecture from state_dict keys.")
    W0 = sd[wkeys[0][1]]
    Wlast = sd[wkeys[-1][1]]
    width = int(W0.shape[0])
    d_in = int(W0.shape[1])
    d_out = int(Wlast.shape[0])
    return d_in, width, d_out


def build_prime_net(d: int, width: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, d),
    )


def extract_prime_state_dict_and_scaler(path: Path, fallback_scaler=None):
    obj = smart_load(path, map_location="cpu")
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
        raise RuntimeError(f"Could not find PRIME state_dict in {path}")
    if scaler is None:
        scaler = fallback_scaler
    if scaler is None:
        raise RuntimeError(
            f"PRIME checkpoint {path} does not contain scaler/transformer and no fallback scaler was provided."
        )
    return obj, sd, scaler


class PrimeImplicitRolloutModel:
    """PRIME/NODE-style MLP vector-field model in scaled observation coordinates."""
    def __init__(self, net: nn.Module, scaler, dt_obs: float, dim: int, device: torch.device, name="PRIME-implicit"):
        self.net = net.to(device)
        self.net.eval()
        self.scaler = scaler
        self.dt_obs = float(dt_obs)
        self.dim = int(dim)
        self.seq_len = 1
        self.name = name
        self.device = device

    @torch.no_grad()
    def rollout_scaled(self, hist_scaled: np.ndarray, horizon: int) -> np.ndarray:
        x = torch.tensor(np.asarray(hist_scaled[-1:], dtype=np.float32), dtype=torch.float32, device=self.device)
        preds = []
        for _ in range(int(horizon)):
            x = x + self.dt_obs * self.net(x)
            preds.append(x.detach().cpu().numpy()[0])
        return np.asarray(preds, dtype=np.float64)


def load_prime_model(path: Path, device: torch.device, dt_obs: float, fallback_scaler=None):
    obj, sd, scaler = extract_prime_state_dict_and_scaler(path, fallback_scaler=fallback_scaler)
    d_in, width, d_out = infer_mlp_dims_from_state_dict(sd)
    if d_in != d_out:
        raise ValueError(f"PRIME checkpoint has d_in={d_in}, d_out={d_out}; expected equal dimensions.")
    net = build_prime_net(d_in, width=width).to(device)
    net.load_state_dict(sd)
    net.eval()
    return PrimeImplicitRolloutModel(net=net, scaler=scaler, dt_obs=dt_obs, dim=d_in, device=device), obj


# -----------------------------------------------------------------------------
# Shared-start evaluation
# -----------------------------------------------------------------------------

def finite_mean(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def finite_std(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if len(arr) else float("nan")


def fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    if not np.isfinite(mean):
        return "nan"
    if not np.isfinite(std):
        return f"{mean:.{digits}g}"
    return f"{mean:.{digits}g} ± {std:.{digits}g}"


def choose_shared_tstarts(T: int, max_seq_len: int, horizon: int, n_starts: int, seed: int) -> np.ndarray:
    # t_start is the first predicted time index. We need [t_start-hist, ..., t_start-1]
    # for the largest history model and [t_start, ..., t_start+horizon-1] as reference.
    lo = int(max_seq_len)
    hi = int(T - horizon)
    if hi <= lo:
        raise ValueError(f"Test series too short for max_seq_len={max_seq_len}, horizon={horizon}: T={T}")
    valid = np.arange(lo, hi, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = min(int(n_starts), len(valid))
    return np.sort(rng.choice(valid, size=n, replace=False)).astype(np.int64)


def sanitize_scaled(pred_s: np.ndarray, threshold: float) -> Tuple[np.ndarray, bool]:
    pred_s = np.asarray(pred_s, dtype=np.float64)
    diverged = False
    if (not np.all(np.isfinite(pred_s))) or (np.nanmax(np.abs(pred_s)) > float(threshold)):
        diverged = True
    pred_s = np.nan_to_num(pred_s, nan=0.0, posinf=float(threshold), neginf=-float(threshold))
    pred_s = np.clip(pred_s, -float(threshold), float(threshold))
    return pred_s, diverged


def inverse_transform_safe(scaler, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    shp = X.shape
    X2 = X.reshape(-1, shp[-1])
    Y2 = scaler.inverse_transform(X2)
    return Y2.reshape(shp)


def rollout_one(model, scaler, test_raw: np.ndarray, t_start: int, horizon: int, threshold: float) -> Tuple[np.ndarray, bool]:
    hist_raw = test_raw[t_start - model.seq_len:t_start]
    if len(hist_raw) != model.seq_len:
        raise RuntimeError(f"Bad history length for {model.name}: got {len(hist_raw)}, need {model.seq_len}")
    hist_s = scaler.transform(hist_raw)
    pred_s = model.rollout_scaled(hist_s, horizon)
    pred_s, diverged = sanitize_scaled(pred_s, threshold=threshold)
    pred_raw = inverse_transform_safe(scaler, pred_s)
    return pred_raw, diverged


def evaluate_loaded_models(models: List[Tuple[str, object, object]], test_raw: np.ndarray,
                           t_starts: np.ndarray, args, device: torch.device) -> List[Dict[str, object]]:
    windows = [100, 400]
    horizon = int(args.horizon)
    rows = []

    plot_dir = Path(args.out_dir) / "combined_eval_plots"
    pred_dir = Path(args.out_dir) / "combined_eval_rollouts"
    plot_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Save reference starts once.
    np.save(pred_dir / f"shared_tstarts_seed{args.start_seed}.npy", t_starts)

    for name, model, scaler in models:
        print("\n" + "=" * 100)
        print(f"Evaluating {name} from {len(t_starts)} shared starts")
        print("=" * 100)

        nmse_vals = {w: [] for w in windows}
        w2_vals, psd_vals, drtd_vals, hre_vals, tailstd_vals = [], [], [], [], []
        div_count = 0
        first_ref = None
        first_pred = None

        # Store all evaluated rollouts for optional later plotting.
        all_preds = []
        all_refs = []

        for i, t0 in enumerate(t_starts):
            ref_raw = test_raw[t0:t0 + horizon]
            pred_raw, diverged = rollout_one(model, scaler, test_raw, t_start=int(t0), horizon=horizon, threshold=args.div_threshold)
            if diverged:
                div_count += 1

            all_preds.append(pred_raw)
            all_refs.append(ref_raw)

            for w in windows:
                nmse_vals[w].append(hb.nmse(ref_raw[:w], pred_raw[:w]))

            # Long-structure metrics on the full evaluation horizon.
            L = min(len(ref_raw), len(pred_raw), horizon)
            ref_L = ref_raw[:L]
            pred_L = pred_raw[:L]

            if L > args.w2_max_points:
                idx = np.linspace(0, L - 1, args.w2_max_points).astype(int)
                A = ref_L[idx]
                B = pred_L[idx]
            else:
                A, B = ref_L, pred_L

            w2_vals.append(hb.sinkhorn_or_sliced(A, B, device=device, blur=args.w2_blur, fallback_proj=args.sliced_proj))
            psd_vals.append(hb.psd_logl1_distance(ref_L, pred_L, dt=args.dt_obs, nfft=args.psd_nfft))
            rec = hb.recurrence_metrics(ref_L, pred_L, dt=args.dt_obs)
            drtd_vals.append(rec.get("Mean_DRTD", np.nan))
            hre_vals.append(rec.get("Mean_HRE", np.nan))
            tailstd_vals.append(hb.collapse_score(pred_L, tail=100))

            if first_ref is None:
                first_ref = ref_raw.copy()
                first_pred = pred_raw.copy()

        # Convert lists to arrays and save.
        all_preds_arr = np.asarray(all_preds, dtype=np.float64)
        all_refs_arr = np.asarray(all_refs, dtype=np.float64)
        safe_name = name.replace("/", "-").replace(" ", "_")
        np.savez_compressed(
            pred_dir / f"{safe_name}_shared_rollouts.npz",
            model=name,
            t_starts=t_starts,
            pred=all_preds_arr,
            ref=all_refs_arr,
        )

        row: Dict[str, object] = {"model": name}
        for w in windows:
            m = finite_mean(nmse_vals[w])
            s = finite_std(nmse_vals[w])
            row[f"NMSE_{w}_mean"] = m
            row[f"NMSE_{w}_std"] = s
            row[f"NMSE_{w}"] = fmt_pm(m, s)

        row["Attractor_W2_mean"] = finite_mean(w2_vals)
        row["PSD_logL1_mean"] = finite_mean(psd_vals)
        row["Mean_DRTD_mean"] = finite_mean(drtd_vals)
        row["Mean_HRE_mean"] = finite_mean(hre_vals)
        row["TailStd_mean"] = finite_mean(tailstd_vals)
        row["Div"] = int(div_count)
        row["seq_len"] = int(model.seq_len)
        rows.append(row)

        if args.plot and first_ref is not None and first_pred is not None:
            save_individual_plot(plot_dir / f"{safe_name}_timeseries.png", name, first_ref, first_pred)

    if args.plot:
        save_overlay_plots(models, test_raw, t_starts[0], args, plot_dir)

    return rows


def save_individual_plot(path: Path, name: str, ref: np.ndarray, pred: np.ndarray):
    d = ref.shape[1]
    t = np.arange(len(ref))
    fig, axes = plt.subplots(d, 1, figsize=(10.5, 2.5 * d), dpi=150, sharex=True)
    if d == 1:
        axes = [axes]
    for ch in range(d):
        axes[ch].plot(t, ref[:, ch], color="black", lw=1.3, label="GT")
        axes[ch].plot(t, pred[:, ch], lw=1.0, label=name)
        axes[ch].set_ylabel(f"ch={ch}")
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("rollout step")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def clip_to_ref_range(Y: np.ndarray, ref: np.ndarray, margin_mult: float = 0.25) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float64).copy()
    lo = np.nanmin(ref, axis=0)
    hi = np.nanmax(ref, axis=0)
    span = np.maximum(hi - lo, 1e-8)
    return np.clip(Y, (lo - margin_mult * span)[None, :], (hi + margin_mult * span)[None, :])


def save_overlay_plots(models: List[Tuple[str, object, object]], test_raw: np.ndarray, t_start: int, args, plot_dir: Path):
    horizon = min(int(args.plot_horizon), len(test_raw) - int(t_start))
    ref = test_raw[int(t_start):int(t_start) + horizon]
    preds = {}
    for name, model, scaler in models:
        pred, _ = rollout_one(model, scaler, test_raw, t_start=int(t_start), horizon=horizon, threshold=args.div_threshold)
        preds[name] = pred

    d = ref.shape[1]
    t = np.arange(horizon)
    fig, axes = plt.subplots(d, 1, figsize=(12, 3.0 * d), dpi=160, sharex=True)
    if d == 1:
        axes = [axes]
    for ch in range(d):
        axes[ch].plot(t, ref[:, ch], color="black", lw=1.6, label="GT")
        for name, pred in preds.items():
            # Clip only for plotting so unstable baselines do not destroy the axis.
            pclip = clip_to_ref_range(pred, ref)
            axes[ch].plot(t, pclip[:, ch], lw=0.95, alpha=0.85, label=name)
        axes[ch].set_ylabel(f"ch={ch}")
    axes[0].legend(ncol=2, fontsize=7.5, frameon=True)
    axes[-1].set_xlabel("rollout step")
    fig.suptitle("Shared-start autonomous rollouts: heterogeneous baselines + PRIME")
    fig.tight_layout()
    fig.savefig(plot_dir / "shared_start_timeseries_overlay.png", bbox_inches="tight")
    plt.close(fig)

    if d >= 2:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5), dpi=170)
        ax.plot(ref[:, 0], ref[:, 1], color="black", lw=1.5, label="GT")
        for name, pred in preds.items():
            pclip = clip_to_ref_range(pred, ref)
            ax.plot(pclip[:, 0], pclip[:, 1], lw=0.95, alpha=0.85, label=name)
        ax.set_xlabel("ch=0")
        ax.set_ylabel("ch=1")
        ax.set_title("Shared-start phase portrait")
        ax.legend(fontsize=7.5, frameon=True)
        fig.tight_layout()
        fig.savefig(plot_dir / "shared_start_phase_overlay.png", bbox_inches="tight")
        plt.close(fig)


def write_outputs(rows: List[Dict[str, object]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "combined_results_with_PRIME.csv"
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    latex_path = out_dir / "combined_latex_rows_with_PRIME.txt"
    with open(latex_path, "w") as f:
        f.write("% Model & NMSE100 & NMSE400 & W2 & PSD & DRTD & HRE & TailStd & Div \\\\ \n")
        for r in rows:
            f.write(
                f"{r['model']} & {r.get('NMSE_100','NA')} & {r.get('NMSE_400','NA')} & "
                f"{float(r.get('Attractor_W2_mean', float('nan'))):.4g} & "
                f"{float(r.get('PSD_logL1_mean', float('nan'))):.4g} & "
                f"{float(r.get('Mean_DRTD_mean', float('nan'))):.4g} & "
                f"{float(r.get('Mean_HRE_mean', float('nan'))):.4g} & "
                f"{float(r.get('TailStd_mean', float('nan'))):.4g} & "
                f"{int(r.get('Div', 0))} \\\\ \n"
            )

    print(f"\nSaved combined CSV: {csv_path}")
    print(f"Saved LaTeX rows: {latex_path}")


def print_summary(rows: List[Dict[str, object]]):
    print("\n" + "=" * 125)
    print("SHARED-START HETEROGENEOUS BASELINES + PRIME SUMMARY")
    print("=" * 125)
    header = (
        f"{'model':22s} | {'NMSE_100':18s} | {'NMSE_400':18s} | "
        f"{'W2':10s} | {'PSD':10s} | {'DRTD':10s} | {'HRE':10s} | {'TailStd':10s} | {'Div':5s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{str(r['model']):22s} | {str(r.get('NMSE_100','')):18s} | {str(r.get('NMSE_400','')):18s} | "
            f"{float(r.get('Attractor_W2_mean', float('nan'))):10.4g} | "
            f"{float(r.get('PSD_logL1_mean', float('nan'))):10.4g} | "
            f"{float(r.get('Mean_DRTD_mean', float('nan'))):10.4g} | "
            f"{float(r.get('Mean_HRE_mean', float('nan'))):10.4g} | "
            f"{float(r.get('TailStd_mean', float('nan'))):10.4g} | "
            f"{int(r.get('Div', 0)):5d}"
        )


# -----------------------------------------------------------------------------
# Checkpoint discovery
# -----------------------------------------------------------------------------

def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def find_prime_ckpt(ckpt_dir: Path, user_path: str = "") -> Optional[Path]:
    if user_path:
        p = Path(user_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Explicit PRIME checkpoint does not exist: {p}")
        return p
    candidates = [
        ckpt_dir / "Implicit_BEST.p",
        ckpt_dir / "implicit_BEST.p",
        ckpt_dir / "IMPLICIT_BEST.p",
        ckpt_dir / "PRIME_implicit_BEST.p",
        ckpt_dir / "PRIME-implicit_BEST.p",
    ]
    p = first_existing(candidates)
    if p is not None:
        return p
    patterns = ["*Implicit*BEST*.p", "*implicit*BEST*.p", "*PRIME*BEST*.p", "*prime*BEST*.p"]
    found = []
    for pat in patterns:
        found.extend(sorted(ckpt_dir.glob(pat), key=lambda x: x.stat().st_mtime))
    return found[-1] if found else None


def load_all_models(args, device: torch.device, test_dim: int):
    out_dir = Path(args.out_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    scaler_path = ckpt_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing heterogeneous scaler: {scaler_path}")
    with open(scaler_path, "rb") as f:
        hetero_scaler = pickle.load(f)

    ordered = [
        ("NGRC-NVAR", ckpt_dir / f"NGRC-NVAR_seed{args.seed}.p"),
        ("Reservoir-ESN", ckpt_dir / f"Reservoir-ESN_seed{args.seed}.p"),
        ("LSTM", ckpt_dir / f"LSTM_seed{args.seed}.p"),
        ("Koopman-AE", ckpt_dir / f"Koopman-AE_seed{args.seed}.p"),
        ("DySLIM-style", ckpt_dir / f"DySLIM-style_seed{args.seed}.p"),
    ]

    models: List[Tuple[str, object, object]] = []
    for name, path in ordered:
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {name}: {path}")
        model = load_heterogeneous_model(path, device=device)
        models.append((model.name, model, hetero_scaler))
        print(f"[OK] loaded {model.name:16s} from {path}")

    prime_path = find_prime_ckpt(ckpt_dir, args.prime_ckpt)
    if prime_path is None:
        raise FileNotFoundError(f"Missing PRIME checkpoint in {ckpt_dir}")
    else:
        prime_model, prime_obj = load_prime_model(prime_path, device=device, dt_obs=args.dt_obs, fallback_scaler=hetero_scaler)
        if prime_model.dim != test_dim:
            raise ValueError(
                f"PRIME dimension {prime_model.dim} does not match selected test dimension {test_dim}. "
                f"Check obs_cols and checkpoint: {prime_path}"
            )
        models.append((prime_model.name, prime_model, prime_model.scaler))
        print(f"[OK] loaded {prime_model.name:16s} from {prime_path}")

    if not models:
        raise RuntimeError(f"No checkpoints loaded from {ckpt_dir}")
    return models, hetero_scaler


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Evaluate heterogeneous baselines + PRIME from shared rollout starts.")
    p.add_argument("--test_path", "--test-path", type=str, default=str(ROOT / "data/lorenz63/test_data.npy"))
    p.add_argument("--out_dir", "--output-dir", type=str, default=str(ROOT / "results/baselines"))
    p.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints/baselines"))
    p.add_argument("--obs_cols", type=str, default="", help="Comma-separated columns, e.g. '0,2'. Empty: use all if d=2 else [0,2].")
    p.add_argument("--seed", type=int, default=1000, help="Checkpoint seed suffix for heterogeneous baselines.")
    p.add_argument("--start_seed", type=int, default=1123, help="Seed used to sample the shared 30 rollout starts.")
    p.add_argument("--eval_starts", type=int, default=30)
    p.add_argument("--horizon", type=int, default=500, help="Horizon for metrics. Must be >= 400 for NMSE400.")
    p.add_argument("--plot_horizon", type=int, default=700)
    p.add_argument("--dt_obs", type=float, default=0.1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--prime_ckpt", type=str, default="", help="Optional explicit path to Implicit_BEST.p")
    p.add_argument("--div_threshold", type=float, default=1e3, help="Scaled-value threshold for divergence/safety clipping.")
    p.add_argument("--w2_blur", type=float, default=0.05)
    p.add_argument("--w2_max_points", type=int, default=700)
    p.add_argument("--sliced_proj", type=int, default=128)
    p.add_argument("--psd_nfft", type=int, default=2048)
    p.add_argument("--plot", action="store_true", default=True)
    p.add_argument("--no_plot", dest="plot", action="store_false")
    return p


def main():
    args = build_parser().parse_args()
    for subdir in ("combined_eval", "combined_eval_rollouts", "combined_eval_plots"):
        previous = Path(args.out_dir) / subdir
        if previous.exists() and any(previous.iterdir()):
            raise FileExistsError("Output directory already contains evaluation results; choose a new --output-dir")
    started = time.perf_counter()
    torch.set_num_threads(1)
    if not hb.HAS_GEOMLOSS:
        raise RuntimeError("Install GeomLoss for the paper transport metric.")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    print("device:", device)
    print("geomloss available:", getattr(hb, "HAS_GEOMLOSS", False))

    test0 = load_array(args.test_path)
    obs_cols = parse_obs_cols(args.obs_cols, test0.shape[1])
    test_raw = select_obs(test0, obs_cols)
    print(f"test observation shape: {test_raw.shape}")

    models, _ = load_all_models(args, device=device, test_dim=test_raw.shape[1])
    max_seq = max(int(m.seq_len) for _, m, _ in models)
    if args.horizon < 400:
        raise ValueError("--horizon must be at least 400 to report NMSE400.")

    t_starts = choose_shared_tstarts(
        T=len(test_raw),
        max_seq_len=max_seq,
        horizon=max(int(args.horizon), 400),
        n_starts=args.eval_starts,
        seed=args.start_seed,
    )
    print(f"shared first-predicted t_start indices ({len(t_starts)}): {t_starts[:10]} ...")
    print(f"max_seq_len={max_seq}; sequence models use history ending at t_start-1; PRIME uses y[t_start-1].")

    rows = evaluate_loaded_models(models, test_raw, t_starts, args=args, device=device)
    print_summary(rows)

    out_eval_dir = Path(args.out_dir) / "combined_eval"
    write_outputs(rows, out_eval_dir)
    manifest = {"arguments": vars(args), "device": str(device), "torch": torch.__version__,
                "numpy": np.__version__, "elapsed_seconds": time.perf_counter()-started,
                "data_sha256": hashlib.sha256(Path(args.test_path).read_bytes()).hexdigest(),
                "checkpoint_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(args.checkpoint_dir).glob("*") if p.is_file()},
                "shared_starts": t_starts.tolist(), "model_count": len(models)}
    (out_eval_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("Saved plots/rollouts under:")
    print("  ", Path(args.out_dir) / "combined_eval_plots")
    print("  ", Path(args.out_dir) / "combined_eval_rollouts")


if __name__ == "__main__":
    main()
