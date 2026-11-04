#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate 4 Lorenz models trained with 5 random seeds.

Two evaluation protocols are used deliberately:

A) Multi-start prediction/structure metrics
   - For each model and each training-seed folder, rollout from 32 different test starts.
   - This gives 5 * 32 = 160 rollouts per model.
   - Report:
       NMSE_20, NMSE_100, NMSE_400: mean ± std over 160 rollouts
       Attractor_W2, PSD_logL1, DySLIM-style MELR, DySLIM-style MELRw, TCM: mean ± std over 160 rollouts

B) Full-horizon Poincare / return-time metrics
   - For each model and each training-seed folder, rollout only once from test_data[0].
   - Horizon is full test horizon by default: K = len(test_data) - 1.
   - This matches the original Poincare-table protocol where GT hits are computed on
     the full test trajectory rather than on short local segments.
   - Report mean over 5 training seeds:
       n_hits and tau_dist_W1 on x=10
       n_hits and tau_dist_W1 on z=40
       peak hits and tau_dist_W1 for the unimodal peak-return section.

Expected folder structure:
  ./1-step MSE 33/DIM_BEST.pt
  ./1-step MSE 33/DIM_MSE_BEST.pt
  ./1-step MSE 33/Implicit_BEST.p
  ./1-step MSE 33/NODE_BEST.p
  ... same for other seed folders
"""

import io
import pickle
import warnings
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MaxAbsScaler

try:
    from scipy.signal import welch, find_peaks
    HAS_SCIPY_SIGNAL = True
except Exception as e:
    HAS_SCIPY_SIGNAL = False
    print("[warn] scipy.signal import failed; using FFT PSD fallback and simple peak fallback")
    print("[warn]", repr(e))

try:
    from geomloss import SamplesLoss
    HAS_GEOMLOSS = True
except Exception as e:
    HAS_GEOMLOSS = False
    print("[warn] geomloss import failed; using sliced-W2 fallback")
    print("[warn]", repr(e))


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
DEFAULT_SEED_FOLDERS = [
    "1-step MSE 33",
    "1-step MSE 123",
    "1-step MSE 1000",
    "1-step MSE 13532",
    "1-step MSE 123456",
]

MODEL_FILES = {
    "PRIME": "Implicit_BEST.p",
    "DIM": "DIM_BEST.pt",
    "DIM+MSE": "DIM_MSE_BEST.pt",
    "MSE": "NODE_BEST.p",
}

MODEL_ORDER = ["PRIME", "DIM", "DIM+MSE", "MSE"]
NMSE_WINDOWS = [20, 100, 400]


# ---------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_p_cpu(path):
    with open(path, "rb") as f:
        return CPU_Unpickler(f).load()


def smart_load(path):
    if np.__version__.startswith("1."):
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
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


def extract_config(obj):
    cfg = {}
    if isinstance(obj, dict):
        for key in ["config", "cfg", "args", "hyperparams", "params"]:
            if key in obj and isinstance(obj[key], dict):
                cfg.update(obj[key])
        for key in ["dt_obs", "dt_base", "sub", "tau_pair", "seed"]:
            if key in obj and key not in cfg:
                cfg[key] = obj[key]
    return cfg


def get_cfg(cfg, keys, default):
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


def load_model_ckpt(path, train_data):
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
        print(f"[warn] no scaler in {path}; fitting MaxAbsScaler on train_data")
        scaler = MaxAbsScaler().fit(np.asarray(train_data, dtype=np.float64))

    cfg = extract_config(obj)
    return obj, sd, scaler, cfg


def infer_mlp_dims_from_state_dict(sd: dict):
    sd = strip_module_prefix(sd)
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
    d_in = int(W0.shape[1])
    d_out = int(Wlast.shape[0])
    return d_in, width, d_out


def build_net(d, width=100):
    return nn.Sequential(
        nn.Linear(d, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, d),
    )


@torch.no_grad()
def rollout_from_x0(net, scaler, x0_raw, K, dt_obs, device, dt_base=None, sub=None):
    """
    Rollout in scaled space. If dt_base/sub are provided, one observed step consists
    of `sub` micro-steps of size dt_base. Otherwise it uses x <- x + dt_obs*f(x).
    """
    if dt_base is None:
        dt_base = dt_obs
    if sub is None:
        sub = int(round(float(dt_obs) / float(dt_base))) if float(dt_base) > 0 else 1
    sub = max(1, int(sub))

    x0_s = scaler.transform(np.asarray(x0_raw).reshape(1, -1)).astype(np.float32)
    x = torch.tensor(x0_s, dtype=torch.float32, device=device)

    pred_s = np.zeros((K + 1, x.shape[1]), dtype=np.float32)
    for k in range(K + 1):
        pred_s[k] = x.detach().cpu().numpy()[0]
        if k == K:
            break
        for _ in range(sub):
            x = x + float(dt_base) * net(x)

    pred = scaler.inverse_transform(pred_s).astype(np.float64)
    return pred


def load_net_scaler_cfg(path, train_data, device, default_dt_obs):
    obj, sd, scaler, cfg = load_model_ckpt(path, train_data)
    d_in, width, d_out = infer_mlp_dims_from_state_dict(sd)
    if d_in != d_out:
        raise ValueError(f"Model at {path} is not square: d_in={d_in}, d_out={d_out}")

    net = build_net(d_in, width=width).to(device)
    net.load_state_dict(sd)
    net.eval()

    dt_obs_model = float(get_cfg(cfg, ["dt_obs", "dt"], default_dt_obs))
    dt_base_model = float(get_cfg(cfg, ["dt_base"], dt_obs_model))
    sub_model = int(get_cfg(cfg, ["sub"], int(round(dt_obs_model / dt_base_model))))
    sub_model = max(1, sub_model)

    return net, scaler, cfg, dt_obs_model, dt_base_model, sub_model


# ---------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------
def nmse_value(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    L = min(len(gt), len(pred))
    gt = gt[:L]
    pred = pred[:L]
    mse = np.mean((gt - pred) ** 2)
    denom = np.mean(gt ** 2) + 1e-12
    return float(mse / denom)


def psd_logl1_distance(ref, pred, dt, nfft=2048, channels=None):
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

        n = min(nfft, L)
        x = x[:n]
        y = y[:n]

        if len(x) < 8 or len(y) < 8:
            dists.append(np.nan)
            continue

        if HAS_SCIPY_SIGNAL:
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



# ---------------------------------------------------------------------
# DySLIM-style spectral / temporal metrics
# ---------------------------------------------------------------------
def _safe_log_ratio(pred_energy, ref_energy, eps=1e-12):
    pred_energy = np.asarray(pred_energy, dtype=np.float64)
    ref_energy = np.asarray(ref_energy, dtype=np.float64)
    return np.abs(np.log((pred_energy + eps) / (ref_energy + eps)))


def radial_energy_spectrum_from_snapshots_1d_state(
    traj,
    eps=1e-12,
    drop_zero_mode=False,
):
    """
    DySLIM-style snapshot Fourier energy spectrum for a 1D vector state.

    This implements Eq. (22) in the DySLIM appendix for a 1D grid: each
    snapshot u is transformed along its state-coordinate axis, and energies
    with the same absolute wave number are summed. The final spectrum is
    averaged over rollout snapshots.

    traj: array, shape (T, d)
        T snapshots of a d-dimensional state. For Lorenz this treats the
        observed vector coordinates as a 1D grid of d points. This is the
        direct snapshot-Fourier analogue of the DySLIM metric; it is NOT a
        temporal FFT over rollout time.

    returns: E, shape (floor(d/2)+1,) or one fewer if drop_zero_mode=True.
    """
    X = np.asarray(traj, dtype=np.float64)

    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f"Expected trajectory shape (T, d), got {X.shape}")

    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    if len(X) == 0:
        return np.zeros((0,), dtype=np.float64)

    T, d = X.shape
    if d < 1:
        return np.zeros((0,), dtype=np.float64)

    # Snapshot-wise Fourier transform over state-coordinate axis.
    # Full FFT lets us explicitly sum +k and -k into the same radial bin.
    F = np.fft.fft(X, axis=1)
    P = np.abs(F) ** 2

    Kmax = d // 2
    E = []
    for k in range(Kmax + 1):
        if k == 0:
            idx = [0]
        elif (d % 2 == 0) and (k == d // 2):
            idx = [k]
        else:
            idx = [k, d - k]

        # Eq. (22): sum energy over modes with |K| = k,
        # then average this snapshot spectrum over rollout time.
        E_k = np.mean(np.sum(P[:, idx], axis=1))
        E.append(E_k)

    E = np.asarray(E, dtype=np.float64)

    if drop_zero_mode and len(E) > 1:
        E = E[1:]

    return E + float(eps)


def melr_melrw_distance(
    ref,
    pred,
    eps=1e-12,
    drop_zero_mode=False,
):
    """
    DySLIM MELR / MELRw from snapshot Fourier energy spectra.

    DySLIM Appendix C.3 defines
        E(K) = sum_{|K|=K} |uhat(K)|^2
        MELR = sum_K w_K |log(E_pred(K) / E_ref(K))|.

    Here:
      - MELR  uses unweighted weights w_K = 1 / #radial-bins.
      - MELRw uses weighted weights w_K = E_ref(K) / sum_K E_ref(K).

    For Lorenz, there is no spatial grid, so this function applies the
    DySLIM snapshot-Fourier formula to the observed state vector itself,
    treating state coordinates as a 1D grid. This is intentionally different
    from PSD_logL1, which remains a temporal Welch PSD metric.
    """
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    L = min(len(ref), len(pred))
    ref = ref[:L]
    pred = pred[:L]

    E_ref = radial_energy_spectrum_from_snapshots_1d_state(
        ref,
        eps=eps,
        drop_zero_mode=drop_zero_mode,
    )
    E_pred = radial_energy_spectrum_from_snapshots_1d_state(
        pred,
        eps=eps,
        drop_zero_mode=drop_zero_mode,
    )

    m = min(len(E_ref), len(E_pred))
    if m == 0:
        return float("nan"), float("nan")

    E_ref = E_ref[:m]
    E_pred = E_pred[:m]

    mask = np.isfinite(E_ref) & np.isfinite(E_pred) & (E_ref > 0) & (E_pred > 0)
    E_ref = E_ref[mask]
    E_pred = E_pred[mask]

    if len(E_ref) == 0:
        return float("nan"), float("nan")

    lr = _safe_log_ratio(E_pred, E_ref, eps=eps)

    # Unweighted DySLIM MELR: w_K = 1/card(K) over the radially binned K's.
    melr = float(np.mean(lr))

    # Weighted DySLIM MELRw: w_K = E_ref(K) / sum_K E_ref(K).
    weights = E_ref / (float(np.sum(E_ref)) + eps)
    melrw = float(np.sum(weights * lr))

    return melr, melrw


def autocorr_1d(x, max_lag, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    n = len(x)
    if n < 8:
        return np.zeros((0,), dtype=np.float64)
    c0 = float(np.dot(x, x) / n)
    if c0 < eps:
        return np.zeros((max_lag + 1,), dtype=np.float64)
    max_lag = int(min(max_lag, n - 2))
    rho = np.empty(max_lag + 1, dtype=np.float64)
    rho[0] = 1.0
    for lag in range(1, max_lag + 1):
        rho[lag] = float(np.dot(x[lag:], x[:-lag]) / (n - lag) / c0)
    return rho


def integrated_autocorr_time_1d(x, dt, max_lag=None, stop_first_nonpositive=True, eps=1e-12):
    """Finite-sample approximation of tau = dt * (1 + 2 * sum_i rho_i)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 8:
        return float("nan")
    if max_lag is None or int(max_lag) <= 0:
        max_lag = min(200, n // 2)
    else:
        max_lag = min(int(max_lag), n // 2)
    rho = autocorr_1d(x, max_lag=max_lag, eps=eps)
    if len(rho) == 0:
        return float("nan")
    if len(rho) == max_lag + 1 and np.allclose(rho, 0.0):
        return 0.0
    tail = rho[1:]
    if stop_first_nonpositive:
        bad = np.where(tail <= 0.0)[0]
        if len(bad):
            tail = tail[:bad[0]]
    tau = float(dt) * (1.0 + 2.0 * float(np.sum(tail)))
    return float(max(tau, 0.0)) if np.isfinite(tau) else float("nan")


def tcm_distance(ref, pred, dt, channels=None, max_lag=None, stop_first_nonpositive=True):
    """
    Time Correlation Metric (TCM): absolute difference between average
    per-channel autocorrelation times of ground truth and prediction.
    Lower is better.
    """
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
    tau_ref, tau_pred = [], []
    for ch in channels:
        tr = integrated_autocorr_time_1d(ref[:, ch], dt=dt, max_lag=max_lag, stop_first_nonpositive=stop_first_nonpositive)
        tp = integrated_autocorr_time_1d(pred[:, ch], dt=dt, max_lag=max_lag, stop_first_nonpositive=stop_first_nonpositive)
        if np.isfinite(tr) and np.isfinite(tp):
            tau_ref.append(tr)
            tau_pred.append(tp)
    if len(tau_ref) == 0:
        return float("nan")
    return float(abs(np.mean(tau_ref) - np.mean(tau_pred)))

def sliced_w2_cloud(A, B, n_proj=256, seed=123):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if len(A) == 0 or len(B) == 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    d = A.shape[1]
    dirs = rng.normal(size=(n_proj, d))
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)

    n = min(len(A), len(B))
    vals = []
    for v in dirs:
        a = np.sort(A @ v)[:n]
        b = np.sort(B @ v)[:n]
        vals.append(np.mean((a - b) ** 2))
    return float(np.mean(vals))


def sinkhorn_w2_cloud(A, B, blur=0.05, device="cpu", max_points=1000):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if len(A) == 0 or len(B) == 0:
        return float("nan")

    if len(A) > max_points:
        idx = np.linspace(0, len(A) - 1, max_points).astype(int)
        A = A[idx]
    if len(B) > max_points:
        idx = np.linspace(0, len(B) - 1, max_points).astype(int)
        B = B[idx]

    if not HAS_GEOMLOSS:
        return sliced_w2_cloud(A, B, n_proj=256, seed=123)

    XA = torch.tensor(A, dtype=torch.float32, device=device)
    XB = torch.tensor(B, dtype=torch.float32, device=device)
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=float(blur), backend="tensorized")
    return float(loss(XA, XB).detach().cpu())


# ---------------------------------------------------------------------
# Poincare section / unimodal metrics
# ---------------------------------------------------------------------
def crossing_happened(a, b, direction):
    if direction > 0:
        return (a < 0.0) and (b >= 0.0)
    elif direction < 0:
        return (a > 0.0) and (b <= 0.0)
    else:
        return (a == 0.0) or (b == 0.0) or (a * b < 0.0)


def extract_poincare_hits(traj, dt, section_dim=0, section_value=0.0, direction=+1):
    X = np.asarray(traj, dtype=np.float64)
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
    if len(hit_points) > 0:
        hit_points = np.asarray(hit_points, dtype=np.float64)
    else:
        hit_points = np.zeros((0, d), dtype=np.float64)

    tau = np.diff(hit_times) if len(hit_times) >= 2 else np.zeros((0,), dtype=np.float64)
    return hit_times, hit_points, tau


def wasserstein_1d_empirical(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    n = min(len(x), len(y))
    xs = np.sort(x)[:n]
    ys = np.sort(y)[:n]
    return float(np.mean(np.abs(xs - ys)))


def tau_dist_w1_on_section(gt, pred, dt, section_dim, section_value, direction=+1):
    hit_gt, _, tau_gt = extract_poincare_hits(
        gt, dt=dt, section_dim=section_dim, section_value=section_value, direction=direction
    )
    hit_pr, _, tau_pr = extract_poincare_hits(
        pred, dt=dt, section_dim=section_dim, section_value=section_value, direction=direction
    )
    return wasserstein_1d_empirical(tau_gt, tau_pr), int(len(hit_gt)), int(len(hit_pr))


def simple_find_peaks_prominence(z, prominence=1.0, distance=None):
    z = np.asarray(z, dtype=np.float64)
    peaks = []
    last = -10**12
    dist = 1 if distance is None else int(distance)
    for i in range(1, len(z) - 1):
        if i - last < dist:
            continue
        if z[i] > z[i - 1] and z[i] >= z[i + 1]:
            left_min = np.min(z[max(0, i - 20):i + 1])
            right_min = np.min(z[i:min(len(z), i + 21)])
            prom = z[i] - max(left_min, right_min)
            if prom >= prominence:
                peaks.append(i)
                last = i
    return np.asarray(peaks, dtype=int), {}


def extract_unimodal_return_info(
    trajectory,
    dt,
    z_idx=1,
    prominence=1.0,
    distance=None,
    refine_peak_time=True,
):
    X = np.asarray(trajectory, dtype=np.float64)
    z_series = X[:, z_idx]

    peak_kwargs = {"prominence": prominence}
    if distance is not None:
        peak_kwargs["distance"] = int(distance)

    if HAS_SCIPY_SIGNAL:
        peaks, _ = find_peaks(z_series, **peak_kwargs)
    else:
        peaks, _ = simple_find_peaks_prominence(z_series, prominence=prominence, distance=distance)

    z_peaks = z_series[peaks].astype(np.float64)
    peak_times = peaks.astype(np.float64) * float(dt)

    if refine_peak_time and len(peaks) > 0:
        for j, k in enumerate(peaks):
            if k <= 0 or k >= len(z_series) - 1:
                continue
            ym1 = float(z_series[k - 1])
            y0 = float(z_series[k])
            yp1 = float(z_series[k + 1])
            denom = ym1 - 2.0 * y0 + yp1
            if abs(denom) > 1e-14:
                delta = 0.5 * (ym1 - yp1) / denom
                if np.isfinite(delta) and (-1.0 <= delta <= 1.0):
                    peak_times[j] = (float(k) + float(delta)) * float(dt)

    tau = np.diff(peak_times) if len(peak_times) >= 2 else np.zeros((0,), dtype=np.float64)
    return {
        "peaks": peaks,
        "peak_times": peak_times,
        "z_peaks": z_peaks,
        "tau": tau,
        "n_peak_hits": int(len(peaks)),
        "n_peak_pairs": int(max(0, len(peaks) - 1)),
    }


def tau_dist_w1_unimodal(gt, pred, dt, z_idx=1, prominence=1.0, distance=None, refine_peak_time=True):
    gt_info = extract_unimodal_return_info(
        gt, dt=dt, z_idx=z_idx, prominence=prominence,
        distance=distance, refine_peak_time=refine_peak_time
    )
    pr_info = extract_unimodal_return_info(
        pred, dt=dt, z_idx=z_idx, prominence=prominence,
        distance=distance, refine_peak_time=refine_peak_time
    )
    w1 = wasserstein_1d_empirical(gt_info["tau"], pr_info["tau"])
    return w1, gt_info["n_peak_hits"], pr_info["n_peak_hits"]


# ---------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------
def finite_mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else float("nan")


def finite_std(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.std(x)) if len(x) else float("nan")


def finite_count(x):
    x = np.asarray(x, dtype=np.float64)
    return int(np.sum(np.isfinite(x)))


def fmt_float(x, nd=6):
    try:
        x = float(x)
    except Exception:
        return str(x)
    return "nan" if not np.isfinite(x) else f"{x:.{nd}g}"


def fmt_mean_std(mu, sd):
    if not np.isfinite(mu) or not np.isfinite(sd):
        return "nan"
    return f"{mu:.4g} ± {sd:.4g}"


# ---------------------------------------------------------------------
# CLI / evaluation
# ---------------------------------------------------------------------


def choose_starts(T, horizon, n_starts, mode="linspace", seed=42):
    valid_count = T - horizon - 1
    if valid_count <= 0:
        raise ValueError(f"test_data too short: T={T}, horizon={horizon}")
    if valid_count < n_starts:
        warnings.warn(f"Only {valid_count} valid starts available; using all of them.")
        return np.arange(valid_count, dtype=int)
    if mode == "random":
        rng = np.random.default_rng(seed)
        starts = rng.choice(valid_count, size=n_starts, replace=False)
        starts = np.sort(starts)
    else:
        starts = np.linspace(0, valid_count - 1, n_starts).astype(int)
        starts = np.unique(starts)
        if len(starts) < n_starts:
            extra = [i for i in range(valid_count) if i not in set(starts)]
            starts = np.sort(np.concatenate([starts, np.asarray(extra[:n_starts - len(starts)], dtype=int)]))
    return starts.astype(int)


def evaluate_one_prediction_rollout(gt, pred, dt, args):
    rec = {}
    for w in NMSE_WINDOWS:
        rec[f"NMSE_{w}"] = nmse_value(gt[:w + 1], pred[:w + 1])

    rec["Attractor_W2"] = sinkhorn_w2_cloud(
        gt, pred, blur=args.w2_blur, device=args.device, max_points=args.w2_max_points
    )
    rec["PSD_logL1"] = psd_logl1_distance(
        gt, pred, dt=dt, nfft=args.psd_nfft, channels=None
    )
    melr, melrw = melr_melrw_distance(
        gt,
        pred,
        eps=args.melr_eps,
        drop_zero_mode=args.melr_drop_zero_mode,
    )
    rec["MELR"] = melr
    rec["MELRw"] = melrw
    rec["TCM"] = tcm_distance(
        gt,
        pred,
        dt=dt,
        channels=None,
        max_lag=args.tcm_max_lag,
        stop_first_nonpositive=args.tcm_truncate_first_nonpositive,
    )
    return rec


def evaluate_full_poincare_rollout(gt_long, pred_long, dt, args, z_idx):
    rec = {}

    w1_x10, gt_hits_x10, pr_hits_x10 = tau_dist_w1_on_section(
        gt_long, pred_long,
        dt=dt,
        section_dim=0,
        section_value=args.x_section_value,
        direction=args.section_direction,
    )

    w1_z40, gt_hits_z40, pr_hits_z40 = tau_dist_w1_on_section(
        gt_long, pred_long,
        dt=dt,
        section_dim=z_idx,
        section_value=args.z_section_value,
        direction=args.section_direction,
    )

    rec["tauW1_x10"] = w1_x10
    rec["gt_hits_x10"] = gt_hits_x10
    rec["pred_hits_x10"] = pr_hits_x10

    rec["tauW1_z40"] = w1_z40
    rec["gt_hits_z40"] = gt_hits_z40
    rec["pred_hits_z40"] = pr_hits_z40

    distance = None if int(args.unimodal_distance) <= 0 else int(args.unimodal_distance)
    w1_umap, gt_peak_hits, pr_peak_hits = tau_dist_w1_unimodal(
        gt_long,
        pred_long,
        dt=dt,
        z_idx=z_idx,
        prominence=args.unimodal_prominence,
        distance=distance,
        refine_peak_time=(not args.no_refine_peak_time),
    )
    rec["tauW1_unimodal"] = w1_umap
    rec["gt_peak_hits"] = gt_peak_hits
    rec["pred_peak_hits"] = pr_peak_hits

    return rec


def aggregate_prediction_results(df_pred):
    rows = []
    for model in MODEL_ORDER:
        sub = df_pred[df_pred["model"] == model]
        if len(sub) == 0:
            continue

        row = {
            "model": model,
            "n_prediction_rollouts": int(len(sub)),
            "n_seed_folders_prediction": int(sub["seed_folder"].nunique()),
        }

        for w in NMSE_WINDOWS:
            vals = sub[f"NMSE_{w}"].to_numpy(dtype=np.float64)
            mu = finite_mean(vals)
            sd = finite_std(vals)
            row[f"NMSE_{w}_mean"] = mu
            row[f"NMSE_{w}_std"] = sd
            row[f"NMSE_{w}"] = fmt_mean_std(mu, sd)

        for metric in ["Attractor_W2", "PSD_logL1", "MELR", "MELRw", "TCM"]:
            vals = sub[metric].to_numpy(dtype=np.float64)
            mu = finite_mean(vals)
            sd = finite_std(vals)
            row[f"{metric}_mean"] = mu
            row[f"{metric}_std"] = sd
            row[f"{metric}"] = fmt_mean_std(mu, sd)
            row[f"{metric}_n_valid"] = finite_count(vals)

        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_poincare_results(df_poi):
    rows = []
    for model in MODEL_ORDER:
        sub = df_poi[df_poi["model"] == model]
        if len(sub) == 0:
            continue

        row = {
            "model": model,
            "n_poincare_seed_models": int(len(sub)),
            "n_seed_folders_poincare": int(sub["seed_folder"].nunique()),
        }

        for metric in ["tauW1_x10", "tauW1_z40", "tauW1_unimodal"]:
            vals = sub[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = finite_mean(vals)
            row[f"{metric}_std"] = finite_std(vals)
            row[f"{metric}_n_valid"] = finite_count(vals)

        for metric in [
            "gt_hits_x10", "pred_hits_x10",
            "gt_hits_z40", "pred_hits_z40",
            "gt_peak_hits", "pred_peak_hits",
        ]:
            vals = sub[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = finite_mean(vals)
            row[f"{metric}_std"] = finite_std(vals)

        rows.append(row)
    return pd.DataFrame(rows)


def merge_summaries(pred_summary, poi_summary):
    if len(pred_summary) == 0:
        return poi_summary
    if len(poi_summary) == 0:
        return pred_summary
    return pred_summary.merge(poi_summary, on="model", how="outer")




