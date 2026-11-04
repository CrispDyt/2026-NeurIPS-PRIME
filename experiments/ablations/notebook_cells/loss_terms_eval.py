# Extracted notebook code; CLI/path/device/display adaptations only.
from _notebook_bootstrap import prepare
globals().update(prepare(__file__))
# ============================================================
# Evaluate Lorenz63 loss-term ablation models
# Modified version:
#   - TCM
#   - Attractor_W2
#   - PSD_logL1
#   - DRTD on three recurrence sections:
#       Sigma1: u0 = 10
#       Sigma2: u2 = 40
#       Sigma3: local maxima of u2 using find_peaks + prominence + optional quadratic refinement
# ============================================================

import os
import io
import glob
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from geomloss import SamplesLoss
    HAS_GEOMLOSS = True
except Exception as e:
    HAS_GEOMLOSS = False
    print("[warn] geomloss not available; using sliced-W2 fallback.")
    print("[warn]", repr(e))

try:
    from scipy.signal import welch
    HAS_WELCH = True
except Exception as e:
    HAS_WELCH = False
    print("[warn] scipy.signal.welch not available; using FFT fallback.")
    print("[warn]", repr(e))

try:
    from scipy.signal import find_peaks
    HAS_FIND_PEAKS = True
except Exception as e:
    HAS_FIND_PEAKS = False
    print("[warn] scipy.signal.find_peaks not available; unimodal extraction will fail.")
    print("[warn]", repr(e))


# ============================================================
# 0. Config
# ============================================================
ROOT_DIR = PORT_CHECKPOINTS / "lorenz_ablation_loss_terms"

TRAIN_PATH = PORT_DATA / "train_data.npy"
TEST_PATH = PORT_DATA / "test_data.npy"

OUT_PRED_CSV = PORT_OUTPUT / "loss_term_ablation_prediction_metrics_TCM_seed1000.csv"
OUT_DRTD_CSV = PORT_OUTPUT / "loss_term_ablation_drtd_TCM_seed1000.csv"
OUT_SUM_CSV = PORT_OUTPUT / "loss_term_ablation_summary_TCM_seed1000.csv"

DEVICE = torch.device(PORT_DEVICE)
print("device:", DEVICE)

# Rollout/eval config
N_STARTS = 30
PRED_HORIZON = 500

# TCM config
TCM_MAX_LAG = 200
TCM_TRUNCATE_FIRST_NONPOSITIVE = False

# DRTD config
DRTD_HORIZON = 1000      # long rollout from test[0]
DT_OBS = 0.1             # Lorenz observation interval
SECTION_DIRECTION = +1   # +1 upward crossing for explicit level sections

# Unimodal-map recurrence event extraction.
# This matches your second script.
UNIMODAL_PROMINENCE = 1.0
UNIMODAL_DISTANCE = None
REFINE_PEAK_TIME = True

# W2 / PSD config
W2_BLUR = 0.05
W2_MAX_POINTS = 1500
PSD_NFFT = 2048

# Model folders
MODEL_SPECS = [
    {
        "name": "full",
        "label": "Full",
        "folder": ROOT_DIR / "full_seed1000",
        "pattern": "LossTerm_full_seed1000_SELECTED*.p",
    },
    {
        "name": "dim_only",
        "label": "DIM-only",
        "folder": ROOT_DIR / "dim_only_seed1000",
        "pattern": "LossTerm_dim_only_seed1000_SELECTED*.p",
    },
    {
        "name": "graph_only",
        "label": "Graph-only",
        "folder": ROOT_DIR / "graph_only_seed1000",
        "pattern": "LossTerm_graph_only_seed1000_SELECTED*.p",
    },
    {
        "name": "marginals_only",
        "label": "Marginals-only",
        "folder": ROOT_DIR / "marginals_only_seed1000",
        "pattern": "LossTerm_marginals_only_seed1000_SELECTED*.p",
    },
    {
        "name": "no_source_marginal",
        "label": "No source marginal",
        "folder": ROOT_DIR / "no_source_marginal_seed1000",
        "pattern": "LossTerm_no_source_marginal_seed1000_SELECTED*.p",
    },
    {
        "name": "no_target_marginal",
        "label": "No target marginal",
        "folder": ROOT_DIR / "no_target_marginal_seed1000",
        "pattern": "LossTerm_no_target_marginal_seed1000_SELECTED*.p",
    },
]


# ============================================================
# 1. Robust checkpoint loading
# ============================================================
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_pickle_cpu(path):
    with open(path, "rb") as f:
        return CPU_Unpickler(f).load()


def smart_load(path):
    last_err = None

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        last_err = e

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    try:
        return load_pickle_cpu(path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint: {path}\n"
            f"torch.load error: {last_err}\n"
            f"pickle/cpu-unpickler error: {e}"
        )


def latest_file(patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(str(p)))
    files = sorted(files, key=os.path.getmtime)
    return files[-1] if files else None


def find_checkpoint(folder, pattern):
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    p = latest_file([folder / pattern])
    if p is not None:
        return Path(p)

    p = latest_file([folder / "*_SELECTED_*.p"])
    if p is not None:
        return Path(p)

    p = latest_file([folder / "*_BEST_*.p"])
    if p is not None:
        return Path(p)

    p = latest_file([folder / "*_FINAL_*.p"])
    if p is not None:
        return Path(p)

    raise FileNotFoundError(f"No checkpoint found in {folder}")


def strip_module_prefix(sd):
    if isinstance(sd, dict) and len(sd) > 0 and all(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def extract_payload(obj, train_data_fallback=None):
    """
    Expected training payload:
      obj["model"]["transformer"] or obj["model"]["scaler"]
      obj["model"]["state_dict"]
      obj["config"]["dt_base"]
    """
    sd = None
    scaler = None
    dt_base = None
    config = {}

    if isinstance(obj, dict):
        config = obj.get("config", {})

        if "model" in obj and isinstance(obj["model"], dict):
            model_obj = obj["model"]

            if "state_dict" in model_obj:
                sd = model_obj["state_dict"]

            if "transformer" in model_obj:
                scaler = model_obj["transformer"]
            elif "scaler" in model_obj:
                scaler = model_obj["scaler"]

        if sd is None:
            for k in ["state_dict", "model_state_dict", "net_state_dict"]:
                if k in obj:
                    sd = obj[k]
                    break

        if scaler is None:
            for k in ["transformer", "scaler"]:
                if k in obj:
                    scaler = obj[k]
                    break

        for k in ["dt_base", "dt", "dt_obs"]:
            if k in obj:
                try:
                    dt_base = float(obj[k])
                    break
                except Exception:
                    pass

        if dt_base is None and isinstance(config, dict):
            for k in ["dt_base", "dt", "dt_obs"]:
                if k in config:
                    try:
                        dt_base = float(config[k])
                        break
                    except Exception:
                        pass

    if sd is None:
        raise RuntimeError("Could not find state_dict in checkpoint payload.")

    if scaler is None:
        if train_data_fallback is None:
            raise RuntimeError("No scaler/transformer found in checkpoint and no fallback given.")
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler().fit(train_data_fallback)
        print("[warn] no scaler in checkpoint; fitted fallback MaxAbsScaler on train data.")

    if dt_base is None:
        dt_base = DT_OBS
        print(f"[warn] no dt_base found; fallback dt_base={dt_base}")

    return strip_module_prefix(sd), scaler, float(dt_base), config


# ============================================================
# 2. Rebuild network
# ============================================================
def natural_key(s):
    import re
    parts = re.split(r"(\d+)", s)
    out = []
    for p in parts:
        out.append(int(p) if p.isdigit() else p)
    return out


def extract_linear_weight_keys(sd):
    return sorted([k for k in sd.keys() if k.endswith(".weight")], key=natural_key)


class GenericSequentialMap(nn.Sequential):
    def __init__(self, dims):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        super().__init__(*layers)


def build_net_from_state_dict(sd, device):
    sd = strip_module_prefix(sd)

    for pref in ["net", "model", "f", "vf", "map"]:
        pref_dot = pref + "."
        if len(sd) > 0 and all(k.startswith(pref_dot) for k in sd.keys()):
            sd = {k[len(pref_dot):]: v for k, v in sd.items()}
            break

    wkeys = extract_linear_weight_keys(sd)
    if len(wkeys) == 0:
        raise RuntimeError("No linear weight keys found in state_dict.")

    shapes = [tuple(sd[k].shape) for k in wkeys]
    dims = [shapes[0][1]] + [s[0] for s in shapes]

    input_dim = int(dims[0])
    output_dim = int(dims[-1])

    net = GenericSequentialMap(dims).to(device)
    missing, unexpected = net.load_state_dict(sd, strict=False)

    if len(missing) > 0 or len(unexpected) > 0:
        print("[warn] non-strict load")
        print("  missing   :", missing)
        print("  unexpected:", unexpected)

    net.eval()
    return net, input_dim, output_dim, dims


# ============================================================
# 3. Vector-field rollout
# ============================================================
class VectorFieldModel:
    def __init__(self, net, scaler, dt_base, dim, input_dim, device):
        self.net = net
        self.scaler = scaler
        self.dt_base = float(dt_base)
        self.dim = int(dim)
        self.input_dim = int(input_dim)
        self.device = device

    @torch.no_grad()
    def rollout(self, x0_raw, horizon):
        x0_raw = np.asarray(x0_raw, dtype=np.float32).reshape(1, -1)
        x_s = self.scaler.transform(x0_raw)

        x = torch.tensor(x_s, dtype=torch.float32, device=self.device)
        out_s = [x.detach().cpu().numpy()[0].copy()]

        for _ in range(int(horizon)):
            if self.input_dim != self.dim:
                raise ValueError(
                    f"Expected input_dim == dim, got input_dim={self.input_dim}, dim={self.dim}"
                )
            f = self.net(x)
            x = x + self.dt_base * f
            out_s.append(x.detach().cpu().numpy()[0].copy())

        out_s = np.asarray(out_s, dtype=np.float32)
        out_raw = self.scaler.inverse_transform(out_s)
        return np.asarray(out_raw, dtype=np.float64)


# ============================================================
# 4. Metrics
# ============================================================
def sliced_w2_cloud(A, B, n_proj=256, seed=123, max_points=1500):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    A = A[np.isfinite(A).all(axis=1)]
    B = B[np.isfinite(B).all(axis=1)]

    if len(A) == 0 or len(B) == 0:
        return np.nan

    rng = np.random.default_rng(seed)

    if len(A) > max_points:
        A = A[rng.choice(len(A), size=max_points, replace=False)]
    if len(B) > max_points:
        B = B[rng.choice(len(B), size=max_points, replace=False)]

    d = A.shape[1]
    dirs = rng.normal(size=(n_proj, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

    n_q = min(len(A), len(B), max_points)
    qs = np.linspace(0.001, 0.999, n_q)

    vals = []
    for v in dirs:
        aq = np.quantile(A @ v, qs)
        bq = np.quantile(B @ v, qs)
        vals.append(np.mean((aq - bq) ** 2))

    return float(np.mean(vals))


def sinkhorn_w2_cloud(A, B, blur=0.05, device="cpu", max_points=1500):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)

    A = A[np.isfinite(A).all(axis=1)]
    B = B[np.isfinite(B).all(axis=1)]

    if len(A) == 0 or len(B) == 0:
        return np.nan

    if len(A) > max_points:
        idx = np.linspace(0, len(A) - 1, max_points).astype(int)
        A = A[idx]

    if len(B) > max_points:
        idx = np.linspace(0, len(B) - 1, max_points).astype(int)
        B = B[idx]

    if not HAS_GEOMLOSS:
        return sliced_w2_cloud(A, B, max_points=max_points)

    try:
        XA = torch.tensor(A, dtype=torch.float32, device=device)
        XB = torch.tensor(B, dtype=torch.float32, device=device)
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=float(blur), backend="tensorized")
        return float(loss(XA, XB).detach().cpu())
    except Exception as e:
        print("[warn] Sinkhorn failed; using sliced-W2 fallback:", repr(e))
        return sliced_w2_cloud(A, B, max_points=max_points)


def psd_logl1_distance(ref, pred, dt=0.1, nfft=2048, channels=None, eps=1e-12):
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

        if len(x) < 8:
            continue

        if HAS_WELCH:
            _, Pxx = welch(
                x,
                fs=fs,
                nperseg=min(512, n),
                noverlap=min(256, n // 2),
            )
            _, Pyy = welch(
                y,
                fs=fs,
                nperseg=min(512, n),
                noverlap=min(256, n // 2),
            )
        else:
            Pxx = np.abs(np.fft.rfft(x)) ** 2 / max(1, len(x))
            Pyy = np.abs(np.fft.rfft(y)) ** 2 / max(1, len(y))

        m = min(len(Pxx), len(Pyy))
        dists.append(
            float(np.mean(np.abs(np.log10(Pxx[:m] + eps) - np.log10(Pyy[:m] + eps))))
        )

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


def crossing_happened(a, b, direction):
    if direction > 0:
        return (a < 0.0) and (b >= 0.0)
    if direction < 0:
        return (a > 0.0) and (b <= 0.0)
    return (a == 0.0) or (b == 0.0) or (a * b < 0.0)


def extract_level_hits(traj, dim, value, dt=0.1, direction=+1):
    """
    Linear-interpolated level-crossing hit times.
    """
    X = np.asarray(traj, dtype=np.float64)
    X = X[np.isfinite(X).all(axis=1)]

    if len(X) < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    hit_times = []

    for k in range(len(X) - 1):
        a = X[k, dim] - value
        b = X[k + 1, dim] - value

        if not crossing_happened(a, b, direction):
            continue

        denom = b - a
        alpha = 0.0 if abs(denom) < 1e-14 else -a / denom

        if 0.0 <= alpha <= 1.0:
            hit_times.append((k + alpha) * dt)

    hit_times = np.asarray(hit_times, dtype=np.float64)
    tau = np.diff(hit_times) if len(hit_times) >= 2 else np.asarray([], dtype=np.float64)
    return hit_times, tau


def extract_unimodal_return_info(
    trajectory,
    dt,
    z_idx=1,
    prominence=1.0,
    distance=None,
    refine_peak_time=True,
):
    """
    Same unimodal extraction logic as your second script.

    Extracts local maxima by scipy.signal.find_peaks with prominence filtering,
    and optionally refines peak event times by quadratic interpolation.
    """
    if not HAS_FIND_PEAKS:
        raise RuntimeError("scipy.signal.find_peaks is required for unimodal extraction.")

    X = np.asarray(trajectory, dtype=np.float64)
    X = X[np.isfinite(X).all(axis=1)]

    if len(X) < 3:
        return {
            "peaks": np.array([], dtype=np.int64),
            "peak_times": np.array([], dtype=np.float64),
            "z_peaks": np.array([], dtype=np.float64),
            "tau": np.array([], dtype=np.float64),
            "zn": np.array([], dtype=np.float64),
            "zn1": np.array([], dtype=np.float64),
            "n_peak_hits": 0,
            "n_peak_pairs": 0,
        }

    z_series = X[:, z_idx]

    peak_kwargs = {"prominence": prominence}
    if distance is not None:
        peak_kwargs["distance"] = int(distance)

    peaks, _ = find_peaks(z_series, **peak_kwargs)
    z_peaks = z_series[peaks].astype(np.float64)

    peak_times = peaks.astype(np.float64) * float(dt)

    if refine_peak_time and len(peaks) > 0:
        for j, k in enumerate(peaks):
            if k <= 0 or k >= len(z_series) - 1:
                continue

            ym1 = float(z_series[k - 1])
            y0  = float(z_series[k])
            yp1 = float(z_series[k + 1])

            denom = ym1 - 2.0 * y0 + yp1

            if abs(denom) > 1e-14:
                delta = 0.5 * (ym1 - yp1) / denom
                if np.isfinite(delta) and (-1.0 <= delta <= 1.0):
                    peak_times[j] = (float(k) + float(delta)) * float(dt)

    if len(z_peaks) >= 2:
        zn = z_peaks[:-1]
        zn1 = z_peaks[1:]
        tau = np.diff(peak_times)
    else:
        zn = np.array([], dtype=np.float64)
        zn1 = np.array([], dtype=np.float64)
        tau = np.array([], dtype=np.float64)

    return {
        "peaks": peaks,
        "peak_times": peak_times,
        "z_peaks": z_peaks,
        "tau": tau,
        "zn": zn,
        "zn1": zn1,
        "n_peak_hits": int(len(peaks)),
        "n_peak_pairs": int(len(zn)),
    }


def extract_local_max_hits(traj, dim=1, dt=0.1):
    """
    Compatibility wrapper used by compute_drtd_sections.
    Now uses find_peaks(..., prominence=UNIMODAL_PROMINENCE).
    """
    info = extract_unimodal_return_info(
        trajectory=traj,
        dt=dt,
        z_idx=dim,
        prominence=UNIMODAL_PROMINENCE,
        distance=UNIMODAL_DISTANCE,
        refine_peak_time=REFINE_PEAK_TIME,
    )
    return info["peak_times"], info["tau"]


def wasserstein_1d_empirical(x, y):
    """
    Empirical 1D W1 / tau_dist_W1:
        sort both arrays, truncate to common length, average absolute difference.
    """
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


def compute_drtd_sections(ref, pred, dt=0.1):
    sections = []

    # Sigma1: u0 = 10
    h_ref, tau_ref = extract_level_hits(
        ref,
        dim=0,
        value=10.0,
        dt=dt,
        direction=SECTION_DIRECTION,
    )
    h_pred, tau_pred = extract_level_hits(
        pred,
        dim=0,
        value=10.0,
        dt=dt,
        direction=SECTION_DIRECTION,
    )
    sections.append({
        "section": "Sigma1_u0_10",
        "gt_hits": len(h_ref),
        "pred_hits": len(h_pred),
        "DRTD": wasserstein_1d_empirical(tau_ref, tau_pred),
    })

    # Sigma2: u2 = 40
    # Here dim=1 because the 2D observation is (u0, u2).
    h_ref, tau_ref = extract_level_hits(
        ref,
        dim=1,
        value=40.0,
        dt=dt,
        direction=SECTION_DIRECTION,
    )
    h_pred, tau_pred = extract_level_hits(
        pred,
        dim=1,
        value=40.0,
        dt=dt,
        direction=SECTION_DIRECTION,
    )
    sections.append({
        "section": "Sigma2_u2_40",
        "gt_hits": len(h_ref),
        "pred_hits": len(h_pred),
        "DRTD": wasserstein_1d_empirical(tau_ref, tau_pred),
    })

    # Sigma3: local maxima of u2 / unimodal recurrence section.
    h_ref, tau_ref = extract_local_max_hits(
        ref,
        dim=1,
        dt=dt,
    )
    h_pred, tau_pred = extract_local_max_hits(
        pred,
        dim=1,
        dt=dt,
    )
    sections.append({
        "section": "Sigma3_u2_local_max_prominence1_refined",
        "gt_hits": len(h_ref),
        "pred_hits": len(h_pred),
        "DRTD": wasserstein_1d_empirical(tau_ref, tau_pred),
    })

    return sections


def finite_mean_std(vals):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return np.nan, np.nan, 0

    return float(np.mean(vals)), float(np.std(vals, ddof=0)), int(len(vals))


def fmt(x, nd=3):
    if not np.isfinite(x):
        return "nan"
    ax = abs(float(x))
    if ax != 0 and (ax < 1e-2 or ax >= 1e4):
        return f"{x:.{nd}e}"
    return f"{x:.{nd}f}"


def write_csv(path, rows):
    if len(rows) == 0:
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
# 5. Load data
# ============================================================
if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Missing {TRAIN_PATH}")

if not TEST_PATH.exists():
    raise FileNotFoundError(f"Missing {TEST_PATH}")

train_raw = np.asarray(np.load(TRAIN_PATH, allow_pickle=True), dtype=np.float64)
test_raw  = np.asarray(np.load(TEST_PATH, allow_pickle=True), dtype=np.float64)

if train_raw.ndim != 2 or test_raw.ndim != 2:
    raise ValueError(f"Expected 2D train/test arrays, got train={train_raw.shape}, test={test_raw.shape}")

print("[OK] train_raw:", train_raw.shape)
print("[OK] test_raw :", test_raw.shape)


# ============================================================
# 6. Load models
# ============================================================
models = []

for spec in MODEL_SPECS:
    ckpt_path = find_checkpoint(spec["folder"], spec["pattern"])
    obj = smart_load(ckpt_path)

    sd, scaler, dt_base, config = extract_payload(obj, train_data_fallback=train_raw)
    net, input_dim, output_dim, dims = build_net_from_state_dict(sd, device=DEVICE)

    model = VectorFieldModel(
        net=net,
        scaler=scaler,
        dt_base=dt_base,
        dim=output_dim,
        input_dim=input_dim,
        device=DEVICE,
    )

    models.append({
        "name": spec["name"],
        "label": spec["label"],
        "checkpoint": str(ckpt_path),
        "model": model,
        "config": config,
        "dt_base": dt_base,
        "dims": dims,
    })

    print(f"[OK] loaded {spec['label']:20s} | {ckpt_path}")
    print("     dims    :", dims)
    print("     dt_base :", dt_base)


# ============================================================
# 7. Prediction metrics over N_STARTS
# ============================================================
max_start = len(test_raw) - PRED_HORIZON - 1

if max_start <= 0:
    raise ValueError(f"test_data too short for PRED_HORIZON={PRED_HORIZON}")

starts = np.linspace(0, max_start, min(N_STARTS, max_start + 1), dtype=int)
starts = np.unique(starts)

print("\nPrediction starts:", starts)
print("N starts:", len(starts))

prediction_rows = []

for item in models:
    name = item["name"]
    label = item["label"]
    model = item["model"]

    print(f"\nEvaluating prediction metrics: {label}")

    for start in starts:
        ref = test_raw[start:start + PRED_HORIZON + 1]
        x0 = ref[0]

        pred = model.rollout(x0, horizon=PRED_HORIZON)

        row = {
            "model": name,
            "label": label,
            "start_idx": int(start),
            "checkpoint": item["checkpoint"],
            "TCM": tcm_distance(
                ref,
                pred,
                dt=DT_OBS,
                max_lag=TCM_MAX_LAG,
                channels=None,
                stop_first_nonpositive=TCM_TRUNCATE_FIRST_NONPOSITIVE,
            ),
            "Attractor_W2": sinkhorn_w2_cloud(
                ref,
                pred,
                blur=W2_BLUR,
                device=DEVICE,
                max_points=W2_MAX_POINTS,
            ),
            "PSD_logL1": psd_logl1_distance(
                ref,
                pred,
                dt=DT_OBS,
                nfft=PSD_NFFT,
                channels=None,
            ),
        }

        prediction_rows.append(row)


# ============================================================
# 8. DRTD from test[0]
# ============================================================
drtd_rows = []

drtd_horizon_eff = min(DRTD_HORIZON, len(test_raw) - 1)
ref_long = test_raw[:drtd_horizon_eff + 1]
x0_long = ref_long[0]

print("\nDRTD horizon:", drtd_horizon_eff)
print("Unimodal extraction:")
print("  prominence :", UNIMODAL_PROMINENCE)
print("  distance   :", UNIMODAL_DISTANCE)
print("  refine time:", REFINE_PEAK_TIME)

for item in models:
    name = item["name"]
    label = item["label"]
    model = item["model"]

    pred_long = model.rollout(x0_long, horizon=drtd_horizon_eff)
    sections = compute_drtd_sections(ref_long, pred_long, dt=DT_OBS)

    for sec in sections:
        row = {
            "model": name,
            "label": label,
            "checkpoint": item["checkpoint"],
            **sec,
        }
        drtd_rows.append(row)


# ============================================================
# 9. Summary
# ============================================================
summary_rows = []

for item in models:
    name = item["name"]
    label = item["label"]

    pred_group = [r for r in prediction_rows if r["model"] == name]
    drtd_group = [r for r in drtd_rows if r["model"] == name]

    tcm_mu, tcm_sd, tcm_n = finite_mean_std([r["TCM"] for r in pred_group])
    w2_mu, w2_sd, w2_n = finite_mean_std([r["Attractor_W2"] for r in pred_group])
    psd_mu, psd_sd, psd_n = finite_mean_std([r["PSD_logL1"] for r in pred_group])

    drtd_vals = [r["DRTD"] for r in drtd_group]
    drtd_mu, drtd_sd, drtd_n = finite_mean_std(drtd_vals)

    summary_rows.append({
        "model": name,
        "label": label,
        "TCM_mean": tcm_mu,
        "TCM_std": tcm_sd,
        "TCM_n": tcm_n,
        "Attractor_W2_mean": w2_mu,
        "Attractor_W2_std": w2_sd,
        "Attractor_W2_n": w2_n,
        "PSD_logL1_mean": psd_mu,
        "PSD_logL1_std": psd_sd,
        "PSD_logL1_n": psd_n,
        "DRTD_mean_over_sections": drtd_mu,
        "DRTD_std_over_sections": drtd_sd,
        "DRTD_n_valid_sections": drtd_n,
    })


# ============================================================
# 10. Save CSV files
# ============================================================
ROOT_DIR.mkdir(parents=True, exist_ok=True)

write_csv(OUT_PRED_CSV, prediction_rows)
write_csv(OUT_DRTD_CSV, drtd_rows)
write_csv(OUT_SUM_CSV, summary_rows)

print("\nSaved:")
print(" prediction metrics:", OUT_PRED_CSV)
print(" DRTD metrics      :", OUT_DRTD_CSV)
print(" summary           :", OUT_SUM_CSV)


# ============================================================
# 11. Print compact table
# ============================================================
print("\n==================== SUMMARY ====================")
print(
    f"{'Model':24s} | {'TCM':>18s} | {'W2':>18s} | {'PSD':>18s} | {'DRTD':>18s}"
)
print("-" * 110)

for r in summary_rows:
    tcm  = f"{fmt(r['TCM_mean'], 3)} ± {fmt(r['TCM_std'], 3)}"
    w2   = f"{fmt(r['Attractor_W2_mean'], 3)} ± {fmt(r['Attractor_W2_std'], 3)}"
    psd  = f"{fmt(r['PSD_logL1_mean'], 3)} ± {fmt(r['PSD_logL1_std'], 3)}"
    drtd = f"{fmt(r['DRTD_mean_over_sections'], 3)} ± {fmt(r['DRTD_std_over_sections'], 3)}"

    print(f"{r['label']:24s} | {tcm:>18s} | {w2:>18s} | {psd:>18s} | {drtd:>18s}")

print("=================================================")


# ============================================================
# 12. Optional: expose summary variables in notebook
# ============================================================
loss_term_prediction_metrics = prediction_rows
loss_term_drtd_metrics = drtd_rows
loss_term_summary = summary_rows
