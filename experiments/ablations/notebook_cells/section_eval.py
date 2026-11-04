# Extracted notebook code; CLI/path/device/display adaptations only.
from _notebook_bootstrap import prepare
globals().update(prepare(__file__))
import os
import io
import pickle
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from scipy.signal import welch
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MaxAbsScaler
from geomloss import SamplesLoss

# ============================================================
# Section Ablation Evaluation Config
# ============================================================

# 你第二个 ablation 的保存根目录
ABLATION_ROOT = str(PORT_CHECKPOINTS / "lorenz_ablation_section")

# 改成你训练 section ablation 时用的 seed
SEED = PORT_ARGS.seed if PORT_ARGS.seed is not None else 1000

# section ablation 的四个模型目录
ABLATION_DIRS = {
    "DIM-only": f"{ABLATION_ROOT}/dim_only_seed{SEED}",
    "PRIME-implicit": f"{ABLATION_ROOT}/implicit_seed{SEED}",
    "PRIME-explicit-Z-band": f"{ABLATION_ROOT}/exact_z40_band_seed{SEED}",
    "PRIME-random-section": f"{ABLATION_ROOT}/random_hits_seed{SEED}",
}

# checkpoint 优先级：SELECTED > BEST > FINAL
def latest_file(patterns):
    files = []
    for p in patterns:
        files.extend(glob(p))
    files = sorted(files, key=os.path.getmtime)
    return files[-1] if len(files) > 0 else None

def find_ablation_ckpt(folder):
    ckpt = latest_file([os.path.join(folder, "*_SELECTED_*.p")])
    if ckpt is not None:
        return ckpt

    ckpt = latest_file([os.path.join(folder, "*_BEST_*.p")])
    if ckpt is not None:
        return ckpt

    ckpt = latest_file([os.path.join(folder, "*_FINAL_*.p")])
    if ckpt is not None:
        return ckpt

    raise FileNotFoundError(f"No checkpoint found in {folder}")

ABLATION_CKPTS = {
    name: find_ablation_ckpt(folder)
    for name, folder in ABLATION_DIRS.items()
}

print("Loaded section-ablation checkpoints:")
for name, path in ABLATION_CKPTS.items():
    print(f"  {name:22s}: {path}")

# ============================================================
# Evaluation Settings
# ============================================================

MSE_WINDOWS = [10, 20, 50, 100, 200, 300, 400, 500]

PSD_NFFT = 2048
PSD_CHANNELS = None
W2_BLUR = 0.05

# short-window NMSE 使用多少个起点
N_SHORT_STARTS = 30

# long-rollout metrics 使用多少个起点
# 如果 test_data 很短，会自动减少
N_LONG_STARTS = 20

# long-rollout 长度。你的 test_data 如果是 1000，可以用 800，避免只能从一个起点开始
LONG_ROLLOUT_K = 800

PLOT_ON = PORT_ARGS.plots
PLOT_N = 1500

# 你的 notebook 里应该已经有 train_data, test_data, dt_obs
assert "train_data" in globals(), "Notebook must define train_data"
assert "test_data" in globals(), "Notebook must define test_data"
assert "dt_obs" in globals(), "Notebook must define dt_obs"

device = PORT_DEVICE
print("device:", device)

train_data = np.asarray(train_data, dtype=np.float64)
test_data = np.asarray(test_data, dtype=np.float64)

T_eval = int(len(test_data))
dt = float(dt_obs)
ref = np.asarray(test_data[:T_eval], dtype=np.float64)
m = ref.shape[1]

LONG_ROLLOUT_K = min(int(LONG_ROLLOUT_K), T_eval - 1)

print(f"eval: T_eval={T_eval}, dim={m}, dt_obs={dt}, LONG_ROLLOUT_K={LONG_ROLLOUT_K}")

# ============================================================
# Poincare Sections for Evaluation
# ============================================================
# 注意：这里的 channel 是 partial observation 的列号
# 如果你的 Lorenz partial observation 是 [X, Z]:
#   column 0 = X / u0
#   column 1 = Z / u2

POINCARE_SECTIONS = [
    ("u0=10",  "level", 0, 10.0, +1),
    ("u2=40",  "level", 1, 40.0, +1),
    ("u2-max", "max",   1, None, None),
]

# 如果你论文表格使用的是 X=-10, Z=40, X=0，就改成下面这组：
# POINCARE_SECTIONS = [
#     ("u0=-10", "level", 0, -10.0, +1),
#     ("u2=40",  "level", 1, 40.0, +1),
#     ("u0=0",   "level", 0, 0.0, +1),
# ]

# ============================================================
# CPU-safe / torch-safe loader
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
# Metrics
# ============================================================

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

        _, Pxx = welch(x, fs=fs, nperseg=min(512, n), noverlap=min(256, n // 2))
        _, Pyy = welch(y, fs=fs, nperseg=min(512, n), noverlap=min(256, n // 2))

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

def sinkhorn_w2_cloud(A, B, blur=0.05, device="cpu"):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)

    if len(A) == 0 or len(B) == 0:
        return float("nan")

    XA = torch.tensor(A, dtype=torch.float32, device=device)
    XB = torch.tensor(B, dtype=torch.float32, device=device)

    loss = SamplesLoss(loss="sinkhorn", p=2, blur=float(blur), backend="tensorized")
    return float(loss(XA, XB).detach().cpu())

def crossing_times_level(X, ch, level, dt, direction=+1, min_gap_steps=1):
    """
    Linear-interpolated crossing times for X[:, ch] = level.
    Returns physical hit times.
    """
    X = np.asarray(X, dtype=np.float64)

    if ch >= X.shape[1]:
        raise ValueError(
            f"Section channel {ch} out of bounds for data with {X.shape[1]} channels."
        )

    x = X[:, ch]
    a = x[:-1] - level
    b = x[1:] - level

    if direction > 0:
        mask = (a < 0) & (b >= 0)
    elif direction < 0:
        mask = (a > 0) & (b <= 0)
    else:
        mask = (a == 0) | (a * b < 0)

    idx = np.where(mask)[0]

    hits = []
    last_i = -10**12

    for i in idx:
        if i - last_i < min_gap_steps:
            continue

        denom = x[i + 1] - x[i]
        if abs(denom) < 1e-12:
            alpha = 0.0
        else:
            alpha = (level - x[i]) / denom
            alpha = float(np.clip(alpha, 0.0, 1.0))

        hits.append((i + alpha) * dt)
        last_i = i

    return np.asarray(hits, dtype=np.float64)

def peak_times(X, ch, dt, min_gap_steps=1):
    """
    Local maxima hit times for X[:, ch].
    Returns physical hit times.
    """
    X = np.asarray(X, dtype=np.float64)

    if ch >= X.shape[1]:
        raise ValueError(
            f"Section channel {ch} out of bounds for data with {X.shape[1]} channels."
        )

    x = X[:, ch]
    idx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1

    hits = []
    last_i = -10**12

    for i in idx:
        if i - last_i < min_gap_steps:
            continue

        hits.append(i * dt)
        last_i = i

    return np.asarray(hits, dtype=np.float64)

def get_section_hits(X, section, dt):
    name, typ, ch, level, direction = section

    if typ == "level":
        return crossing_times_level(
            X, ch=ch, level=float(level), dt=dt,
            direction=int(direction), min_gap_steps=1
        )

    if typ == "max":
        return peak_times(
            X, ch=ch, dt=dt,
            min_gap_steps=1
        )

    raise ValueError(f"Unknown section type: {typ}")

def return_time_distribution(hit_times):
    hit_times = np.asarray(hit_times, dtype=np.float64)
    if len(hit_times) < 2:
        return np.asarray([], dtype=np.float64)

    return np.diff(hit_times)

def drtd_w1(ref_hits, pred_hits):
    tau_ref = return_time_distribution(ref_hits)
    tau_pred = return_time_distribution(pred_hits)

    if len(tau_ref) == 0 or len(tau_pred) == 0:
        return float("nan")

    return float(wasserstein_distance(tau_ref, tau_pred))

def hit_rate_error(ref_hits, pred_hits, T_phys):
    if T_phys <= 0:
        return float("nan")

    return float(abs(len(pred_hits) / T_phys - len(ref_hits) / T_phys))

def recurrence_metrics(ref, pred, dt, sections=POINCARE_SECTIONS):
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    L = min(len(ref), len(pred))
    ref = ref[:L]
    pred = pred[:L]

    T_phys = (L - 1) * dt

    out = {}
    drtd_vals = []
    hre_vals = []

    for section in sections:
        sec_name = section[0]

        h_ref = get_section_hits(ref, section, dt)
        h_pred = get_section_hits(pred, section, dt)

        d = drtd_w1(h_ref, h_pred)
        h = hit_rate_error(h_ref, h_pred, T_phys)

        out[f"{sec_name}_GT_hits"] = int(len(h_ref))
        out[f"{sec_name}_pred_hits"] = int(len(h_pred))
        out[f"{sec_name}_DRTD"] = d
        out[f"{sec_name}_HRE"] = h

        if np.isfinite(d):
            drtd_vals.append(d)
        if np.isfinite(h):
            hre_vals.append(h)

    out["Mean_DRTD"] = float(np.mean(drtd_vals)) if len(drtd_vals) else float("nan")
    out["Mean_HRE"] = float(np.mean(hre_vals)) if len(hre_vals) else float("nan")

    return out

# ============================================================
# Network / Checkpoint Utils
# ============================================================

def infer_mlp_dims_from_state_dict(sd: dict):
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

        if scaler is None and "transformer" in obj:
            scaler = obj["transformer"]

        if scaler is None and "scaler" in obj:
            scaler = obj["scaler"]
    else:
        sd = obj

    if sd is None:
        raise RuntimeError(f"Could not find usable state_dict in checkpoint: {path}")

    if scaler is None:
        scaler = MaxAbsScaler().fit(np.asarray(train_data, dtype=np.float64))

    return obj, sd, scaler

@torch.no_grad()
def rollout_from_x0(net, scaler, x0_raw, K, dt_obs, device):
    x0_s = scaler.transform(np.asarray(x0_raw).reshape(1, -1)).astype(np.float32)
    x = torch.tensor(x0_s, dtype=torch.float32, device=device)

    pred_s = np.zeros((K + 1, x.shape[1]), dtype=np.float32)

    for k in range(K + 1):
        pred_s[k] = x.detach().cpu().numpy()[0]
        x = x + float(dt_obs) * net(x)

    pred = scaler.inverse_transform(pred_s).astype(np.float64)

    return pred

# ============================================================
# Evaluation Core
# ============================================================

def choose_indices(T, K, n_starts, seed=42):
    valid_len = T - K - 1

    if valid_len <= 0:
        return np.array([0], dtype=np.int64)

    rng = np.random.default_rng(seed)

    if valid_len < n_starts:
        return np.arange(valid_len, dtype=np.int64)

    return rng.choice(valid_len, size=n_starts, replace=False).astype(np.int64)

def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan"), float("nan")

    return float(np.mean(values)), float(np.std(values))

def fmt_mean_std(mean, std, ndigits=4):
    if not np.isfinite(mean):
        return "nan"
    if not np.isfinite(std):
        return f"{mean:.{ndigits}g}"
    return f"{mean:.{ndigits}g} ± {std:.{ndigits}g}"

def evaluate_model(
    name,
    net,
    scaler,
    test_data_full,
    dt,
    short_windows=MSE_WINDOWS,
    n_short_starts=N_SHORT_STARTS,
    long_K=LONG_ROLLOUT_K,
    n_long_starts=N_LONG_STARTS,
):
    row = {"model": name}

    test_data_full = np.asarray(test_data_full, dtype=np.float64)

    # -----------------------------
    # Short-window MSE / NMSE
    # -----------------------------
    max_k = max(short_windows)
    short_indices = choose_indices(
        T=len(test_data_full),
        K=max_k,
        n_starts=n_short_starts,
        seed=42,
    )

    mse_vals = {k: [] for k in short_windows}
    nmse_vals = {k: [] for k in short_windows}

    for idx in short_indices:
        x0 = test_data_full[idx]
        gt = test_data_full[idx: idx + max_k + 1]

        pred = rollout_from_x0(net, scaler, x0, max_k, dt, device)

        for k in short_windows:
            err = gt[:k + 1] - pred[:k + 1]
            mse = np.mean(err ** 2)
            nmse = mse / (np.mean(gt[:k + 1] ** 2) + 1e-12)

            mse_vals[k].append(mse)
            nmse_vals[k].append(nmse)

    for k in short_windows:
        mse_mean, mse_std = summarize(mse_vals[k])
        nmse_mean, nmse_std = summarize(nmse_vals[k])

        row[f"MSE_{k}_mean"] = mse_mean
        row[f"MSE_{k}_std"] = mse_std
        row[f"MSE_{k}"] = fmt_mean_std(mse_mean, mse_std)

        row[f"NMSE_{k}_mean"] = nmse_mean
        row[f"NMSE_{k}_std"] = nmse_std
        row[f"NMSE_{k}"] = fmt_mean_std(nmse_mean, nmse_std)

    # -----------------------------
    # Long-window W2 / PSD / Recurrence
    # -----------------------------
    long_indices = choose_indices(
        T=len(test_data_full),
        K=long_K,
        n_starts=n_long_starts,
        seed=123,
    )

    w2_vals = []
    psd_vals = []
    rec_dicts = []

    first_pred = None
    first_ref = None

    for idx in long_indices:
        x0 = test_data_full[idx]
        gt = test_data_full[idx: idx + long_K + 1]

        pred = rollout_from_x0(net, scaler, x0, long_K, dt, device)

        if first_pred is None:
            first_pred = pred
            first_ref = gt

        w2_vals.append(
            sinkhorn_w2_cloud(gt, pred, blur=W2_BLUR, device=device)
        )

        psd_vals.append(
            psd_logl1_distance(gt, pred, dt, nfft=PSD_NFFT, channels=PSD_CHANNELS)
        )

        rec_dicts.append(
            recurrence_metrics(gt, pred, dt, sections=POINCARE_SECTIONS)
        )

    w2_mean, w2_std = summarize(w2_vals)
    psd_mean, psd_std = summarize(psd_vals)

    row["Attractor_W2_mean"] = w2_mean
    row["Attractor_W2_std"] = w2_std
    row["Attractor_W2"] = fmt_mean_std(w2_mean, w2_std)

    row["PSD_logL1_mean"] = psd_mean
    row["PSD_logL1_std"] = psd_std
    row["PSD_logL1"] = fmt_mean_std(psd_mean, psd_std)

    # Aggregate recurrence dicts
    rec_keys = sorted(rec_dicts[0].keys()) if len(rec_dicts) else []

    for key in rec_keys:
        vals = [d[key] for d in rec_dicts if key in d]
        mean, std = summarize(vals)

        row[f"{key}_mean"] = mean
        row[f"{key}_std"] = std

        if key.endswith("hits"):
            row[key] = fmt_mean_std(mean, std, ndigits=3)
        else:
            row[key] = fmt_mean_std(mean, std, ndigits=4)

    return row, first_pred, first_ref

# ============================================================
# Load + Evaluate Models
# ============================================================

rows = []
models_pred = {}
models_ref = {}

for name, ckpt_path in ABLATION_CKPTS.items():
    print("\n" + "=" * 80)
    print(f"Evaluating {name}")
    print(f"checkpoint: {ckpt_path}")
    print("=" * 80)

    obj, sd, scaler = load_model_ckpt(ckpt_path)

    d_in, width, d_out = infer_mlp_dims_from_state_dict(sd)

    net = build_net(d_in, width=width).to(device)
    net.load_state_dict(sd)
    net.eval()

    row, pred, ref_long = evaluate_model(
        name=name,
        net=net,
        scaler=scaler,
        test_data_full=test_data,
        dt=dt,
        short_windows=MSE_WINDOWS,
        n_short_starts=N_SHORT_STARTS,
        long_K=LONG_ROLLOUT_K,
        n_long_starts=N_LONG_STARTS,
    )

    # checkpoint metadata
    if isinstance(obj, dict):
        cfg = obj.get("config", {})
        stats = obj.get("stats", {})

        row["ckpt_path"] = ckpt_path
        row["selected_source"] = cfg.get("selected_source", "unknown")
        row["section_mode"] = cfg.get("section_mode", cfg.get("ablation_section_mode", "unknown"))
        row["train_N_hits"] = len(stats.get("hits", [])) if isinstance(stats, dict) else np.nan

    models_pred[name] = pred
    models_ref[name] = ref_long
    rows.append(row)

# ============================================================
# Print Tables
# ============================================================

def fmt(x):
    if isinstance(x, float):
        return "nan" if not np.isfinite(x) else f"{x:.6g}"
    return str(x)

COL_WIDTH = 22

print("\n==================== SHORT-WINDOW NMSE ====================")
nmse_cols = ["model"] + [f"NMSE_{k}" for k in MSE_WINDOWS]
header = " | ".join([f"{c:>{COL_WIDTH}s}" for c in nmse_cols])
print(header)
print("-" * len(header))

for r in rows:
    print(" | ".join([f"{fmt(r[c]):>{COL_WIDTH}s}" for c in nmse_cols]))

print("\n==================== SECTION ABLATION SUMMARY ====================")
summary_cols = [
    "model",
    "train_N_hits",
    "NMSE_100",
    "NMSE_400",
    "Attractor_W2",
    "PSD_logL1",
    "Mean_DRTD",
    "Mean_HRE",
]
header = " | ".join([f"{c:>22s}" for c in summary_cols])
print(header)
print("-" * len(header))

for r in rows:
    print(" | ".join([f"{fmt(r.get(c, 'nan')):>22s}" for c in summary_cols]))

print("\n==================== PER-SECTION RECURRENCE METRICS ====================")
rec_cols = ["model"]
for sec in POINCARE_SECTIONS:
    name = sec[0]
    rec_cols += [
        f"{name}_GT_hits",
        f"{name}_pred_hits",
        f"{name}_DRTD",
        f"{name}_HRE",
    ]

header = " | ".join([f"{c:>22s}" for c in rec_cols])
print(header)
print("-" * len(header))

for r in rows:
    print(" | ".join([f"{fmt(r.get(c, 'nan')):>22s}" for c in rec_cols]))

# ============================================================
# Save CSV
# ============================================================

try:
    import pandas as pd

    df = pd.DataFrame(rows)
    out_csv = os.path.join(
        str(PORT_OUTPUT),
        f"lorenz_section_ablation_eval_seed{SEED}.csv"
    )
    df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV: {out_csv}")

except Exception as e:
    print(f"[WARN] Could not save CSV: {e}")

# ============================================================
# Plot first long rollout from each model
# ============================================================

if PLOT_ON:
    Nplot = min(PLOT_N, LONG_ROLLOUT_K + 1)

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), dpi=160, sharex=True)

    # Use first model's reference as common reference if available
    first_name = list(models_pred.keys())[0]
    ref_plot = models_ref[first_name]

    tt = np.arange(Nplot) * dt

    ax[0].plot(tt, ref_plot[:Nplot, 0], lw=1.4, label="ref", color="black")

    for name, pred in models_pred.items():
        ax[0].plot(tt, pred[:Nplot, 0], lw=1.0, label=name)

    ax[0].set_title("Section Ablation: time series overlay (ch=0)")
    ax[0].legend()

    for name, pred in models_pred.items():
        L = min(Nplot, len(pred), len(ref_plot))
        err = np.linalg.norm(pred[:L] - ref_plot[:L], axis=1)
        ax[1].plot(tt[:L], err, lw=1.0, label=name)

    ax[1].set_yscale("log")
    ax[1].set_title("Section Ablation: trajectory error norm")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

print("\nDone.")
