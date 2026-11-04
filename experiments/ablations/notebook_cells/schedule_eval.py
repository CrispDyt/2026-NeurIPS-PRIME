# Extracted notebook code; CLI/path/device/display adaptations only.
from _notebook_bootstrap import prepare
globals().update(prepare(__file__))
import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import welch
from sklearn.preprocessing import MaxAbsScaler
from geomloss import SamplesLoss

# ============================================================
# checkpoint paths
# ============================================================
from glob import glob
from scipy.stats import wasserstein_distance

ABLATION_ROOT = str(PORT_CHECKPOINTS / "lorenz_ablation_K")
SEED = PORT_ARGS.seed if PORT_ARGS.seed is not None else 1000

ABLATION_DIRS = {
    "DIM-only": f"{ABLATION_ROOT}/dim_only_seed{SEED}",
    "PRIME-fixed-K": f"{ABLATION_ROOT}/fixed_seed{SEED}",
    "PRIME-shuffled-K": f"{ABLATION_ROOT}/shuffled_seed{SEED}",
    "PRIME-correct-K": f"{ABLATION_ROOT}/correct_seed{SEED}",
}

def latest_file(patterns):
    files = []
    for p in patterns:
        files.extend(glob(p))
    files = sorted(files, key=os.path.getmtime)
    if len(files) == 0:
        return None
    return files[-1]

def find_ablation_ckpt(folder):
    """
    Priority:
      SELECTED > BEST > FINAL
    """
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

print("Loaded ablation checkpoints:")
for name, path in ABLATION_CKPTS.items():
    print(f"  {name:18s}: {path}")

# keep the same windows as your current script
MSE_WINDOWS = [10, 20, 50, 100, 200, 300, 400, 500]

PSD_NFFT = 2048
PSD_CHANNELS = None
W2_BLUR = 0.05

PLOT_ON = PORT_ARGS.plots
PLOT_N = 1500
dt_obs = 0.1

assert "train_data" in globals()
assert "test_data" in globals()
assert "dt_obs" in globals()

device = PORT_DEVICE
print("device:", device)

T_eval = int(len(test_data))
dt = float(dt_obs)
ref = np.asarray(test_data[:T_eval], dtype=np.float64)
m = ref.shape[1]
print(f"eval: T_eval={T_eval}, dim={m}, dt_obs={dt}")

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
    """
    .pt / torch.save(...) -> torch.load
    .p  / pickle.dump(...) -> CPU-safe pickle fallback
    """
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
# basic metrics for Long-term Structure
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

POINCARE_SECTIONS = [
    ("u0=10",  "level", 0, 10.0, +1),
    ("u2=40",  "level", 1, 40.0, +1),
    ("u2-max", "max",   1, None, None),
]
def crossing_times_level(X, ch, level, dt, direction=+1, min_gap_steps=1):
    """
    Linear-interpolated crossing times for X[:, ch] = level.
    Returns physical times.
    """
    X = np.asarray(X, dtype=np.float64)
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
    Returns physical times.
    """
    X = np.asarray(X, dtype=np.float64)
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
    elif typ == "max":
        return peak_times(
            X, ch=ch, dt=dt, min_gap_steps=1
        )
    else:
        raise ValueError(f"Unknown section type: {typ}")

def return_time_distribution(hit_times):
    hit_times = np.asarray(hit_times, dtype=np.float64)
    if len(hit_times) < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(hit_times)

def drtd_w1(ref_hits, pred_hits):
    """
    Wasserstein-1 distance between empirical return-time distributions.
    """
    tau_ref = return_time_distribution(ref_hits)
    tau_pred = return_time_distribution(pred_hits)

    if len(tau_ref) == 0 or len(tau_pred) == 0:
        return float("nan")

    return float(wasserstein_distance(tau_ref, tau_pred))

def hit_rate_error(ref_hits, pred_hits, T_phys):
    """
    Absolute difference of hit rates per unit physical time.
    """
    if T_phys <= 0:
        return float("nan")
    return float(abs(len(pred_hits) / T_phys - len(ref_hits) / T_phys))

def recurrence_metrics(ref, pred, dt, sections=POINCARE_SECTIONS):
    """
    Computes per-section DRTD and HRE, plus mean values.
    """
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
# network / checkpoint utils
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
    d_in  = int(W0.shape[1])
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
# Evaluation Core: 50 Starts for MSE/NMSE (Mean ± STD), 1 Start for W2/PSD
# ============================================================
def evaluate_model(name, net, scaler, test_data_full, ref_full, dt, KMAX, n_starts=0):
    row = {"model": name}
    test_data_full = np.asarray(test_data_full, dtype=np.float64)
    
    # -------------------------------------------------------------
    # 1. Short/Mid-term Metrics (MSE, NMSE) via 50 Different Starts
    # -------------------------------------------------------------
    max_k = max(MSE_WINDOWS)
    valid_len = len(test_data_full) - max_k - 1
    
    if valid_len < n_starts:
        indices = np.arange(max(0, valid_len))
    else:
        np.random.seed(42) # fixed seed for reproducible comparison
        indices = np.random.choice(valid_len, n_starts, replace=False)

    results_mse = {k: [] for k in MSE_WINDOWS}
    results_nmse = {k: [] for k in MSE_WINDOWS}

    for idx in indices:
        x0 = test_data_full[idx]
        gt = test_data_full[idx : idx + max_k + 1]
        
        # Rollout specifically for the required window length
        pred = rollout_from_x0(net, scaler, x0, max_k, dt, device)

        for k in MSE_WINDOWS:
            err = gt[:k+1] - pred[:k+1]
            mse = np.mean(err**2)
            nmse = mse / (np.mean(gt[:k+1]**2) + 1e-12)
            
            results_mse[k].append(mse)
            results_nmse[k].append(nmse)

    # 格式化为 Mean ± STD 的字符串
    for k in MSE_WINDOWS:
        if len(results_mse[k]) > 0:
            mean_mse = np.mean(results_mse[k])
            std_mse = np.std(results_mse[k])
            row[f"MSE_{k}"] = f"{mean_mse:.4g} ± {std_mse:.4g}"
            
            mean_nmse = np.mean(results_nmse[k])
            std_nmse = np.std(results_nmse[k])
            row[f"NMSE_{k}"] = f"{mean_nmse:.4g} ± {std_nmse:.4g}"
        else:
            row[f"MSE_{k}"] = "nan"
            row[f"NMSE_{k}"] = "nan"

    # -------------------------------------------------------------
    # 2. Long-term Metrics (W2, PSD) via Single Long Rollout
    # -------------------------------------------------------------
    pred_long = rollout_from_x0(net, scaler, test_data_full[0], KMAX, dt, device)

    row["Attractor_W2"] = sinkhorn_w2_cloud(ref_full, pred_long, blur=W2_BLUR, device=device)
    row["PSD_logL1"] = psd_logl1_distance(ref_full, pred_long, dt, nfft=PSD_NFFT, channels=PSD_CHANNELS)

    # recurrence metrics: DRTD + HRE
    rec = recurrence_metrics(ref_full, pred_long, dt, sections=POINCARE_SECTIONS)
    row.update(rec)

    return row, pred_long

# ============================================================
# load + rollout + eval
# ============================================================
rows = []
models_pred = {}

KMAX = T_eval - 1

# ============================================================
# load ablation models + rollout + eval
# ============================================================
rows = []
models_pred = {}

KMAX = T_eval - 1

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

    row, pred = evaluate_model(
        name, net, scaler,
        test_data_full=test_data,
        ref_full=ref,
        dt=dt,
        KMAX=KMAX,
        n_starts=50
    )

    # checkpoint metadata if available
    if isinstance(obj, dict) and "config" in obj:
        cfg = obj["config"]
        row["ckpt_mode"] = cfg.get("ablation_K_mode", "unknown")
        row["selected_source"] = cfg.get("selected_source", "unknown")
        row["ckpt_path"] = ckpt_path

    models_pred[name] = pred
    rows.append(row)

# ============================================================
# print
# ============================================================
def fmt(x):
    if isinstance(x, float):
        return "nan" if not np.isfinite(x) else f"{x:.6g}"
    return str(x)

# 将列宽设为 20，以容纳 "0.1234 ± 0.0567" 这样的字符串
COL_WIDTH = 20

print("\n==================== SHORT-WINDOW MSE (Mean ± STD over 50 starts, lower better) ====================")
mse_cols = ["model"] + [f"MSE_{k}" for k in MSE_WINDOWS]
header = " | ".join([f"{c:>{COL_WIDTH}s}" for c in mse_cols])
print(header)
print("-" * len(header))
for r in rows:
    print(" | ".join([f"{fmt(r[c]):>{COL_WIDTH}s}" for c in mse_cols]))

print("\n==================== SHORT-WINDOW NMSE (Mean ± STD over 50 starts, lower better) ====================")
nmse_cols = ["model"] + [f"NMSE_{k}" for k in MSE_WINDOWS]
header = " | ".join([f"{c:>{COL_WIDTH}s}" for c in nmse_cols])
print(header)
print("-" * len(header))
for r in rows:
    print(" | ".join([f"{fmt(r[c]):>{COL_WIDTH}s}" for c in nmse_cols]))

print("\n==================== STRUCTURE METRICS (1 Long Rollout, lower better) ====================")
cols = [
    "model",
    "Attractor_W2",
    "PSD_logL1",
]
header = " | ".join([f"{c:>16s}" for c in cols])
print(header)
print("-" * len(header))
for r in rows:
    print(" | ".join([f"{fmt(r[c]):>16s}" for c in cols]))

print("\n==================== RECURRENCE METRICS (1 Long Rollout, lower better) ====================")
cols = [
    "model",
    "Mean_DRTD",
    "Mean_HRE",
]
for sec in POINCARE_SECTIONS:
    cols.append(f"{sec[0]}_pred_hits")
    cols.append(f"{sec[0]}_DRTD")
    cols.append(f"{sec[0]}_HRE")

header = " | ".join([f"{c:>18s}" for c in cols])
print(header)
print("-" * len(header))
for r in rows:
    print(" | ".join([f"{fmt(r.get(c, 'nan')):>18s}" for c in cols]))

print("\n==================== STRUCTURE + RECURRENCE SUMMARY (1 Long Rollout, lower better) ====================")
cols = [
    "model",
    "Attractor_W2",
    "PSD_logL1",
    "Mean_DRTD",
    "Mean_HRE",
]
header = " | ".join([f"{c:>16s}" for c in cols])
print(header)
print("-" * len(header))
for r in rows:
    print(" | ".join([f"{fmt(r[c]):>16s}" for c in cols]))

# ============================================================
# plots
# ============================================================
if PLOT_ON:
    Nplot = min(PLOT_N, T_eval)
    tt = np.arange(Nplot) * dt

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), dpi=160, sharex=True)

    ax[0].plot(tt, ref[:Nplot, 0], lw=1.2, label="ref")
    for name, pred in models_pred.items():
        ax[0].plot(tt, pred[:Nplot, 0], lw=1.0, label=name)
    ax[0].set_title("Time series overlay (ch=0) - 1 Long Rollout")
    ax[0].legend()

    for name, pred in models_pred.items():
        err = np.linalg.norm(pred[:Nplot] - ref[:Nplot], axis=1)
        ax[1].plot(tt, err, lw=1.0, label=name)
    ax[1].set_yscale("log")
    ax[1].set_title("Trajectory error norm (log)")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

print("\nDone.")

try:
    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = os.path.join(str(PORT_OUTPUT), f"lorenz_K_ablation_eval_seed{SEED}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV: {out_csv}")
except Exception as e:
    print(f"[WARN] Could not save CSV: {e}")
