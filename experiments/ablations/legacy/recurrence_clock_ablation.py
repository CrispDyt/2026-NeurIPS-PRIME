#!/usr/bin/env python3
"""
Lorenz63 PRIME recurrence-clock robustness runner.

Robustness experiment: How sensitive is PRIME to recurrence-clock noise?

Default variants:
  exact_k       : full PRIME loss with the exact data-derived recurrence intervals K_i.
  jitter_k_10   : full PRIME loss with perturbed recurrence intervals K_i' = round(K_i(1+eps_i)), eps_i ~ clipped N(0, 0.10^2).
  jitter_k_20   : full PRIME loss with perturbed recurrence intervals K_i' = round(K_i(1+eps_i)), eps_i ~ clipped N(0, 0.20^2).

Control / initialization:
  dim_only      : DIM pretraining only; also used as the shared initialization for --all.

Run all variants in fresh subprocesses:
  python lorenz_prime_clock_robustness.py --all --train_path train_data.npy

Run one variant from an existing DIM checkpoint:
  python lorenz_prime_clock_robustness.py --mode jitter_k_10 --train_path train_data.npy \
      --init_checkpoint /path/to/shared_dim_selected.p

Recommended protocol:
  --all first trains a shared DIM checkpoint for --warm_steps, then starts exact_k,
  jitter_k_10, and jitter_k_20 from that same DIM checkpoint and runs --refine_steps
  updates. The data recurrence section is held fixed; only the model-side recurrence
  clock used inside the PRIME recurrence-block objective is perturbed. This isolates
  whether the refinement depends on exact recurrence intervals.

Data format:
  train_data must be a 2D array (T, d). Supported: .npy, .npz, .pkl/.pickle/.p, .pt/.pth
"""

import argparse
import csv
import datetime
import io
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as tnn
from geomloss import SamplesLoss
from sklearn.preprocessing import MaxAbsScaler
from torch import optim


# =========================================================
# Loading utilities
# =========================================================
def extract_array_from_object(obj):
    if isinstance(obj, np.ndarray):
        return obj
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        for key in ["train_data", "train", "training", "data", "X", "Y"]:
            if key in obj:
                val = obj[key]
                return val.detach().cpu().numpy() if torch.is_tensor(val) else val
        raise KeyError(f"Could not infer train array key. Available keys: {list(obj.keys())}")
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if torch.is_tensor(item):
                arr = item.detach().cpu().numpy()
            else:
                try:
                    arr = np.asarray(item)
                except Exception:
                    continue
            if arr.ndim == 2:
                return arr
    raise TypeError(f"Could not extract a 2D train array from object of type {type(obj)}")


def load_train_array(path: str) -> np.ndarray:
    if not path:
        raise ValueError("Use --train_path /path/to/train_data.npy or set TRAIN_DATA_PATH.")
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Training data file not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p, allow_pickle=True)
    elif suffix == ".npz":
        obj = np.load(p, allow_pickle=True)
        for key in ["train_data", "train", "training", "data", "X", "Y"]:
            if key in obj:
                arr = obj[key]
                break
        else:
            keys = list(obj.keys())
            if len(keys) == 1:
                arr = obj[keys[0]]
            else:
                raise KeyError(f"Could not infer train array key in {p}. Available keys: {keys}")
    elif suffix in {".pkl", ".pickle", ".p"}:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        arr = extract_array_from_object(obj)
    elif suffix in {".pt", ".pth"}:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        arr = extract_array_from_object(obj)
    else:
        raise ValueError(f"Unsupported training data format: {suffix}")

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"train_data must be 2D (T, d), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("train_data contains non-finite values.")
    return arr


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
        return load_pickle_cpu(path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint: {path}\n"
            f"torch.load error: {last_err}\n"
            f"pickle error: {e}"
        )


def extract_state_dict_from_ckpt(path):
    obj = smart_load(path)
    sd = None
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"].get("state_dict", None)
        if sd is None and "state_dict" in obj:
            sd = obj["state_dict"]
        if sd is None and "model_state_dict" in obj:
            sd = obj["model_state_dict"]
    else:
        sd = obj
    if sd is None:
        raise RuntimeError(f"Could not find state_dict in checkpoint: {path}")
    return sd


# =========================================================
# General utilities
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_sinkhorn(blur: float):
    return SamplesLoss(loss="sinkhorn", p=2, blur=float(blur), backend="tensorized")


CLOCK_MODES = {"dim_only", "exact_k", "jitter_k_10", "jitter_k_20"}

CLOCK_JITTER_FRACS = {
    "dim_only": 0.0,
    "exact_k": 0.0,
    "jitter_k_10": 0.10,
    "jitter_k_20": 0.20,
}

# Keep this alias so the rest of the original runner remains minimally changed.
LOSS_TERM_MODES = CLOCK_MODES


def loss_term_weights_for_mode(mode: str, w_marg_u0: float, w_marg_u1: float, w_defect: float):
    """
    In this robustness experiment the PRIME loss terms are not ablated.
    All clock variants use the full PRIME recurrence-block objective:
      graph + source marginal + target marginal + optional defect.
    dim_only is the shared DIM pretraining control.
    """
    if mode == "dim_only":
        return 0.0, 0.0, 0.0, 0.0
    if mode in {"exact_k", "jitter_k_10", "jitter_k_20"}:
        return 1.0, float(w_marg_u0), float(w_marg_u1), float(w_defect)
    raise ValueError(f"Unknown clock robustness mode: {mode}")


def describe_loss_terms(mode: str, graph_w: float, marg0_w: float, marg1_w: float, defect_w: float) -> str:
    return (
        f"mode={mode} | full PRIME weights: "
        f"graph={graph_w:g}, source_marg={marg0_w:g}, "
        f"target_marg={marg1_w:g}, defect={defect_w:g}"
    )


def build_net(d: int, width: int = 100):
    return tnn.Sequential(
        tnn.Linear(d, width), tnn.Tanh(),
        tnn.Linear(width, width), tnn.Tanh(),
        tnn.Linear(width, width), tnn.Tanh(),
        tnn.Linear(width, d),
    )


def cpu_state_dict(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def build_delay_full_onech(y_scaled, tau, dim, ch):
    """
    y_scaled: (T, d)
    returns D_full: (T, dim), with lag blocks
      [y_ch(t), y_ch(t-tau), ..., y_ch(t-(dim-1)tau)]
    """
    T, _ = y_scaled.shape
    min_t = (dim - 1) * tau
    D = np.zeros((T, dim), dtype=np.float64)
    s = y_scaled[:, ch]
    for t in range(min_t, T):
        D[t] = np.array([s[t - i * tau] for i in range(dim)], dtype=np.float64)
    return D, min_t


def farthest_point_sampling(X, M, start=0):
    N = int(X.shape[0])
    M = int(min(int(M), N))
    idx = np.zeros((M,), dtype=np.int64)
    idx[0] = int(start)
    dist2 = np.sum((X - X[idx[0]]) ** 2, axis=1)
    for m in range(1, M):
        j = int(np.argmax(dist2))
        idx[m] = j
        d2_new = np.sum((X - X[j]) ** 2, axis=1)
        dist2 = np.minimum(dist2, d2_new)
    return idx


def extract_hits_from_landmarks(U_dw, C_dw, eps_in, eps_out, L_sec):
    """Hysteresis + min-gap hit extraction."""
    T = int(U_dw.shape[0])
    hits = []
    visit_lens = []
    last = -10**12
    active = False
    active_i = -1
    enter_t = -1

    for t in range(T):
        d2 = np.sum((C_dw - U_dw[t]) ** 2, axis=1)
        i = int(np.argmin(d2))
        d = float(np.sqrt(d2[i]))

        if not active:
            if d <= eps_in and (t - last) >= int(L_sec):
                hits.append(t)
                last = t
                active = True
                active_i = i
                enter_t = t
        else:
            d_active = float(np.sqrt(np.sum((U_dw[t] - C_dw[active_i]) ** 2)))
            if d_active >= eps_out:
                active = False
                active_i = -1
                if enter_t >= 0:
                    visit_lens.append(t - enter_t)
                enter_t = -1

    return np.array(hits, dtype=np.int64), np.array(visit_lens, dtype=np.int64)


def extract_exact_level_hits_raw(train_raw, ch, level, direction, min_t, L_sec):
    """
    Integer hit indices from raw-coordinate level crossings.
    direction=+1: upward crossing. direction=-1: downward crossing. direction=0: any crossing.
    Hit index is i+1, the first sample after crossing.
    """
    x = np.asarray(train_raw[:, ch], dtype=np.float64)
    a = x[:-1] - float(level)
    b = x[1:] - float(level)

    if direction > 0:
        mask = (a < 0) & (b >= 0)
    elif direction < 0:
        mask = (a > 0) & (b <= 0)
    else:
        mask = (a == 0) | (a * b < 0)

    idx = np.where(mask)[0] + 1
    hits = []
    last = -10**12
    for t in idx:
        t = int(t)
        if t < min_t:
            continue
        if t - last >= int(L_sec):
            hits.append(t)
            last = t
    return np.asarray(hits, dtype=np.int64)


def sample_random_hits(T, min_t, n_hits, L_sec, seed):
    """
    Random negative-control recurrence times with roughly the same number of hits.
    Uses greedy min-gap filtering over a random permutation.
    """
    rng = np.random.default_rng(seed)
    candidates = np.arange(min_t, T, dtype=np.int64)
    if len(candidates) == 0:
        return np.asarray([], dtype=np.int64)

    for _ in range(200):
        perm = rng.permutation(candidates)
        selected = []
        for t in perm:
            if all(abs(int(t) - int(s)) >= int(L_sec) for s in selected):
                selected.append(int(t))
                if len(selected) >= n_hits:
                    break
        if len(selected) >= max(4, min(n_hits, len(candidates))):
            return np.asarray(sorted(selected[:n_hits]), dtype=np.int64)

    # Fallback: evenly spaced random-ish grid with jitter.
    n = min(n_hits, max(1, len(candidates) // max(1, L_sec)))
    base = np.linspace(min_t, T - 1, n, dtype=np.int64)
    jitter = rng.integers(-max(1, L_sec // 2), max(2, L_sec // 2 + 1), size=n)
    hits = np.clip(base + jitter, min_t, T - 1)
    hits = np.unique(np.sort(hits)).astype(np.int64)
    return hits


def jitter_recurrence_intervals(taus_exact, frac, rng, clip_sigma=2.0):
    """
    Perturb recurrence intervals K_i while preserving positivity.

    frac=0.10 means multiplicative Gaussian clock noise with standard deviation 10%:
      K_i' = round(K_i * (1 + eps_i)), eps_i ~ N(0, frac^2), clipped to +/- clip_sigma*frac.

    This simulates imperfect recurrence-clock extraction while keeping the data recurrence
    coordinates fixed. The PRIME target graph remains the true data return graph, but
    the model-side multi-shooting horizons are noisy.
    """
    taus_exact = np.asarray(taus_exact, dtype=np.int64)
    if frac <= 0.0 or len(taus_exact) == 0:
        return taus_exact.copy()
    eps = rng.normal(loc=0.0, scale=float(frac), size=len(taus_exact))
    eps = np.clip(eps, -float(clip_sigma) * float(frac), float(clip_sigma) * float(frac))
    taus = np.rint(taus_exact.astype(np.float64) * (1.0 + eps)).astype(np.int64)
    return np.maximum(1, taus)


def build_delay_data_tensor_onech(y_np, dim_delay, tau_delay, ch, device):
    """One-channel delay tensor: (1, L, dim_delay)."""
    Y = torch.tensor(y_np, dtype=torch.float32, device=device)
    s = Y[:, ch]
    T = s.shape[0]
    L = T - tau_delay * (dim_delay - 1)
    delayed = torch.zeros((1, L, dim_delay), dtype=torch.float32, device=device)
    for i in range(dim_delay):
        start = dim_delay * tau_delay - (i + 1) * tau_delay
        delayed[0, :, i] = s[start:start + L]
    return delayed


def plot_history(path, values, labels, title):
    plt.figure(figsize=(10, 3), dpi=140)
    for v, lab in zip(values, labels):
        plt.plot(v, label=lab)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def latest_file(patterns):
    files = []
    for p in patterns:
        files.extend(Path().glob(p) if not os.path.isabs(p) else [])
    # Path().glob does not handle normal relative string with directories well when p has './'.
    # Use glob module robustly below.
    import glob
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(files, key=os.path.getmtime)
    return files[-1] if files else None


def find_selected_checkpoint(folder):
    folder = str(folder)
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


# =========================================================
# Main training routine for one mode
# =========================================================
def train_one(args):
    dt_obs = args.dt_obs
    dt_base = args.dt_base
    sub = int(round(dt_obs / dt_base))
    if sub < 1:
        raise ValueError(f"sub={sub} < 1; check dt_obs={dt_obs}, dt_base={dt_base}")

    seed = args.seed
    mode = args.mode
    allowed_modes = LOSS_TERM_MODES
    if mode not in allowed_modes:
        raise ValueError(f"Unknown mode={mode}. Allowed: {sorted(allowed_modes)}")

    cand_frac = args.cand_frac
    q_speed = args.q_speed
    M_landmarks = args.M_landmarks
    L_sec = args.L_sec
    tau_delay = args.tau_delay
    dim_delay = args.dim_delay
    eps_in_mult = args.eps_in_mult
    eps_out_mult = args.eps_out_mult
    DELAY_CH = args.delay_ch

    # If initialized from a shared DIM checkpoint, skip warmup and use refine_steps.
    if args.init_checkpoint:
        Nsteps = int(args.refine_steps)
        warm_steps = 0
    else:
        Nsteps = int(args.Nsteps)
        warm_steps = int(args.warm_steps)

    tau_pair = args.tau_pair
    stage1_batch = args.stage1_batch
    lr1 = args.lr1
    L_hits_fullbatch = args.L_hits_fullbatch
    BATCH_SIZE_STAGE2 = args.BATCH_SIZE_STAGE2
    lr2 = args.lr2
    alpha_pair_state = args.alpha_pair_state
    alpha_pair_delay = args.alpha_pair_delay
    w_marg_u0 = args.w_marg_u0
    w_marg_u1 = args.w_marg_u1
    w_defect = args.w_defect
    graph_weight, marg0_weight, marg1_weight, defect_weight = loss_term_weights_for_mode(
        mode, w_marg_u0=w_marg_u0, w_marg_u1=w_marg_u1, w_defect=w_defect
    )
    loss_term_description = describe_loss_terms(mode, graph_weight, marg0_weight, marg1_weight, defect_weight)
    blur_start = args.blur_start
    blur_end = args.blur_end
    stage2_steps_per_cycle = args.stage2_steps_per_cycle
    stage1_steps_per_cycle = args.stage1_steps_per_cycle
    w2_const = args.w2_const

    if mode == "dim_only":
        warm_steps = Nsteps
        stage2_steps_per_cycle = 0
        stage1_steps_per_cycle = 1

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    set_seed(seed)

    out_dir = Path(args.run_root) / f"{mode}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_prefix = f"ClockRobust_{mode}_seed{seed}"

    print("device:", device)
    print("OUT_DIR:", out_dir)
    print("mode:", mode)
    print("prime_loss_terms:", loss_term_description)
    print("init_checkpoint:", args.init_checkpoint if args.init_checkpoint else "None")

    train = load_train_array(args.train_path)
    T, d = train.shape
    if DELAY_CH < 0 or DELAY_CH >= d:
        raise ValueError(f"delay_ch={DELAY_CH} out of range for d={d}")

    scaler = MaxAbsScaler().fit(train)
    y_scaled = scaler.transform(train).astype(np.float64)

    # Delay and augmented coordinates used for implicit section and PRIME atoms.
    D_full, min_t = build_delay_full_onech(y_scaled, tau=tau_delay, dim=dim_delay, ch=DELAY_CH)
    U_full = np.concatenate([y_scaled, D_full], axis=1)
    T_eff = T - min_t
    if T_eff <= 10:
        raise ValueError(f"Too few effective samples: T={T}, min_t={min_t}")

    # Build implicit hits no matter what; random_hits uses its hit count, and dim_only stores stats.
    speed = np.linalg.norm(y_scaled[1:] - y_scaled[:-1], axis=1)
    speed = np.concatenate([[speed[0]], speed], axis=0)
    idx_eff = np.arange(min_t, T)
    thr = np.quantile(speed[idx_eff], q_speed)
    pool = idx_eff[speed[idx_eff] >= thr]
    cand_target = max(1, int(round(cand_frac * T_eff)))
    if len(pool) >= cand_target:
        idx_cand = np.random.choice(pool, size=cand_target, replace=False)
    else:
        idx_cand = pool.copy()
    idx_cand = np.sort(idx_cand)
    if len(idx_cand) < 2:
        raise RuntimeError("Too few candidate points for landmark construction.")

    Ucand = U_full[idx_cand]
    u_mean_c = Ucand.mean(axis=0)
    u_std_c = Ucand.std(axis=0) + 1e-6
    Ucand_dw = (Ucand - u_mean_c[None, :]) / u_std_c[None, :]
    Ufull_dw = (U_full - u_mean_c[None, :]) / u_std_c[None, :]

    start0 = int(np.random.randint(0, len(idx_cand)))
    fps_idx_local = farthest_point_sampling(Ucand_dw, M=M_landmarks, start=start0)
    C_dw = Ucand_dw[fps_idx_local]

    nn = []
    for i in range(len(C_dw)):
        d2 = np.sum((C_dw - C_dw[i]) ** 2, axis=1)
        d2[i] = np.inf
        nn.append(np.sqrt(np.min(d2)))
    nn_med = float(np.median(nn))
    eps_in = eps_in_mult * nn_med
    eps_out = eps_out_mult * eps_in

    implicit_hits_raw, implicit_visit_lens = extract_hits_from_landmarks(
        Ufull_dw, C_dw, eps_in, eps_out, L_sec=L_sec
    )
    implicit_hits = np.sort(implicit_hits_raw[implicit_hits_raw >= min_t])

    # Choose exact section defaults based on data dimension.
    # Partial Lorenz [X,Z] => Z is channel 1. Full Lorenz [X,Y,Z] => Z is channel 2.
    z_ch_default = 1 if d == 2 else 2
    exact_z_ch = args.exact_z_ch if args.exact_z_ch is not None else z_ch_default
    if exact_z_ch >= d:
        raise ValueError(f"exact_z_ch={exact_z_ch} out of range for train_data dimension d={d}")

    exact_z40_hits = extract_exact_level_hits_raw(
        train, ch=exact_z_ch, level=args.exact_z_level,
        direction=args.exact_direction, min_t=min_t, L_sec=L_sec
    )
    exact_xm10_hits = extract_exact_level_hits_raw(
        train, ch=0, level=-10.0, direction=args.exact_direction, min_t=min_t, L_sec=L_sec
    )
    exact_x0_hits = extract_exact_level_hits_raw(
        train, ch=0, level=0.0, direction=args.exact_direction, min_t=min_t, L_sec=L_sec
    )

    # Clock-robustness experiment holds the data recurrence section fixed.
    # Exact-K uses the empirical inter-event intervals K_i. Jitter variants keep
    # the same data return graph but perturb the model-side multi-shooting horizons.
    hits = implicit_hits
    visit_lens = implicit_visit_lens
    section_description = "implicit_landmark_recurrence_section"

    hits = np.sort(np.asarray(hits, dtype=np.int64))
    hits = hits[(hits >= min_t) & (hits < T)]
    if len(hits) < 4 and mode != "dim_only":
        raise RuntimeError(f"Too few hits for mode={mode}: {len(hits)}")

    # Training code uses at most L_hits_fullbatch.
    if len(hits) < L_hits_fullbatch and mode != "dim_only":
        print(f"[WARN] N_hits={len(hits)} < L_hits_fullbatch={L_hits_fullbatch}; using smaller blocks.")

    taus_exact = np.diff(hits).astype(np.int64) if len(hits) >= 2 else np.array([], dtype=np.int64)
    clock_jitter_frac = float(CLOCK_JITTER_FRACS[mode])
    rng_clock = np.random.default_rng(
        int(seed) + int(args.clock_jitter_seed_offset) + int(round(10000.0 * clock_jitter_frac))
    )
    taus_model = jitter_recurrence_intervals(
        taus_exact, frac=clock_jitter_frac, rng=rng_clock, clip_sigma=args.clock_jitter_clip_sigma
    )
    taus = taus_exact  # backward-compatible name: exact data recurrence intervals.

    if mode != "dim_only":
        assert len(taus_exact) == len(hits) - 1
        assert len(taus_model) == len(hits) - 1
        assert np.all(taus_exact > 0)
        assert np.all(taus_model > 0)

    if len(taus_exact) > 0:
        delta_tau = taus_model.astype(np.float64) - taus_exact.astype(np.float64)
        rel_abs = np.abs(delta_tau) / np.maximum(1.0, taus_exact.astype(np.float64))
        drift = np.cumsum(delta_tau)
    else:
        delta_tau = np.array([], dtype=np.float64)
        rel_abs = np.array([], dtype=np.float64)
        drift = np.array([], dtype=np.float64)

    clock_description = (
        f"data_section=fixed implicit landmarks | model_clock={mode} | "
        f"jitter_frac={clock_jitter_frac:.3f}"
    )

    print("=" * 72)
    print(f"Recurrence-clock robustness mode: {mode}")
    print(f"section_description: {section_description}")
    print(f"clock_description: {clock_description}")
    print(f"loss_term_description: {loss_term_description}")
    print(f"N_hits={len(hits)} | implicit_N={len(implicit_hits)} | exact_z40_N={len(exact_z40_hits)}")
    if len(taus_exact) > 0:
        print(
            f"Exact K stats: median={np.median(taus_exact):.3f}, mean={taus_exact.mean():.3f}, "
            f"std={taus_exact.std():.3f}, p10={np.quantile(taus_exact, 0.1):.1f}, "
            f"p90={np.quantile(taus_exact, 0.9):.1f}"
        )
        print(
            f"Model K stats: median={np.median(taus_model):.3f}, mean={taus_model.mean():.3f}, "
            f"std={taus_model.std():.3f}, p10={np.quantile(taus_model, 0.1):.1f}, "
            f"p90={np.quantile(taus_model, 0.9):.1f}"
        )
        print(
            f"Clock perturbation: mean_abs_rel={rel_abs.mean():.4f}, max_abs_rel={rel_abs.max():.4f}, "
            f"final_cumulative_drift={drift[-1]:.1f}, max_abs_cumulative_drift={np.max(np.abs(drift)):.1f}"
        )
    print("=" * 72)

    # Data blocks at selected hits.
    if len(hits) > 0:
        XU_data = y_scaled[hits]
        DU_data = D_full[hits]
        U_hits_data = np.concatenate([XU_data, DU_data], axis=1)
    else:
        # only for dim_only fallback
        XU_data = np.zeros((1, d), dtype=np.float64)
        DU_data = np.zeros((1, dim_delay), dtype=np.float64)
        U_hits_data = np.concatenate([XU_data, DU_data], axis=1)

    Y_all_t = torch.tensor(y_scaled, dtype=torch.float32, device=device)

    u_mean = U_hits_data.mean(axis=0)
    u_std = U_hits_data.std(axis=0) + 1e-6
    u_mean_t = torch.tensor(u_mean, dtype=torch.float32, device=device)
    u_std_t = torch.tensor(u_std, dtype=torch.float32, device=device)

    def whiten_u(U):
        return (U - u_mean_t[None, :]) / u_std_t[None, :]

    x_mean = y_scaled.mean(axis=0)
    x_std = y_scaled.std(axis=0) + 1e-6
    x_mean_t = torch.tensor(x_mean, dtype=torch.float32, device=device)
    x_std_t = torch.tensor(x_std, dtype=torch.float32, device=device)

    def whiten_x(X):
        return (X - x_mean_t[None, :]) / x_std_t[None, :]

    y_delayed_data = build_delay_data_tensor_onech(y_scaled, dim_delay, tau_delay, DELAY_CH, device)

    net = build_net(d, width=args.width).to(device)
    if args.init_checkpoint:
        init_sd = extract_state_dict_from_ckpt(args.init_checkpoint)
        net.load_state_dict(init_sd)
        print(f"Loaded init checkpoint: {args.init_checkpoint}")

    opt1 = optim.Adam(net.parameters(), lr=lr1)
    opt2 = optim.Adam(net.parameters(), lr=lr2)

    def compute_L1(loss_sink):
        Tmax = len(y_scaled) - tau_pair - 1
        idx = np.random.randint(0, Tmax, size=stage1_batch)
        x0 = torch.tensor(y_scaled[idx], dtype=torch.float32, device=device)
        xT_d = torch.tensor(y_scaled[idx + tau_pair], dtype=torch.float32, device=device)

        x = x0.clone()
        for _ in range(int(tau_pair * sub)):
            x = x + float(dt_base) * net(x)
        xT_m = x

        x0w = whiten_x(x0)
        xTd = whiten_x(xT_d)
        xTm = whiten_x(xT_m)
        pair_d_state = torch.cat([x0w, xTd], dim=1)
        pair_m_state = torch.cat([x0w, xTm], dim=1)
        L_state = loss_sink(pair_m_state, pair_d_state)

        Kfine = tau_delay * dim_delay * sub
        yy = Y_all_t.clone()
        sols = []
        for _ in range(Kfine):
            sols.append(yy)
            for _ in range(sub):
                yy = yy + float(dt_base) * net(yy)
        sol = torch.stack(sols, dim=0)
        sol = sol[:, min_t:, :]

        step_fine = tau_delay * sub
        sol_ch = sol[:, :, DELAY_CH].permute(1, 0).contiguous()
        DIM_model = torch.flip(sol_ch[:, ::step_fine], dims=[1]).unsqueeze(0).contiguous()
        y_delay = y_delayed_data
        L_delay = loss_sink(DIM_model, y_delay).sum()
        glue_defect = torch.mean((DIM_model[:, 0, :-1] - y_delay[:, 0, :-1]) ** 2)
        return L_state + L_delay + glue_defect

    def build_induced_block(k0, L_hits, loss_sink):
        # Data-side recurrence coordinates stay at the extracted section hits.
        hit_times_data = hits[k0:k0 + L_hits].astype(np.int64)

        # Model-side recurrence clock can be exact or jittered. The first event is anchored
        # at the same data hit, then subsequent model-side events use K_i' from taus_model.
        if L_hits > 1:
            model_intervals = taus_model[k0:k0 + L_hits - 1].astype(np.int64)
            hit_offsets_model = np.concatenate([[0], np.cumsum(model_intervals)]).astype(np.int64)
        else:
            hit_offsets_model = np.array([0], dtype=np.int64)
        hit_times_model = (int(hit_times_data[0]) + hit_offsets_model).astype(np.int64)

        X_blk_d = torch.tensor(XU_data[k0:k0 + L_hits], dtype=torch.float32, device=device)
        D_blk_d = torch.tensor(DU_data[k0:k0 + L_hits], dtype=torch.float32, device=device)

        t0 = int(hit_times_model[0])
        t_start = t0 - int((dim_delay - 1) * tau_delay)
        if t_start < 0:
            raise RuntimeError(f"t_start={t_start}<0")
        t_end = int(hit_times_model[-1])
        total_obs_steps = int(t_end - t_start)

        x = torch.tensor(y_scaled[t_start], dtype=torch.float32, device=device)
        hit_map = {int(t): j for j, t in enumerate(hit_times_model.tolist())}
        if len(hit_map) != L_hits:
            raise RuntimeError(f"Duplicate model hit times after clock jitter. mode={mode}, k0={k0}")
        x_hits_m = [None] * L_hits
        x_hist = [x.clone()]

        for n in range(total_obs_steps + 1):
            t_abs = t_start + n
            if t_abs in hit_map:
                x_hits_m[hit_map[t_abs]] = x
                x = x.detach()
            if n == total_obs_steps:
                break
            for _ in range(sub):
                x = x + float(dt_base) * net(x)
            x_hist.append(x.clone())

        if any(v is None for v in x_hits_m):
            raise RuntimeError(f"Some model hits were not recorded. mode={mode}, k0={k0}, data_hits={hit_times_data.tolist()}, model_hits={hit_times_model.tolist()}")

        X_blk_m = torch.stack(x_hits_m, dim=0)

        Dm_list = []
        for t_abs in hit_times_model:
            idx0 = int(t_abs - t_start)
            lag_vals = []
            for i in range(dim_delay):
                lag_idx = idx0 - i * tau_delay
                if lag_idx < 0:
                    raise RuntimeError(f"Negative lag index: {lag_idx}")
                lag_vals.append(x_hist[lag_idx][DELAY_CH])
            Dm_list.append(torch.stack(lag_vals, dim=0))
        D_blk_m = torch.stack(Dm_list, dim=0)

        U_blk_m = torch.cat([X_blk_m, D_blk_m], dim=1)
        U_blk_d = torch.cat([X_blk_d, D_blk_d], dim=1)

        Um = whiten_u(U_blk_m)
        Ud = whiten_u(U_blk_d)
        xm, dm = Um[:, :d], Um[:, d:]
        xd, dd = Ud[:, :d], Ud[:, d:]

        u_m = torch.cat([alpha_pair_state * xm, alpha_pair_delay * dm], dim=1)
        u_d = torch.cat([alpha_pair_state * xd, alpha_pair_delay * dd], dim=1)

        u0_m, u1_m = u_m[:-1], u_m[1:]
        u0_d, u1_d = u_d[:-1], u_d[1:]
        pair_m = torch.cat([u0_m, u1_m], dim=1)
        pair_d = torch.cat([u0_d, u1_d], dim=1)

        L2_graph = loss_sink(pair_m, pair_d)
        L2_marg0 = loss_sink(u0_m, u0_d)
        L2_marg1 = loss_sink(u1_m, u1_d)
        defect = torch.mean((Um[:, :d] - Ud[:, :d]) ** 2)

        # Clock robustness: every non-DIM variant uses the full PRIME loss terms;
        # the only experimental change is exact vs jittered model-side recurrence K_i.
        L2_raw = (
            float(graph_weight) * L2_graph
            + float(marg0_weight) * L2_marg0
            + float(marg1_weight) * L2_marg1
            + float(defect_weight) * defect
        )
        L2_norm = L2_raw / float(L_hits)
        return L2_norm, L2_raw, L2_graph, L2_marg0, L2_marg1, defect

    best_total = float("inf")
    best_total_it = -1
    best_total_sd = None
    best_total_L1 = None
    best_total_L2raw = None
    best_total_L2norm = None

    best_L2raw = float("inf")
    best_L2_it = -1
    best_L2_sd = None
    best_L2_at_best = None
    best_L1_at_best = None
    best_L2norm_at_best = None

    loss_tot_hist, loss_L1_hist, loss_L2norm_hist, loss_L2raw_hist, stage_hist = [], [], [], [], []
    loss_L2graph_hist, loss_L2marg0_hist, loss_L2marg1_hist, loss_defect_hist = [], [], [], []

    print("=" * 72)
    print("Start training")
    print(f"mode={mode} | section={section_description}")
    print(clock_description)
    print(loss_term_description)
    print(f"Nsteps={Nsteps} | warm={warm_steps} | N_hits={len(hits)} | L_hits_fullbatch={L_hits_fullbatch}")
    print(f"Stage cycle: stage2={stage2_steps_per_cycle}, stage1={stage1_steps_per_cycle}")
    print("=" * 72)

    net.train()
    for step in range(Nsteps):
        tfrac = step / max(1, Nsteps - 1)
        blur = blur_start + (blur_end - blur_start) * tfrac
        loss_sink = make_sinkhorn(blur)

        if step < warm_steps:
            stage = "stage1"
        else:
            cyc = stage2_steps_per_cycle + stage1_steps_per_cycle
            if cyc <= 0:
                stage = "stage1"
            else:
                k = (step - warm_steps) % cyc
                stage = "stage2" if k < stage2_steps_per_cycle else "stage1"

        opt = opt1 if stage == "stage1" else opt2
        opt.zero_grad(set_to_none=True)
        L1 = compute_L1(loss_sink)

        if stage == "stage1" or mode == "dim_only":
            L2_norm = torch.tensor(0.0, device=device)
            L2_raw = torch.tensor(0.0, device=device)
            L2_graph_v = torch.tensor(0.0, device=device)
            L2_marg0_v = torch.tensor(0.0, device=device)
            L2_marg1_v = torch.tensor(0.0, device=device)
            L2_defect_v = torch.tensor(0.0, device=device)
            Ltot = L1
        else:
            L_hits = min(L_hits_fullbatch, len(hits))
            if L_hits < 4:
                raise RuntimeError(f"Too few hits for stage2: {len(hits)}")
            k0max = len(hits) - L_hits
            l2n_list, l2r_list = [], []
            l2g_list, l2m0_list, l2m1_list, l2d_list = [], [], [], []
            for _ in range(BATCH_SIZE_STAGE2):
                k0 = np.random.randint(0, k0max + 1)
                l2n, l2r, l2g, l2m0, l2m1, l2d = build_induced_block(k0, L_hits, loss_sink)
                l2n_list.append(l2n)
                l2r_list.append(l2r)
                l2g_list.append(l2g)
                l2m0_list.append(l2m0)
                l2m1_list.append(l2m1)
                l2d_list.append(l2d)
            L2_norm = torch.stack(l2n_list).mean()
            L2_raw = torch.stack(l2r_list).mean()
            L2_graph_v = torch.stack(l2g_list).mean()
            L2_marg0_v = torch.stack(l2m0_list).mean()
            L2_marg1_v = torch.stack(l2m1_list).mean()
            L2_defect_v = torch.stack(l2d_list).mean()
            Ltot = L1 + float(w2_const) * L2_norm

        Ltot.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        ltot_v = float(Ltot.detach().cpu())
        l1_v = float(L1.detach().cpu())
        l2n_v = float(L2_norm.detach().cpu())
        l2r_v = float(L2_raw.detach().cpu())
        l2g_v = float(L2_graph_v.detach().cpu())
        l2m0_v = float(L2_marg0_v.detach().cpu())
        l2m1_v = float(L2_marg1_v.detach().cpu())
        l2d_v = float(L2_defect_v.detach().cpu())

        loss_tot_hist.append(ltot_v)
        loss_L1_hist.append(l1_v)
        loss_L2norm_hist.append(l2n_v)
        loss_L2raw_hist.append(l2r_v)
        loss_L2graph_hist.append(l2g_v)
        loss_L2marg0_hist.append(l2m0_v)
        loss_L2marg1_hist.append(l2m1_v)
        loss_defect_hist.append(l2d_v)
        stage_hist.append(stage)

        if stage == "stage2" and mode != "dim_only":
            if ltot_v < best_total:
                best_total = ltot_v
                best_total_it = step
                best_total_sd = cpu_state_dict(net)
                best_total_L1 = l1_v
                best_total_L2norm = l2n_v
                best_total_L2raw = l2r_v
                print(
                    f"  --> [New BEST_TOTAL] Iter {step} | L_total={best_total:.6f} | "
                    f"L1={best_total_L1:.6f} | L2_norm={best_total_L2norm:.6f} | L2_raw={best_total_L2raw:.6f}"
                )
            if l2r_v < best_L2raw:
                best_L2raw = l2r_v
                best_L2_it = step
                best_L2_sd = cpu_state_dict(net)
                best_L2_at_best = ltot_v
                best_L1_at_best = l1_v
                best_L2norm_at_best = l2n_v
                print(
                    f"  --> [New BEST_L2] Iter {step} | L2_raw={best_L2raw:.6f} | "
                    f"L_total={best_L2_at_best:.6f} | L1={best_L1_at_best:.6f} | L2_norm={best_L2norm_at_best:.6f}"
                )

        if step % args.print_every == 0:
            print(
                f"Iter {step:5d} [{stage}] | Ltot={ltot_v:.6f} | L1={l1_v:.6f} | "
                f"L2_norm={l2n_v:.6f} | L2_raw={l2r_v:.6f} | "
                f"graph={l2g_v:.6f} | m0={l2m0_v:.6f} | m1={l2m1_v:.6f} | "
                f"blur={blur:.3f}"
            )

    print("Training done.")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path_final = out_dir / f"{save_prefix}_FINAL_{stamp}.p"
    path_best = out_dir / f"{save_prefix}_BEST_{stamp}.p"
    path_best_l2 = out_dir / f"{save_prefix}_BEST_L2_{stamp}.p"
    path_selected = out_dir / f"{save_prefix}_SELECTED_{stamp}.p"

    config = dict(
        seed=int(seed),
        mode=str(mode),
        section_description=str(section_description),
        clock_description=str(clock_description),
        clock_jitter_frac=float(clock_jitter_frac),
        clock_jitter_clip_sigma=float(args.clock_jitter_clip_sigma),
        clock_jitter_seed_offset=int(args.clock_jitter_seed_offset),
        loss_term_description=str(loss_term_description),
        init_checkpoint=str(args.init_checkpoint) if args.init_checkpoint else "",
        dt_obs=float(dt_obs), dt_base=float(dt_base), sub=int(sub),
        tau_pair=int(tau_pair), tau_delay=int(tau_delay), dim_delay=int(dim_delay),
        L_sec=int(L_sec), M_landmarks=int(M_landmarks), cand_frac=float(cand_frac), q_speed=float(q_speed),
        eps_in_mult=float(eps_in_mult), eps_out_mult=float(eps_out_mult),
        L_hits_fullbatch=int(L_hits_fullbatch), BATCH_SIZE_STAGE2=int(BATCH_SIZE_STAGE2),
        lr1=float(lr1), lr2=float(lr2), w2_const=float(w2_const),
        alpha_pair_state=float(alpha_pair_state), alpha_pair_delay=float(alpha_pair_delay),
        w_marg_u0=float(w_marg_u0), w_marg_u1=float(w_marg_u1), w_defect=float(w_defect),
        graph_weight=float(graph_weight), marg0_weight=float(marg0_weight),
        marg1_weight=float(marg1_weight), defect_weight=float(defect_weight),
        blur_start=float(blur_start), blur_end=float(blur_end),
        warm_steps=int(warm_steps), Nsteps=int(Nsteps), refine_steps=int(args.refine_steps),
        min_t=int(min_t), eps_in=float(eps_in), eps_out=float(eps_out), nn_med=float(nn_med),
        N_hits=int(len(hits)), implicit_N_hits=int(len(implicit_hits)), exact_z40_N_hits=int(len(exact_z40_hits)),
        exact_K_mean=float(taus_exact.mean()) if len(taus_exact) else 0.0,
        model_K_mean=float(taus_model.mean()) if len(taus_model) else 0.0,
        mean_abs_relative_K_error=float(rel_abs.mean()) if len(rel_abs) else 0.0,
        max_abs_relative_K_error=float(rel_abs.max()) if len(rel_abs) else 0.0,
        final_cumulative_K_drift=float(drift[-1]) if len(drift) else 0.0,
        max_abs_cumulative_K_drift=float(np.max(np.abs(drift))) if len(drift) else 0.0,
        exact_z_ch=int(exact_z_ch), exact_z_level=float(args.exact_z_level), exact_direction=int(args.exact_direction),
    )

    base_payload = dict(
        config=config,
        model=dict(transformer=scaler, state_dict=None),
        stats=dict(
            hits=hits,
            data_hits=hits,
            taus_steps=taus_exact,
            exact_taus_steps=taus_exact,
            model_taus_steps=taus_model,
            clock_delta_steps=delta_tau,
            clock_abs_relative_error=rel_abs,
            clock_cumulative_drift=drift,
            visit_lens=visit_lens,
            C_dw=C_dw,
            u_mean_c=u_mean_c,
            u_std_c=u_std_c,
            u_mean=u_mean,
            u_std=u_std,
            implicit_hits=implicit_hits,
            exact_z40_hits=exact_z40_hits,
            exact_xm10_hits=exact_xm10_hits,
            exact_x0_hits=exact_x0_hits,
        ),
        train_hist=dict(
            stage=np.array(stage_hist, dtype=object),
            L_total=np.array(loss_tot_hist, dtype=np.float64),
            L1=np.array(loss_L1_hist, dtype=np.float64),
            L2_norm=np.array(loss_L2norm_hist, dtype=np.float64),
            L2_raw=np.array(loss_L2raw_hist, dtype=np.float64),
            L2_graph_unweighted=np.array(loss_L2graph_hist, dtype=np.float64),
            L2_source_marginal_unweighted=np.array(loss_L2marg0_hist, dtype=np.float64),
            L2_target_marginal_unweighted=np.array(loss_L2marg1_hist, dtype=np.float64),
            L2_defect_unweighted=np.array(loss_defect_hist, dtype=np.float64),
        )
    )

    payload_final = dict(base_payload)
    payload_final["model"] = dict(transformer=scaler, state_dict=cpu_state_dict(net))
    with open(path_final, "wb") as f:
        pickle.dump(payload_final, f)
    print("Saved FINAL:", path_final)

    if best_total_sd is not None:
        payload_best = dict(base_payload)
        payload_best["config"] = dict(config)
        payload_best["config"].update(dict(
            best_total_iter=int(best_total_it),
            best_total=float(best_total),
            best_total_L1=float(best_total_L1),
            best_total_L2_norm=float(best_total_L2norm),
            best_total_L2_raw=float(best_total_L2raw),
        ))
        payload_best["model"] = dict(transformer=scaler, state_dict=best_total_sd)
        with open(path_best, "wb") as f:
            pickle.dump(payload_best, f)
        print(f"Saved BEST_TOTAL: {path_best}  (L_total={best_total:.6f} @ iter {best_total_it})")
    else:
        print("No BEST_TOTAL saved.")

    if best_L2_sd is not None:
        payload_best_l2 = dict(base_payload)
        payload_best_l2["config"] = dict(config)
        payload_best_l2["config"].update(dict(
            best_l2_iter=int(best_L2_it),
            best_l2_L2_raw=float(best_L2raw),
            best_l2_L_total=float(best_L2_at_best),
            best_l2_L1=float(best_L1_at_best),
            best_l2_L2_norm=float(best_L2norm_at_best),
        ))
        payload_best_l2["model"] = dict(transformer=scaler, state_dict=best_L2_sd)
        with open(path_best_l2, "wb") as f:
            pickle.dump(payload_best_l2, f)
        print(f"Saved BEST_L2: {path_best_l2}  (L2_raw={best_L2raw:.6f} @ iter {best_L2_it})")
    else:
        print("No BEST_L2 saved.")

    if mode == "dim_only":
        selected_sd = cpu_state_dict(net)
        selected_source = "FINAL_DIM_ONLY"
    elif best_total_sd is not None:
        selected_sd = best_total_sd
        selected_source = "BEST_TOTAL"
    else:
        selected_sd = cpu_state_dict(net)
        selected_source = "FINAL_FALLBACK"

    payload_selected = dict(base_payload)
    payload_selected["config"] = dict(config)
    payload_selected["config"].update(dict(
        selected_source=selected_source,
        selected_path=str(path_selected),
    ))
    payload_selected["model"] = dict(transformer=scaler, state_dict=selected_sd)
    with open(path_selected, "wb") as f:
        pickle.dump(payload_selected, f)
    print(f"Saved SELECTED: {path_selected}  source={selected_source}")

    plot_history(out_dir / f"{save_prefix}_loss_{stamp}.png",
                 [loss_tot_hist, loss_L1_hist], ["L_total", "L1"], "Loss")
    plot_history(out_dir / f"{save_prefix}_L2norm_{stamp}.png",
                 [loss_L2norm_hist], ["L2_norm"], "Stage2 loss norm")
    plot_history(out_dir / f"{save_prefix}_L2raw_{stamp}.png",
                 [loss_L2raw_hist], ["L2_raw"], "Stage2 loss raw")
    plot_history(out_dir / f"{save_prefix}_L2components_{stamp}.png",
                 [loss_L2graph_hist, loss_L2marg0_hist, loss_L2marg1_hist, loss_defect_hist],
                 ["graph", "source_marg", "target_marg", "defect"],
                 "Unweighted recurrence-block components")

    return str(path_selected)


# =========================================================
# Summary utility
# =========================================================
def write_clock_robustness_summary(run_root, modes, seed):
    """Write a lightweight CSV summary from the SELECTED checkpoints."""
    run_root = Path(run_root)
    rows = []
    for mode in modes:
        folder = run_root / f"{mode}_seed{seed}"
        try:
            ckpt = find_selected_checkpoint(folder)
            obj = smart_load(ckpt)
            cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
            hist = obj.get("train_hist", {}) if isinstance(obj, dict) else {}
            L_total = np.asarray(hist.get("L_total", []), dtype=np.float64)
            L1 = np.asarray(hist.get("L1", []), dtype=np.float64)
            L2_raw = np.asarray(hist.get("L2_raw", []), dtype=np.float64)
            rows.append({
                "mode": mode,
                "selected_checkpoint": str(ckpt),
                "N_hits": cfg.get("N_hits", ""),
                "clock_jitter_frac": cfg.get("clock_jitter_frac", ""),
                "mean_abs_relative_K_error": cfg.get("mean_abs_relative_K_error", ""),
                "max_abs_relative_K_error": cfg.get("max_abs_relative_K_error", ""),
                "final_cumulative_K_drift": cfg.get("final_cumulative_K_drift", ""),
                "max_abs_cumulative_K_drift": cfg.get("max_abs_cumulative_K_drift", ""),
                "selected_source": cfg.get("selected_source", ""),
                "best_total": cfg.get("best_total", ""),
                "best_total_iter": cfg.get("best_total_iter", ""),
                "best_total_L1": cfg.get("best_total_L1", ""),
                "best_total_L2_raw": cfg.get("best_total_L2_raw", ""),
                "final_L_total": float(L_total[-1]) if L_total.size else "",
                "final_L1": float(L1[-1]) if L1.size else "",
                "final_L2_raw": float(L2_raw[-1]) if L2_raw.size else "",
            })
        except Exception as e:
            rows.append({"mode": mode, "selected_checkpoint": "", "error": str(e)})

    out_csv = run_root / f"clockrobust_summary_seed{seed}.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print("Clock-robustness summary written to:", out_csv)
    for row in rows:
        print(row)
    print("=" * 80)
    return str(out_csv)


# =========================================================
# Runner for all modes
# =========================================================
def run_all(args):
    script = Path(__file__).resolve()
    run_root = Path(args.run_root)
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    prime_modes = args.all_modes.split(",") if args.all_modes else ["exact_k", "jitter_k_10", "jitter_k_20"]
    prime_modes = [m.strip() for m in prime_modes if m.strip()]

    # 1) Shared DIM pretrain checkpoint.
    print("=" * 80)
    print("Step 1/2: Training shared DIM pretrain checkpoint")
    print("=" * 80)
    cmd_dim = [
        sys.executable, str(script),
        "--mode", "dim_only",
        "--train_path", args.train_path,
        "--run_root", args.run_root,
        "--seed", str(args.seed),
        "--dt_obs", str(args.dt_obs),
        "--dt_base", str(args.dt_base),
        "--Nsteps", str(args.warm_steps),
        "--warm_steps", str(args.warm_steps),
        "--width", str(args.width),
        "--print_every", str(args.print_every),
        "--tau_delay", str(args.tau_delay),
        "--dim_delay", str(args.dim_delay),
        "--delay_ch", str(args.delay_ch),
        "--L_hits_fullbatch", str(args.L_hits_fullbatch),
        "--BATCH_SIZE_STAGE2", str(args.BATCH_SIZE_STAGE2),
        "--w2_const", str(args.w2_const),
        "--cand_frac", str(args.cand_frac),
        "--q_speed", str(args.q_speed),
        "--M_landmarks", str(args.M_landmarks),
        "--L_sec", str(args.L_sec),
        "--eps_in_mult", str(args.eps_in_mult),
        "--eps_out_mult", str(args.eps_out_mult),
        "--tau_pair", str(args.tau_pair),
        "--stage1_batch", str(args.stage1_batch),
        "--lr1", str(args.lr1),
        "--lr2", str(args.lr2),
        "--alpha_pair_state", str(args.alpha_pair_state),
        "--alpha_pair_delay", str(args.alpha_pair_delay),
        "--w_marg_u0", str(args.w_marg_u0),
        "--w_marg_u1", str(args.w_marg_u1),
        "--w_defect", str(args.w_defect),
        "--blur_start", str(args.blur_start),
        "--blur_end", str(args.blur_end),
        "--stage2_steps_per_cycle", str(args.stage2_steps_per_cycle),
        "--stage1_steps_per_cycle", str(args.stage1_steps_per_cycle),
        "--clock_jitter_seed_offset", str(args.clock_jitter_seed_offset),
        "--clock_jitter_clip_sigma", str(args.clock_jitter_clip_sigma),
    ]
    if args.cpu:
        cmd_dim.append("--cpu")

    dim_log = log_dir / f"clockrobust_dim_only_seed{args.seed}.log"
    print("Running:", " ".join(cmd_dim))
    print("Log:", dim_log)
    with open(dim_log, "w") as f:
        proc = subprocess.run(cmd_dim, stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"DIM pretrain failed. Check log: {dim_log}")

    dim_folder = run_root / f"dim_only_seed{args.seed}"
    init_ckpt = find_selected_checkpoint(dim_folder)
    print("Shared DIM checkpoint:", init_ckpt)

    # 2) PRIME recurrence-clock variants.
    print("=" * 80)
    print("Step 2/2: Running PRIME recurrence-clock variants from the same DIM checkpoint")
    print("Modes:", prime_modes)
    print("=" * 80)

    for mode in prime_modes:
        log_path = log_dir / f"clockrobust_{mode}_seed{args.seed}.log"
        cmd = [
            sys.executable, str(script),
            "--mode", mode,
            "--train_path", args.train_path,
            "--run_root", args.run_root,
            "--seed", str(args.seed),
            "--dt_obs", str(args.dt_obs),
            "--dt_base", str(args.dt_base),
            "--Nsteps", str(args.refine_steps),
            "--warm_steps", "0",
            "--refine_steps", str(args.refine_steps),
            "--init_checkpoint", str(init_ckpt),
            "--width", str(args.width),
            "--print_every", str(args.print_every),
            "--tau_delay", str(args.tau_delay),
            "--dim_delay", str(args.dim_delay),
            "--delay_ch", str(args.delay_ch),
            "--L_hits_fullbatch", str(args.L_hits_fullbatch),
            "--BATCH_SIZE_STAGE2", str(args.BATCH_SIZE_STAGE2),
            "--w2_const", str(args.w2_const),
            "--cand_frac", str(args.cand_frac),
            "--q_speed", str(args.q_speed),
            "--M_landmarks", str(args.M_landmarks),
            "--L_sec", str(args.L_sec),
            "--eps_in_mult", str(args.eps_in_mult),
            "--eps_out_mult", str(args.eps_out_mult),
            "--tau_pair", str(args.tau_pair),
            "--stage1_batch", str(args.stage1_batch),
            "--lr1", str(args.lr1),
            "--lr2", str(args.lr2),
            "--alpha_pair_state", str(args.alpha_pair_state),
            "--alpha_pair_delay", str(args.alpha_pair_delay),
            "--w_marg_u0", str(args.w_marg_u0),
            "--w_marg_u1", str(args.w_marg_u1),
            "--w_defect", str(args.w_defect),
            "--blur_start", str(args.blur_start),
            "--blur_end", str(args.blur_end),
            "--stage2_steps_per_cycle", str(args.stage2_steps_per_cycle),
            "--stage1_steps_per_cycle", str(args.stage1_steps_per_cycle),
            "--exact_z_level", str(args.exact_z_level),
            "--exact_direction", str(args.exact_direction),
            "--clock_jitter_seed_offset", str(args.clock_jitter_seed_offset),
            "--clock_jitter_clip_sigma", str(args.clock_jitter_clip_sigma),
        ]
        if args.exact_z_ch is not None:
            cmd.extend(["--exact_z_ch", str(args.exact_z_ch)])
        if args.cpu:
            cmd.append("--cpu")

        print("=" * 80)
        print("Running:", " ".join(cmd))
        print("Log:", log_path)
        print("=" * 80)
        with open(log_path, "w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Run failed for mode={mode}. Check log: {log_path}")
        print(f"Finished mode={mode}. Log saved to {log_path}")

    print("All recurrence-clock robustness runs finished.")
    write_clock_robustness_summary(run_root, ["dim_only"] + prime_modes, args.seed)


# =========================================================
# CLI
# =========================================================
def build_arg_parser():
    p = argparse.ArgumentParser(description="Lorenz PRIME recurrence-clock robustness")
    p.add_argument("--all", action="store_true", help="Run shared DIM pretrain plus all PRIME recurrence-clock variants.")
    p.add_argument("--all_modes", type=str, default="exact_k,jitter_k_10,jitter_k_20",
                   help="Comma-separated PRIME recurrence-clock modes for --all.")
    p.add_argument("--mode", type=str, default="exact_k",
                   choices=sorted(LOSS_TERM_MODES))
    p.add_argument("--train_path", type=str, default=os.environ.get("TRAIN_DATA_PATH", ""))
    p.add_argument("--run_root", type=str, default="./lorenz_clock_robustness")
    p.add_argument("--init_checkpoint", type=str, default="")
    p.add_argument("--cpu", action="store_true")

    # General config
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--dt_obs", type=float, default=0.1)
    p.add_argument("--dt_base", type=float, default=0.1)
    p.add_argument("--width", type=int, default=100)

    # Section construction
    p.add_argument("--cand_frac", type=float, default=0.5)
    p.add_argument("--q_speed", type=float, default=0.3)
    p.add_argument("--M_landmarks", type=int, default=200)
    p.add_argument("--L_sec", type=int, default=5)
    p.add_argument("--tau_delay", type=int, default=1)
    p.add_argument("--dim_delay", type=int, default=5)
    p.add_argument("--delay_ch", type=int, default=0)
    p.add_argument("--eps_in_mult", type=float, default=0.35)
    p.add_argument("--eps_out_mult", type=float, default=1.8)
    p.add_argument("--clock_jitter_seed_offset", type=int, default=88017)
    p.add_argument("--clock_jitter_clip_sigma", type=float, default=2.0,
                   help="Clip Gaussian K_i jitter to +/- this many standard deviations.")

    # Exact-section options
    p.add_argument("--exact_z_ch", type=int, default=None,
                   help="Raw-data channel for Z. Default: 1 for 2D partial [X,Z], 2 for full [X,Y,Z].")
    p.add_argument("--exact_z_level", type=float, default=40.0)
    p.add_argument("--exact_direction", type=int, default=+1, choices=[-1, 0, 1])

    # Training config
    p.add_argument("--Nsteps", type=int, default=4000)
    p.add_argument("--warm_steps", type=int, default=3000)
    p.add_argument("--refine_steps", type=int, default=1000)
    p.add_argument("--tau_pair", type=int, default=2)
    p.add_argument("--stage1_batch", type=int, default=512)
    p.add_argument("--lr1", type=float, default=1e-3)
    p.add_argument("--lr2", type=float, default=1e-3)
    p.add_argument("--L_hits_fullbatch", type=int, default=32)
    p.add_argument("--BATCH_SIZE_STAGE2", type=int, default=16)
    p.add_argument("--alpha_pair_state", type=float, default=1.0)
    p.add_argument("--alpha_pair_delay", type=float, default=0.7)
    p.add_argument("--w_marg_u0", type=float, default=0.20)
    p.add_argument("--w_marg_u1", type=float, default=0.20)
    p.add_argument("--w_defect", type=float, default=0.0)
    p.add_argument("--blur_start", type=float, default=0.2)
    p.add_argument("--blur_end", type=float, default=0.1)
    p.add_argument("--stage2_steps_per_cycle", type=int, default=1)
    p.add_argument("--stage1_steps_per_cycle", type=int, default=1)
    p.add_argument("--w2_const", type=float, default=0.5)
    p.add_argument("--print_every", type=int, default=100)
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.all:
        run_all(args)
    else:
        train_one(args)
