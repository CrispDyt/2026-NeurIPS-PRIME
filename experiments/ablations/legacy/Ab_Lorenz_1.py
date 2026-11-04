#!/usr/bin/env python3
"""
Lorenz63 PRIME recurrence-clock ablation runner.

Runs one of four variants:
  dim_only  : DIM stage only
  fixed     : PRIME with a fixed median return interval K
  shuffled  : PRIME with shuffled empirical K_i intervals
  correct   : PRIME with the correct empirical K_i intervals

Run all four variants in separate fresh Python processes:
  python lorenz_prime_k_ablation.py --all --train_path /path/to/train_data.npy

Run one variant:
  python lorenz_prime_k_ablation.py --mode correct --train_path /path/to/train_data.npy

The script expects train_data to be a 2D array with shape (T, d). Supported formats:
  .npy, .npz, .pkl/.pickle, .pt/.pth
For pickle/dict files, it tries keys: train_data, train, training, data, X, Y.
"""

import argparse
import datetime
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
# Data loading
# =========================================================
def load_train_array(path: str) -> np.ndarray:
    if not path:
        raise ValueError(
            "No training data path was provided. Use --train_path /path/to/train_data.npy "
            "or set TRAIN_DATA_PATH. The file must contain a 2D array (T, d)."
        )

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
        obj = torch.load(p, map_location="cpu")
        arr = extract_array_from_object(obj)
    else:
        raise ValueError(f"Unsupported training data format: {suffix}")

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"train_data must be 2D (T, d), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("train_data contains non-finite values.")
    return arr


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
        # Try the first 2D array-like item.
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


# =========================================================
# Utilities
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_sinkhorn(blur: float):
    return SamplesLoss(loss="sinkhorn", p=2, blur=float(blur), backend="tensorized")


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
    returns:
      D_full: (T, dim), with lag blocks [y_ch(t), y_ch(t-tau), ..., y_ch(t-(dim-1)tau)]
      min_t
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


# =========================================================
# Main training routine for one mode
# =========================================================
def train_one(args):
    # -------------------------
    # Config
    # -------------------------
    dt_obs = args.dt_obs
    dt_base = args.dt_base
    sub = int(round(dt_obs / dt_base))
    if sub < 1:
        raise ValueError(f"sub={sub} < 1; check dt_obs={dt_obs}, dt_base={dt_base}")

    seed = args.seed
    cand_frac = args.cand_frac
    q_speed = args.q_speed
    M_landmarks = args.M_landmarks
    L_sec = args.L_sec
    tau_delay = args.tau_delay
    dim_delay = args.dim_delay
    eps_in_mult = args.eps_in_mult
    eps_out_mult = args.eps_out_mult
    DELAY_CH = args.delay_ch

    Nsteps = args.Nsteps
    warm_steps = args.warm_steps
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
    blur_start = args.blur_start
    blur_end = args.blur_end
    stage2_steps_per_cycle = args.stage2_steps_per_cycle
    stage1_steps_per_cycle = args.stage1_steps_per_cycle
    w2_const = args.w2_const

    mode = args.mode
    assert mode in {"dim_only", "correct", "fixed", "shuffled"}

    if mode == "dim_only":
        warm_steps = Nsteps
        stage2_steps_per_cycle = 0
        stage1_steps_per_cycle = 1

    K_SHUFFLE_SEED = args.K_shuffle_seed if args.K_shuffle_seed is not None else seed + 20260427
    K_FIXED_VALUE = args.K_fixed_value

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    set_seed(seed)

    out_dir = Path(args.run_root) / f"{mode}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_prefix = f"Implicit_K_{mode}_seed{seed}"

    print("device:", device)
    print("OUT_DIR:", out_dir)
    print("mode:", mode)

    # -------------------------
    # Load and scale data
    # -------------------------
    train = load_train_array(args.train_path)
    T, d = train.shape
    if DELAY_CH < 0 or DELAY_CH >= d:
        raise ValueError(f"delay_ch={DELAY_CH} out of range for d={d}")

    scaler = MaxAbsScaler().fit(train)
    y_scaled = scaler.transform(train).astype(np.float64)

    # -------------------------
    # Build implicit hits on TRAIN
    # -------------------------
    D_full, min_t = build_delay_full_onech(y_scaled, tau=tau_delay, dim=dim_delay, ch=DELAY_CH)
    U_full = np.concatenate([y_scaled, D_full], axis=1)

    T_eff = T - min_t
    if T_eff <= 10:
        raise ValueError(f"Too few effective samples: T={T}, min_t={min_t}")

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

    hits_raw, visit_lens = extract_hits_from_landmarks(Ufull_dw, C_dw, eps_in, eps_out, L_sec=L_sec)
    hits = hits_raw[hits_raw >= min_t]
    hits = np.sort(hits)
    if len(hits) < max(4, L_hits_fullbatch):
        raise RuntimeError(
            f"Too few hits ({len(hits)}). Need at least L_hits_fullbatch={L_hits_fullbatch}. "
            "Adjust section parameters."
        )

    taus = np.diff(hits).astype(np.int64)
    assert len(taus) == len(hits) - 1
    assert np.all(taus > 0)

    K_median = int(np.round(np.median(taus)))
    K_FIXED = max(1, K_median) if K_FIXED_VALUE is None else int(K_FIXED_VALUE)
    rng_k_ablate = np.random.default_rng(K_SHUFFLE_SEED)
    K_shuffled_global = rng_k_ablate.permutation(taus).astype(np.int64)

    print("=" * 60)
    print(f"Ablation K mode: {mode}")
    print(f"N_hits={len(hits)} | median K={K_median} | fixed K={K_FIXED}")
    print(
        f"K stats: mean={taus.mean():.3f}, std={taus.std():.3f}, "
        f"p10={np.quantile(taus, 0.1):.1f}, p90={np.quantile(taus, 0.9):.1f}"
    )
    print("=" * 60)

    XU_data = y_scaled[hits]
    DU_data = D_full[hits]
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
    opt1 = optim.Adam(net.parameters(), lr=lr1)
    opt2 = optim.Adam(net.parameters(), lr=lr2)

    def compute_L1(loss_sink):
        # 1) State-space graph measure
        Tmax = len(y_scaled) - tau_pair - 1
        idx = np.random.randint(0, Tmax, size=stage1_batch)

        x0 = torch.tensor(y_scaled[idx], dtype=torch.float32, device=device)
        xT_d = torch.tensor(y_scaled[idx + tau_pair], dtype=torch.float32, device=device)

        x = x0.clone()
        n_micro = int(tau_pair * sub)
        for _ in range(n_micro):
            x = x + float(dt_base) * net(x)
        xT_m = x

        x0w = whiten_x(x0)
        xTd = whiten_x(xT_d)
        xTm = whiten_x(xT_m)

        pair_d_state = torch.cat([x0w, xTd], dim=1)
        pair_m_state = torch.cat([x0w, xTm], dim=1)
        L_state = loss_sink(pair_m_state, pair_d_state)

        # 2) Delay-space graph measure, teacher-forced whole-trajectory style
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

    def get_model_schedule_for_block(k0: int, L_hits: int, schedule_mode: str):
        data_hit_times = hits[k0:k0 + L_hits].astype(np.int64)
        assert len(data_hit_times) == L_hits

        if schedule_mode == "correct":
            model_times = data_hit_times.copy()
        elif schedule_mode == "fixed":
            Kseq = np.full((L_hits - 1,), K_FIXED, dtype=np.int64)
            model_times = np.empty((L_hits,), dtype=np.int64)
            model_times[0] = data_hit_times[0]
            model_times[1:] = model_times[0] + np.cumsum(Kseq)
        elif schedule_mode == "shuffled":
            Kseq = K_shuffled_global[k0:k0 + L_hits - 1].astype(np.int64)
            if len(Kseq) != L_hits - 1:
                raise RuntimeError("Invalid shuffled K block length.")
            model_times = np.empty((L_hits,), dtype=np.int64)
            model_times[0] = data_hit_times[0]
            model_times[1:] = model_times[0] + np.cumsum(Kseq)
        else:
            raise ValueError(f"Unknown mode: {schedule_mode}")

        assert np.all(np.diff(model_times) > 0)
        return data_hit_times, model_times

    def build_induced_block(k0, L_hits, loss_sink):
        data_hit_times, model_times = get_model_schedule_for_block(k0, L_hits, mode)

        X_blk_d = torch.tensor(XU_data[k0:k0 + L_hits], dtype=torch.float32, device=device)
        D_blk_d = torch.tensor(DU_data[k0:k0 + L_hits], dtype=torch.float32, device=device)

        hist_len = int((dim_delay - 1) * tau_delay)
        t0_model = int(model_times[0])
        t_start = t0_model - hist_len
        if t_start < 0:
            raise RuntimeError(f"t_start={t_start} < 0")

        t_end = int(model_times[-1])
        total_obs_steps = int(t_end - t_start)
        if total_obs_steps < 0:
            raise RuntimeError("Negative total_obs_steps")

        x = torch.tensor(y_scaled[t_start], dtype=torch.float32, device=device)
        model_time_map = {int(t): j for j, t in enumerate(model_times.tolist())}
        x_hits_m = [None] * L_hits
        x_hist = [x.clone()]

        for n in range(total_obs_steps + 1):
            t_abs = t_start + n
            if t_abs in model_time_map:
                j = model_time_map[t_abs]
                x_hits_m[j] = x
                x = x.detach()

            if n == total_obs_steps:
                break

            for _ in range(sub):
                x = x + float(dt_base) * net(x)
            x_hist.append(x.clone())

        if any(v is None for v in x_hits_m):
            raise RuntimeError(f"Some model hits were not recorded. mode={mode}, k0={k0}")

        X_blk_m = torch.stack(x_hits_m, dim=0)

        Dm_list = []
        for t_abs in model_times:
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
        L2_marg = w_marg_u0 * L2_marg0 + w_marg_u1 * L2_marg1

        defect = torch.mean((Um[:, :d] - Ud[:, :d]) ** 2)
        L2_raw = L2_graph + L2_marg + w_defect * defect
        L2_norm = L2_raw / float(L_hits)
        return L2_norm, L2_raw

    # -------------------------
    # BEST trackers
    # -------------------------
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

    print("=" * 60)
    print("Start training (TRAIN split only)")
    print(f"Ablation K mode={mode}")
    print(f"dt_obs={dt_obs} | dt_base={dt_base} | sub={sub}")
    print(f"Nsteps={Nsteps} | warm={warm_steps} | N_hits={len(hits)} | L_hits={L_hits_fullbatch}")
    print(f"Stage2 objective: L_total = L1 + w2_const * L2_norm | w2_const={w2_const}")
    print("=" * 60)

    net.train()
    for step in range(Nsteps):
        frac = step / max(1, Nsteps - 1)
        blur = blur_start + (blur_end - blur_start) * frac
        loss_sink = make_sinkhorn(blur)

        if step < warm_steps:
            stage = "stage1"
        else:
            cyc = stage2_steps_per_cycle + stage1_steps_per_cycle
            kk = (step - warm_steps) % cyc
            stage = "stage2" if kk < stage2_steps_per_cycle else "stage1"

        opt = opt1 if stage == "stage1" else opt2
        opt.zero_grad(set_to_none=True)

        L1 = compute_L1(loss_sink)

        if stage == "stage1":
            L2_norm = torch.tensor(0.0, device=device)
            L2_raw = torch.tensor(0.0, device=device)
            Ltot = L1
        else:
            L_hits = min(L_hits_fullbatch, len(hits))
            k0max = len(hits) - L_hits
            l2n_list = []
            l2r_list = []
            for _ in range(BATCH_SIZE_STAGE2):
                k0 = np.random.randint(0, k0max + 1)
                l2n, l2r = build_induced_block(k0, L_hits, loss_sink)
                l2n_list.append(l2n)
                l2r_list.append(l2r)

            L2_norm = torch.stack(l2n_list).mean()
            L2_raw = torch.stack(l2r_list).mean()
            Ltot = L1 + float(w2_const) * L2_norm

        Ltot.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        ltot_v = float(Ltot.detach().cpu())
        l1_v = float(L1.detach().cpu())
        l2n_v = float(L2_norm.detach().cpu())
        l2r_v = float(L2_raw.detach().cpu())

        loss_tot_hist.append(ltot_v)
        loss_L1_hist.append(l1_v)
        loss_L2norm_hist.append(l2n_v)
        loss_L2raw_hist.append(l2r_v)
        stage_hist.append(stage)

        if stage == "stage2":
            if ltot_v < best_total:
                best_total = ltot_v
                best_total_it = step
                best_total_sd = cpu_state_dict(net)
                best_total_L1 = l1_v
                best_total_L2norm = l2n_v
                best_total_L2raw = l2r_v
                print(
                    f"  --> [New BEST_TOTAL] Iter {step} | L_total={best_total:.6f} "
                    f"| L1={best_total_L1:.6f} | L2_norm={best_total_L2norm:.6f} "
                    f"| L2_raw={best_total_L2raw:.6f}"
                )

            if l2r_v < best_L2raw:
                best_L2raw = l2r_v
                best_L2_it = step
                best_L2_sd = cpu_state_dict(net)
                best_L2_at_best = ltot_v
                best_L1_at_best = l1_v
                best_L2norm_at_best = l2n_v
                print(
                    f"  --> [New BEST_L2]    Iter {step} | L2_raw={best_L2raw:.6f} "
                    f"| L_total={best_L2_at_best:.6f} | L1={best_L1_at_best:.6f} "
                    f"| L2_norm={best_L2norm_at_best:.6f}"
                )

        if step % args.print_every == 0:
            print(
                f"Iter {step:5d} [{stage}] | Ltot={ltot_v:.6f} | L1={l1_v:.6f} "
                f"| L2_norm={l2n_v:.6f} | L2_raw={l2r_v:.6f} | blur={blur:.3f}"
            )

    print("Training done.")

    # -------------------------
    # Save checkpoints
    # -------------------------
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path_final = out_dir / f"{save_prefix}_FINAL_{stamp}.p"
    path_best = out_dir / f"{save_prefix}_BEST_{stamp}.p"
    path_best_l2 = out_dir / f"{save_prefix}_BEST_L2_{stamp}.p"
    path_selected = out_dir / f"{save_prefix}_SELECTED_{stamp}.p"

    config = dict(
        seed=int(seed),
        train_path=str(Path(args.train_path).expanduser()),
        dt_obs=float(dt_obs),
        dt_base=float(dt_base),
        sub=int(sub),
        tau_pair=int(tau_pair),
        tau_delay=int(tau_delay),
        dim_delay=int(dim_delay),
        delay_ch=int(DELAY_CH),
        L_sec=int(L_sec),
        M_landmarks=int(M_landmarks),
        cand_frac=float(cand_frac),
        q_speed=float(q_speed),
        eps_in_mult=float(eps_in_mult),
        eps_out_mult=float(eps_out_mult),
        L_hits_fullbatch=int(L_hits_fullbatch),
        BATCH_SIZE_STAGE2=int(BATCH_SIZE_STAGE2),
        lr1=float(lr1),
        lr2=float(lr2),
        w2_const=float(w2_const),
        alpha_pair_state=float(alpha_pair_state),
        alpha_pair_delay=float(alpha_pair_delay),
        w_marg_u0=float(w_marg_u0),
        w_marg_u1=float(w_marg_u1),
        w_defect=float(w_defect),
        blur_start=float(blur_start),
        blur_end=float(blur_end),
        warm_steps=int(warm_steps),
        Nsteps=int(Nsteps),
        min_t=int(min_t),
        eps_in=float(eps_in),
        eps_out=float(eps_out),
        nn_med=float(nn_med),
        N_hits=int(len(hits)),
        ablation_K_mode=str(mode),
        K_fixed=int(K_FIXED),
        K_median=int(K_median),
        K_shuffle_seed=int(K_SHUFFLE_SEED),
        width=int(args.width),
    )

    base_payload = dict(
        config=config,
        model=dict(transformer=scaler, state_dict=None),
        stats=dict(
            hits=hits,
            taus_steps=taus,
            visit_lens=visit_lens,
            C_dw=C_dw,
            u_mean_c=u_mean_c,
            u_std_c=u_std_c,
            u_mean=u_mean,
            u_std=u_std,
            K_shuffled_global=K_shuffled_global,
        ),
        train_hist=dict(
            stage=np.array(stage_hist, dtype=object),
            L_total=np.array(loss_tot_hist, dtype=np.float64),
            L1=np.array(loss_L1_hist, dtype=np.float64),
            L2_norm=np.array(loss_L2norm_hist, dtype=np.float64),
            L2_raw=np.array(loss_L2raw_hist, dtype=np.float64),
        ),
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
        print("No BEST_TOTAL saved (maybe never entered stage2).")

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
        print("No BEST_L2 saved (maybe never entered stage2).")

    # Unified SELECTED checkpoint for later evaluation.
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

    # Save loss plots to files, no GUI blocking.
    plot_history(out_dir / f"{save_prefix}_loss_total_{stamp}.png",
                 [loss_tot_hist, loss_L1_hist], ["L_total", "L1"], "Loss")
    plot_history(out_dir / f"{save_prefix}_loss_L2norm_{stamp}.png",
                 [loss_L2norm_hist], ["L2_norm"], "Stage2 loss norm")
    plot_history(out_dir / f"{save_prefix}_loss_L2raw_{stamp}.png",
                 [loss_L2raw_hist], ["L2_raw"], "Stage2 loss raw")

    return str(path_selected)


# =========================================================
# Run-all wrapper
# =========================================================
def run_all(args):
    modes = ["dim_only", "fixed", "shuffled", "correct"]
    log_dir = Path(args.run_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    common = [
        "--train_path", str(args.train_path),
        "--run_root", str(args.run_root),
        "--seed", str(args.seed),
        "--dt_obs", str(args.dt_obs),
        "--dt_base", str(args.dt_base),
        "--Nsteps", str(args.Nsteps),
        "--warm_steps", str(args.warm_steps),
        "--width", str(args.width),
        "--print_every", str(args.print_every),
    ]

    # Add CPU flag if requested.
    if args.cpu:
        common.append("--cpu")

    print("Running all ablation modes in fresh subprocesses:", modes)
    for mode in modes:
        log_path = log_dir / f"{mode}_seed{args.seed}.log"
        cmd = [sys.executable, str(script_path), "--mode", mode] + common
        print("=" * 80)
        print("Running:", " ".join(cmd))
        print("Log:", log_path)
        print("=" * 80)
        with open(log_path, "w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Run failed for mode={mode}. Check log: {log_path}")
        print(f"Finished mode={mode}")
    print("All ablation runs finished.")


# =========================================================
# CLI
# =========================================================
def build_argparser():
    p = argparse.ArgumentParser(description="Lorenz PRIME recurrence-clock ablation trainer")
    p.add_argument("--train_path", type=str, default=os.environ.get("TRAIN_DATA_PATH", ""),
                   help="Path to train_data array: .npy, .npz, .pkl/.pickle/.p, .pt/.pth")
    p.add_argument("--run_root", type=str, default="./lorenz_ablation_K")
    p.add_argument("--mode", type=str, default=os.environ.get("ABLATION_K_MODE", "correct"),
                   choices=["dim_only", "fixed", "shuffled", "correct"])
    p.add_argument("--all", action="store_true", help="Run all four modes in fresh subprocesses.")

    # Basic config
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--width", type=int, default=100)
    p.add_argument("--print_every", type=int, default=100)

    # Time/integration
    p.add_argument("--dt_obs", type=float, default=0.1)
    p.add_argument("--dt_base", type=float, default=0.1)

    # Section/delay config
    p.add_argument("--cand_frac", type=float, default=0.5)
    p.add_argument("--q_speed", type=float, default=0.3)
    p.add_argument("--M_landmarks", type=int, default=200)
    p.add_argument("--L_sec", type=int, default=5)
    p.add_argument("--tau_delay", type=int, default=1)
    p.add_argument("--dim_delay", type=int, default=5)
    p.add_argument("--eps_in_mult", type=float, default=0.35)
    p.add_argument("--eps_out_mult", type=float, default=1.8)
    p.add_argument("--delay_ch", type=int, default=0)

    # Training config
    p.add_argument("--Nsteps", type=int, default=4000)
    p.add_argument("--warm_steps", type=int, default=3000)
    p.add_argument("--tau_pair", type=int, default=2)
    p.add_argument("--stage1_batch", type=int, default=512)
    p.add_argument("--lr1", type=float, default=1e-3)
    p.add_argument("--L_hits_fullbatch", type=int, default=32)
    p.add_argument("--BATCH_SIZE_STAGE2", type=int, default=16)
    p.add_argument("--lr2", type=float, default=1e-3)
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

    # Ablation config
    p.add_argument("--K_shuffle_seed", type=int, default=None)
    p.add_argument("--K_fixed_value", type=int, default=None)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.all:
        run_all(args)
    else:
        train_one(args)
