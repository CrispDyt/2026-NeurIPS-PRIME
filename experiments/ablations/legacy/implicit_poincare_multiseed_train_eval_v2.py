"""
Multi-seed implicit Poincare-section parameter ablation.

Default:
  q_speed ∈ {0.10, 0.20, 0.30, 0.40}
  training seeds ∈ {123, 456, 789, 2024, 3407}

For each q_speed and seed, this script:
  1) trains the original implicit-section PRIME model;
  2) selects the Implicit_BEST_TOTAL checkpoint by training objective;
  3) evaluates it on fixed 30 rollout starts;
  4) reports the appendix metrics aggregated over training seeds:

       NMSE_500     : mean ± std over training seeds
       Attractor_W2 : mean ± std over training seeds
       PSD_logL1    : mean ± std over training seeds

Data loading matches your setup:
    train_data = np.load('../train_data.npy')
    test_data  = np.load('../test_data.npy')
    dt_obs  = 0.1
    dt_base = 0.1

Run:
    python implicit_poincare_multiseed_train_eval.py

Quick smoke test:
    python implicit_poincare_multiseed_train_eval.py --quick
"""


import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import csv
import json
import math
import pickle
import argparse
import datetime
import random
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as tnn
from torch import optim

from geomloss import SamplesLoss
from sklearn.preprocessing import MaxAbsScaler

try:
    from scipy.signal import welch
    HAS_WELCH = True
except Exception as _welch_err:
    HAS_WELCH = False
    WELCH_ERR = repr(_welch_err)


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    # data / time
    train_path: str = "train_data.npy"
    test_path: str = "test_data.npy"
    dt_obs: float = 0.1
    dt_base: float = 0.1

    # reproducibility
    seed: int = 123

    # implicit-section parameters
    cand_frac: float = 0.5
    q_speed: float = 0.3
    M_landmarks: int = 200
    L_sec: int = 5
    tau_delay: int = 1
    dim_delay: int = 5
    eps_in_mult: float = 0.35
    eps_out_mult: float = 1.8
    delay_ch: int = 0

    # training config
    Nsteps: int = 4000
    warm_steps: int = 3000
    tau_pair: int = 2
    stage1_batch: int = 512
    lr1: float = 1e-3
    L_hits_fullbatch: int = 32
    BATCH_SIZE_STAGE2: int = 16
    lr2: float = 1e-3

    # Stage-2 induced map loss weights
    alpha_pair_state: float = 1.0
    alpha_pair_delay: float = 0.7
    w_marg_u0: float = 0.20
    w_marg_u1: float = 0.20
    w_defect: float = 0.0

    # Sinkhorn blur schedule
    blur_start: float = 0.2
    blur_end: float = 0.1

    # stage cycling after warmup
    stage2_steps_per_cycle: int = 1
    stage1_steps_per_cycle: int = 1

    # total objective
    w2_const: float = 0.5

    # network
    width: int = 100

    # single-start diagnostic eval kept for compatibility
    eval_windows: Tuple[int, ...] = (20, 50, 100, 200, 400, 500)
    eval_rollout_len: int = 500

    # final 30-start test metrics for appendix ablation
    eval_n_rollouts: int = 30
    eval_struct_len: int = 500
    eval_start_mode: str = "linspace"
    eval_start_seed: int = 42
    psd_nfft: int = 2048
    w2_blur: float = 0.05
    w2_max_points: int = 1500

    # safety
    min_hits_for_training: int = 8

    # output
    out_dir: str = "./implicit_poincare_ablation_runs"
    save_prefix: str = "Implicit"


# ============================================================
# Basic utils
# ============================================================

def set_seed(s: int, deterministic: bool = True) -> None:
    random.seed(int(s))
    np.random.seed(int(s))
    torch.manual_seed(int(s))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(s))

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def make_sinkhorn(blur: float) -> SamplesLoss:
    return SamplesLoss(loss="sinkhorn", p=2, blur=float(blur), backend="tensorized")


def build_net(d: int, width: int = 100) -> tnn.Module:
    return tnn.Sequential(
        tnn.Linear(d, width), tnn.Tanh(),
        tnn.Linear(width, width), tnn.Tanh(),
        tnn.Linear(width, width), tnn.Tanh(),
        tnn.Linear(width, d),
    )


def cpu_state_dict(net: tnn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def ensure_2d_array(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with shape (T, d), got shape={x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN or Inf values.")
    return x.astype(np.float64, copy=False)


def pstats(a: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a)
    if len(a) == 0:
        return dict(median=np.nan, mean=np.nan, std=np.nan, p10=np.nan, p90=np.nan)
    return dict(
        median=float(np.median(a)),
        mean=float(np.mean(a)),
        std=float(np.std(a)),
        p10=float(np.quantile(a, 0.1)),
        p90=float(np.quantile(a, 0.9)),
    )


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return None
        return y
    except Exception:
        return None


# ============================================================
# Delay embedding and implicit-hit extraction
# ============================================================

def build_delay_full_onech(y_scaled: np.ndarray, tau: int, dim: int, ch: int) -> Tuple[np.ndarray, int]:
    """
    y_scaled: (T, d)
    returns D_full: (T, dim), with lag blocks
        [y_ch(t), y_ch(t-tau), ..., y_ch(t-(dim-1)tau)]
    and min_t.
    """
    T, d = y_scaled.shape
    if ch < 0 or ch >= d:
        raise ValueError(f"delay_ch={ch} is out of range for d={d}")
    min_t = int((dim - 1) * tau)
    if T <= min_t + 2:
        raise ValueError(f"Training trajectory too short for dim_delay={dim}, tau_delay={tau}: T={T}, min_t={min_t}")

    D = np.zeros((T, dim), dtype=np.float64)
    s = y_scaled[:, ch]
    for t in range(min_t, T):
        D[t] = np.array([s[t - i * tau] for i in range(dim)], dtype=np.float64)
    return D, min_t


def farthest_point_sampling(X: np.ndarray, M: int, start: int = 0) -> np.ndarray:
    N = int(X.shape[0])
    M = int(min(int(M), N))
    if N <= 0:
        raise ValueError("farthest_point_sampling received an empty array.")
    if M <= 0:
        raise ValueError("M must be positive.")

    idx = np.zeros((M,), dtype=np.int64)
    idx[0] = int(np.clip(start, 0, N - 1))
    dist2 = np.sum((X - X[idx[0]]) ** 2, axis=1)
    for m in range(1, M):
        j = int(np.argmax(dist2))
        idx[m] = j
        d2_new = np.sum((X - X[j]) ** 2, axis=1)
        dist2 = np.minimum(dist2, d2_new)
    return idx


def extract_hits_from_landmarks(
    U_dw: np.ndarray,
    C_dw: np.ndarray,
    eps_in: float,
    eps_out: float,
    L_sec: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hysteresis + min-gap hit extraction.
    Returns:
        hits: visit entry times
        visit_lens: duration before leaving eps_out neighborhood
    """
    T = int(U_dw.shape[0])
    hits: List[int] = []
    visit_lens: List[int] = []
    last = -10 ** 12
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


def build_delay_data_tensor_onech(
    y_np: np.ndarray,
    dim_delay: int,
    tau_delay: int,
    ch: int,
    device: str,
) -> torch.Tensor:
    """
    Single-channel delay embedding for Stage-1 DIM-style loss.
    Returns shape: (1, L, dim_delay)
    """
    Y = torch.tensor(y_np, dtype=torch.float32, device=device)
    s = Y[:, ch]
    T = s.shape[0]
    L = T - tau_delay * (dim_delay - 1)
    delayed = torch.zeros((1, L, dim_delay), dtype=torch.float32, device=device)

    for i in range(dim_delay):
        start = dim_delay * tau_delay - (i + 1) * tau_delay
        delayed[0, :, i] = s[start:start + L]
    return delayed


def construct_implicit_section(
    y_scaled: np.ndarray,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    """
    Build implicit Poincare hits on TRAIN.
    This is the main object of ablation: q_speed changes the speed-thresholded
    candidate pool before FPS landmark selection.
    """
    D_full, min_t = build_delay_full_onech(
        y_scaled,
        tau=cfg.tau_delay,
        dim=cfg.dim_delay,
        ch=cfg.delay_ch,
    )
    U_full = np.concatenate([y_scaled, D_full], axis=1)

    T = y_scaled.shape[0]
    T_eff = T - min_t
    idx_eff = np.arange(min_t, T)

    speed = np.linalg.norm(y_scaled[1:] - y_scaled[:-1], axis=1)
    speed = np.concatenate([[speed[0]], speed], axis=0)

    thr = float(np.quantile(speed[idx_eff], cfg.q_speed))
    pool = idx_eff[speed[idx_eff] >= thr]

    cand_target = int(round(cfg.cand_frac * T_eff))
    cand_target = max(1, cand_target)

    if len(pool) <= 0:
        raise RuntimeError(
            f"Empty candidate pool: q_speed={cfg.q_speed}, speed threshold={thr}."
        )

    if len(pool) >= cand_target:
        idx_cand = np.random.choice(pool, size=cand_target, replace=False)
    else:
        idx_cand = pool.copy()

    idx_cand = np.sort(idx_cand)
    Ucand = U_full[idx_cand]

    if len(Ucand) < 2:
        raise RuntimeError(
            f"Too few candidates after speed filtering: N_candidates={len(Ucand)}."
        )

    u_mean_c = Ucand.mean(axis=0)
    u_std_c = Ucand.std(axis=0) + 1e-6
    Ucand_dw = (Ucand - u_mean_c[None, :]) / u_std_c[None, :]
    Ufull_dw = (U_full - u_mean_c[None, :]) / u_std_c[None, :]

    start0 = int(np.random.randint(0, len(idx_cand)))
    fps_idx_local = farthest_point_sampling(Ucand_dw, M=cfg.M_landmarks, start=start0)
    C_dw = Ucand_dw[fps_idx_local]

    if len(C_dw) < 2:
        raise RuntimeError(f"Too few landmarks: N_landmarks={len(C_dw)}.")

    nn = []
    for i in range(len(C_dw)):
        d2 = np.sum((C_dw - C_dw[i]) ** 2, axis=1)
        d2[i] = np.inf
        nn.append(np.sqrt(np.min(d2)))
    nn = np.asarray(nn, dtype=np.float64)
    nn_med = float(np.median(nn))

    eps_in = float(cfg.eps_in_mult * nn_med)
    eps_out = float(cfg.eps_out_mult * eps_in)

    hits_raw, visit_lens = extract_hits_from_landmarks(
        Ufull_dw, C_dw, eps_in, eps_out, L_sec=cfg.L_sec
    )
    hits = hits_raw[hits_raw >= min_t]
    hits = np.sort(hits)
    taus = np.diff(hits).astype(np.float64)

    XU_data = y_scaled[hits]
    DU_data = D_full[hits]
    U_hits_data = np.concatenate([XU_data, DU_data], axis=1) if len(hits) else np.zeros((0, U_full.shape[1]))

    stats = dict(
        q_speed=float(cfg.q_speed),
        speed_thr=float(thr),
        T=int(T),
        min_t=int(min_t),
        T_eff=int(T_eff),
        pool_size=int(len(pool)),
        pool_frac=float(len(pool) / max(1, T_eff)),
        cand_target=int(cand_target),
        cand_size=int(len(idx_cand)),
        cand_frac_real=float(len(idx_cand) / max(1, T_eff)),
        M_landmarks_requested=int(cfg.M_landmarks),
        M_landmarks_actual=int(len(C_dw)),
        nn_med=float(nn_med),
        eps_in=float(eps_in),
        eps_out=float(eps_out),
        N_hits=int(len(hits)),
        hit_frac=float(len(hits) / max(1, T_eff)),
        N_taus=int(len(taus)),
        tau_mean_steps=safe_float(np.mean(taus)) if len(taus) else None,
        tau_std_steps=safe_float(np.std(taus)) if len(taus) else None,
        tau_median_steps=safe_float(np.median(taus)) if len(taus) else None,
        tau_p10_steps=safe_float(np.quantile(taus, 0.1)) if len(taus) else None,
        tau_p90_steps=safe_float(np.quantile(taus, 0.9)) if len(taus) else None,
        visit_mean_steps=safe_float(np.mean(visit_lens)) if len(visit_lens) else None,
        visit_median_steps=safe_float(np.median(visit_lens)) if len(visit_lens) else None,
    )

    return dict(
        D_full=D_full,
        min_t=min_t,
        U_full=U_full,
        speed=speed,
        idx_eff=idx_eff,
        pool=pool,
        idx_cand=idx_cand,
        u_mean_c=u_mean_c,
        u_std_c=u_std_c,
        Ucand_dw=Ucand_dw,
        Ufull_dw=Ufull_dw,
        C_dw=C_dw,
        nn=nn,
        nn_med=nn_med,
        eps_in=eps_in,
        eps_out=eps_out,
        hits_raw=hits_raw,
        hits=hits,
        taus=taus,
        visit_lens=visit_lens,
        XU_data=XU_data,
        DU_data=DU_data,
        U_hits_data=U_hits_data,
        stats=stats,
    )


# ============================================================
# Evaluation helpers
# ============================================================

@torch.no_grad()
def rollout_scaled(
    net: tnn.Module,
    x0_np: np.ndarray,
    n_obs_steps: int,
    dt_base: float,
    sub: int,
    device: str,
) -> np.ndarray:
    net.eval()
    x = torch.tensor(x0_np, dtype=torch.float32, device=device)
    xs = [x.detach().cpu().numpy().astype(np.float64)]
    for _ in range(int(n_obs_steps)):
        for _ in range(int(sub)):
            x = x + float(dt_base) * net(x)
        xs.append(x.detach().cpu().numpy().astype(np.float64))
    return np.stack(xs, axis=0)


def nmse(pred: np.ndarray, true: np.ndarray, denom_ref: Optional[np.ndarray] = None) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mse = float(np.mean((pred - true) ** 2))
    if denom_ref is None:
        denom_ref = true
    denom = float(np.mean((denom_ref - denom_ref.mean(axis=0, keepdims=True)) ** 2)) + 1e-12
    return mse / denom


def evaluate_on_test(
    net: tnn.Module,
    scaler: MaxAbsScaler,
    train_scaled: np.ndarray,
    test_data: np.ndarray,
    cfg: TrainConfig,
    device: str,
) -> Dict[str, float]:
    if test_data is None or len(test_data) < 3:
        return {}

    test_scaled = scaler.transform(test_data).astype(np.float64)
    max_len = min(int(cfg.eval_rollout_len), len(test_scaled) - 1)
    if max_len <= 1:
        return {}

    sub = int(round(cfg.dt_obs / cfg.dt_base))
    pred = rollout_scaled(
        net=net,
        x0_np=test_scaled[0],
        n_obs_steps=max_len,
        dt_base=cfg.dt_base,
        sub=sub,
        device=device,
    )
    gt = test_scaled[: max_len + 1]

    out: Dict[str, float] = {}
    denom_ref = test_scaled
    out["NMSE_full"] = nmse(pred, gt, denom_ref=denom_ref)
    out["MSE_full"] = float(np.mean((pred - gt) ** 2))

    for w in cfg.eval_windows:
        w_eff = min(int(w), max_len)
        if w_eff >= 2:
            out[f"NMSE_{w}"] = nmse(pred[: w_eff + 1], gt[: w_eff + 1], denom_ref=denom_ref)
            out[f"MSE_{w}"] = float(np.mean((pred[: w_eff + 1] - gt[: w_eff + 1]) ** 2))
    return out


def save_rollout_plot(
    net: tnn.Module,
    scaler: MaxAbsScaler,
    test_data: np.ndarray,
    cfg: TrainConfig,
    device: str,
    path: str,
) -> None:
    if test_data is None or len(test_data) < 3:
        return
    test_scaled = scaler.transform(test_data).astype(np.float64)
    max_len = min(int(cfg.eval_rollout_len), len(test_scaled) - 1)
    sub = int(round(cfg.dt_obs / cfg.dt_base))
    pred = rollout_scaled(net, test_scaled[0], max_len, cfg.dt_base, sub, device)
    gt = test_scaled[: max_len + 1]

    d = gt.shape[1]
    plt.figure(figsize=(6.2, 5.2), dpi=160)
    if d >= 2:
        plt.plot(gt[:, 0], gt[:, 1], lw=1.4, label="GT")
        plt.plot(pred[:, 0], pred[:, 1], lw=1.1, alpha=0.75, label="prediction")
        plt.xlabel("x0 scaled")
        plt.ylabel("x1 scaled")
        plt.title("Test rollout from test[0]")
    else:
        t = np.arange(len(gt)) * cfg.dt_obs
        plt.plot(t, gt[:, 0], lw=1.4, label="GT")
        plt.plot(t, pred[:, 0], lw=1.1, alpha=0.75, label="prediction")
        plt.xlabel("time")
        plt.ylabel("x0 scaled")
        plt.title("Test rollout from test[0]")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



# ============================================================
# Final 30-start metrics used for appendix ablation
# ============================================================

def nmse_value_raw(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    L = min(len(gt), len(pred))
    gt = gt[:L]
    pred = pred[:L]
    mse = float(np.mean((gt - pred) ** 2))
    denom = float(np.mean(gt ** 2)) + 1e-12
    return mse / denom


@torch.no_grad()
def rollout_raw_from_x0(
    net: tnn.Module,
    scaler: MaxAbsScaler,
    x0_raw: np.ndarray,
    K_obs: int,
    dt_base: float,
    sub: int,
    device: str,
) -> np.ndarray:
    """Roll out in scaled space and return raw-space prediction."""
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


def sinkhorn_w2_cloud(A: np.ndarray, B: np.ndarray, blur: float, device: str, max_points: int = 1500) -> float:
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

    try:
        XA = torch.tensor(A, dtype=torch.float32, device=device)
        XB = torch.tensor(B, dtype=torch.float32, device=device)
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=float(blur), backend="tensorized")
        return float(loss(XA, XB).detach().cpu())
    except Exception as e:
        print(f"[warn] Sinkhorn W2 failed; using sliced-W2 fallback. Error: {repr(e)}")
        return sliced_w2_cloud(A, B, n_proj=256, seed=123)


def choose_eval_starts(T: int, horizon: int, n_rollouts: int, mode: str = "linspace", seed: int = 42) -> np.ndarray:
    valid_count = int(T) - int(horizon) - 1
    if valid_count <= 0:
        raise ValueError(f"test_data too short: T={T}, horizon={horizon}")

    n = min(int(n_rollouts), valid_count)
    if mode == "random":
        rng = np.random.default_rng(int(seed))
        starts = np.sort(rng.choice(valid_count, size=n, replace=False))
    else:
        starts = np.linspace(0, valid_count - 1, n).astype(int)

    return np.unique(starts)


def evaluate_30start_metrics(
    net: tnn.Module,
    scaler: MaxAbsScaler,
    test_data: np.ndarray,
    cfg: TrainConfig,
    device: str,
) -> Dict[str, Any]:
    """
    Model-level metrics over fixed rollout starts:
      - NMSE_500_mean/std over rollout starts
      - Attractor_W2_mean over rollout starts
      - PSD_logL1_mean over rollout starts

    These model-level means are later aggregated over training seeds.
    """
    test = ensure_2d_array(test_data, "test_data")
    sub = int(round(cfg.dt_obs / cfg.dt_base))

    horizon = int(cfg.eval_struct_len)
    horizon = min(horizon, len(test) - 2)
    if horizon < 500:
        print(f"[warn] eval_struct_len={cfg.eval_struct_len} shrunk to {horizon}; NMSE_500 uses horizon={horizon}.")

    starts = choose_eval_starts(
        T=len(test),
        horizon=horizon,
        n_rollouts=cfg.eval_n_rollouts,
        mode=cfg.eval_start_mode,
        seed=cfg.eval_start_seed,
    )

    nmse_500_vals = []
    w2_vals = []
    psd_vals = []

    for s in starts:
        s = int(s)
        gt = test[s:s + horizon + 1]
        pred = rollout_raw_from_x0(
            net=net,
            scaler=scaler,
            x0_raw=test[s],
            K_obs=horizon,
            dt_base=cfg.dt_base,
            sub=sub,
            device=device,
        )

        k = min(500, horizon)
        nmse_500_vals.append(nmse_value_raw(gt[:k + 1], pred[:k + 1]))
        w2_vals.append(sinkhorn_w2_cloud(gt, pred, blur=cfg.w2_blur, device=device, max_points=cfg.w2_max_points))
        psd_vals.append(psd_logl1_distance(gt, pred, dt=cfg.dt_obs, nfft=cfg.psd_nfft, channels=None))

    def finite_array(vals):
        arr = np.asarray(vals, dtype=np.float64)
        return arr[np.isfinite(arr)]

    nmse_arr = finite_array(nmse_500_vals)
    w2_arr = finite_array(w2_vals)
    psd_arr = finite_array(psd_vals)

    return dict(
        n_rollouts_actual=int(len(starts)),
        eval_starts=starts.tolist(),
        eval_horizon=int(horizon),
        NMSE_500_mean=float(np.mean(nmse_arr)) if len(nmse_arr) else np.nan,
        NMSE_500_std_rollouts=float(np.std(nmse_arr)) if len(nmse_arr) else np.nan,
        Attractor_W2_mean=float(np.mean(w2_arr)) if len(w2_arr) else np.nan,
        Attractor_W2_std_rollouts=float(np.std(w2_arr)) if len(w2_arr) else np.nan,
        PSD_logL1_mean=float(np.mean(psd_arr)) if len(psd_arr) else np.nan,
        PSD_logL1_std_rollouts=float(np.std(psd_arr)) if len(psd_arr) else np.nan,
    )


# ============================================================
# Training one ablation case
# ============================================================

def train_one_case(
    case_name: str,
    train_data: np.ndarray,
    test_data: np.ndarray,
    cfg: TrainConfig,
    device: str,
    case_dir: str,
) -> Dict[str, Any]:
    os.makedirs(case_dir, exist_ok=True)
    set_seed(cfg.seed)

    sub = int(round(cfg.dt_obs / cfg.dt_base))
    if sub <= 0:
        raise ValueError(f"Invalid sub={sub}. Check dt_obs={cfg.dt_obs}, dt_base={cfg.dt_base}")
    if abs(sub * cfg.dt_base - cfg.dt_obs) > 1e-10:
        raise ValueError(
            f"dt_obs/dt_base must be close to an integer. Got dt_obs={cfg.dt_obs}, dt_base={cfg.dt_base}, sub={sub}."
        )

    train = ensure_2d_array(train_data, "train_data")
    test = ensure_2d_array(test_data, "test_data")
    T, d = train.shape

    scaler = MaxAbsScaler().fit(train)
    y_scaled = scaler.transform(train).astype(np.float64)

    # -------------------------
    # Build implicit hits on TRAIN
    # -------------------------
    section = construct_implicit_section(y_scaled, cfg)
    hits = section["hits"]

    hit_stats_path = os.path.join(case_dir, "implicit_section_stats.json")
    with open(hit_stats_path, "w") as f:
        json.dump(section["stats"], f, indent=2)

    print("------------------------------------------------------------")
    print(f"Case: {case_name}")
    print(
        f"q_speed={cfg.q_speed:.3f} | pool={section['stats']['pool_size']} "
        f"({section['stats']['pool_frac']:.3f}) | cand={section['stats']['cand_size']} | "
        f"landmarks={section['stats']['M_landmarks_actual']} | hits={len(hits)} | "
        f"tau_mean={section['stats']['tau_mean_steps']}"
    )

    if len(hits) < int(cfg.min_hits_for_training):
        status = "skipped_too_few_hits"
        summary = dict(case=case_name, status=status, **section["stats"])
        with open(os.path.join(case_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[SKIP] {case_name}: only {len(hits)} hits; min_hits_for_training={cfg.min_hits_for_training}")
        return summary

    D_full = section["D_full"]
    min_t = int(section["min_t"])
    XU_data = section["XU_data"]
    DU_data = section["DU_data"]
    U_hits_data = section["U_hits_data"]
    taus = section["taus"]
    visit_lens = section["visit_lens"]

    # -------------------------
    # Tensors and whitening
    # -------------------------
    Y_all_t = torch.tensor(y_scaled, dtype=torch.float32, device=device)

    u_mean = U_hits_data.mean(axis=0)
    u_std = U_hits_data.std(axis=0) + 1e-6
    u_mean_t = torch.tensor(u_mean, dtype=torch.float32, device=device)
    u_std_t = torch.tensor(u_std, dtype=torch.float32, device=device)

    def whiten_u(U: torch.Tensor) -> torch.Tensor:
        return (U - u_mean_t[None, :]) / u_std_t[None, :]

    x_mean = y_scaled.mean(axis=0)
    x_std = y_scaled.std(axis=0) + 1e-6
    x_mean_t = torch.tensor(x_mean, dtype=torch.float32, device=device)
    x_std_t = torch.tensor(x_std, dtype=torch.float32, device=device)

    def whiten_x(X: torch.Tensor) -> torch.Tensor:
        return (X - x_mean_t[None, :]) / x_std_t[None, :]

    y_delayed_data = build_delay_data_tensor_onech(
        y_scaled,
        dim_delay=cfg.dim_delay,
        tau_delay=cfg.tau_delay,
        ch=cfg.delay_ch,
        device=device,
    )

    # -------------------------
    # Model + optimizers
    # -------------------------
    net = build_net(d, width=cfg.width).to(device)
    opt1 = optim.Adam(net.parameters(), lr=cfg.lr1)
    opt2 = optim.Adam(net.parameters(), lr=cfg.lr2)

    # =========================================================
    # Stage-1 L1: IM(state graph) + IM(delay graph) + glue
    # =========================================================
    def compute_L1(loss_sink: SamplesLoss) -> torch.Tensor:
        # ---------- 1) IM: state-space graph measure ----------
        Tmax = len(y_scaled) - cfg.tau_pair - 1
        if Tmax <= 2:
            raise ValueError(f"Training data too short for tau_pair={cfg.tau_pair}")
        idx = np.random.randint(0, Tmax, size=cfg.stage1_batch)

        x0 = torch.tensor(y_scaled[idx], dtype=torch.float32, device=device)
        xT_d = torch.tensor(y_scaled[idx + cfg.tau_pair], dtype=torch.float32, device=device)

        x = x0.clone()
        n_micro = int(cfg.tau_pair * sub)
        for _ in range(n_micro):
            x = x + float(cfg.dt_base) * net(x)
        xT_m = x

        x0w = whiten_x(x0)
        xTd = whiten_x(xT_d)
        xTm = whiten_x(xT_m)

        pair_d_state = torch.cat([x0w, xTd], dim=1)
        pair_m_state = torch.cat([x0w, xTm], dim=1)
        L_state = loss_sink(pair_m_state, pair_d_state)

        # ---------- 2) DIM: delay-space graph measure ----------
        Kfine = int(cfg.tau_delay * cfg.dim_delay * sub)
        yy = Y_all_t.clone()
        sols = []

        for _ in range(Kfine):
            sols.append(yy)
            for _ in range(sub):
                yy = yy + float(cfg.dt_base) * net(yy)

        sol = torch.stack(sols, dim=0)       # (Kfine, T, d)
        sol = sol[:, min_t:, :]              # (Kfine, L_delay, d)
        step_fine = int(cfg.tau_delay * sub)

        sol_ch = sol[:, :, cfg.delay_ch]                 # (Kfine, L_delay)
        sol_ch = sol_ch.permute(1, 0).contiguous()       # (L_delay, Kfine)
        DIM_model = torch.flip(sol_ch[:, ::step_fine], dims=[1])
        DIM_model = DIM_model.unsqueeze(0).contiguous()  # (1, L_delay, dim_delay)

        y_delay = y_delayed_data
        L_delay = loss_sink(DIM_model, y_delay).sum()

        # Glue term only anchors the initial delay slice. Kept exactly as in your script.
        glue_defect = torch.mean((DIM_model[:, 0, :-1] - y_delay[:, 0, :-1]) ** 2)
        L1 = L_state + L_delay + glue_defect
        return L1

    # =========================================================
    # Stage-2 induced return-block loss over implicit hits
    # =========================================================
    def build_induced_block(k0: int, L_hits: int, loss_sink: SamplesLoss) -> Tuple[torch.Tensor, torch.Tensor]:
        hit_times = hits[k0:k0 + L_hits].astype(np.int64)

        X_blk_d = torch.tensor(XU_data[k0:k0 + L_hits], dtype=torch.float32, device=device)
        D_blk_d = torch.tensor(DU_data[k0:k0 + L_hits], dtype=torch.float32, device=device)

        t0 = int(hit_times[0])
        t_start = t0 - int((cfg.dim_delay - 1) * cfg.tau_delay)
        t_end = int(hit_times[-1])
        total_obs_steps = int(t_end - t_start)

        x = torch.tensor(y_scaled[t_start], dtype=torch.float32, device=device)

        hit_map = {int(t): j for j, t in enumerate(hit_times.tolist())}
        x_hits_m: List[Optional[torch.Tensor]] = [None] * L_hits
        x_hist: List[torch.Tensor] = [x.clone()]

        for n in range(total_obs_steps + 1):
            t_abs = t_start + n

            if t_abs in hit_map:
                x_hits_m[hit_map[t_abs]] = x
                x = x.detach()  # multi-shoot cut

            if n == total_obs_steps:
                break

            for _ in range(sub):
                x = x + float(cfg.dt_base) * net(x)
            x_hist.append(x.clone())

        if any(v is None for v in x_hits_m):
            raise RuntimeError("Internal error: some hit states were not recorded.")

        X_blk_m = torch.stack([v for v in x_hits_m if v is not None], dim=0)

        Dm_list = []
        for t_abs in hit_times:
            idx0 = int(t_abs - t_start)
            lag_vals = []
            for i in range(cfg.dim_delay):
                lag_vals.append(x_hist[idx0 - i * cfg.tau_delay][cfg.delay_ch])
            Dm_list.append(torch.stack(lag_vals, dim=0))
        D_blk_m = torch.stack(Dm_list, dim=0)

        U_blk_m = torch.cat([X_blk_m, D_blk_m], dim=1)
        U_blk_d = torch.cat([X_blk_d, D_blk_d], dim=1)

        Um = whiten_u(U_blk_m)
        Ud = whiten_u(U_blk_d)

        xm, dm = Um[:, :d], Um[:, d:]
        xd, dd = Ud[:, :d], Ud[:, d:]
        u_m = torch.cat([cfg.alpha_pair_state * xm, cfg.alpha_pair_delay * dm], dim=1)
        u_d = torch.cat([cfg.alpha_pair_state * xd, cfg.alpha_pair_delay * dd], dim=1)

        u0_m, u1_m = u_m[:-1], u_m[1:]
        u0_d, u1_d = u_d[:-1], u_d[1:]
        pair_m = torch.cat([u0_m, u1_m], dim=1)
        pair_d = torch.cat([u0_d, u1_d], dim=1)

        L2_graph = loss_sink(pair_m, pair_d)
        L2_marg0 = loss_sink(u0_m, u0_d)
        L2_marg1 = loss_sink(u1_m, u1_d)
        L2_marg = cfg.w_marg_u0 * L2_marg0 + cfg.w_marg_u1 * L2_marg1

        defect = torch.mean((Um[:, :d] - Ud[:, :d]) ** 2)

        L2_raw = L2_graph + L2_marg + cfg.w_defect * defect
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

    # -------------------------
    # Training loop
    # -------------------------
    loss_tot_hist: List[float] = []
    loss_L1_hist: List[float] = []
    loss_L2norm_hist: List[float] = []
    loss_L2raw_hist: List[float] = []
    stage_hist: List[str] = []

    print("============================================================")
    print("Start training")
    print(f"case={case_name}")
    print(f"device={device} | dt_obs={cfg.dt_obs} | dt_base={cfg.dt_base} | sub={sub}")
    print(f"Nsteps={cfg.Nsteps} | warm={cfg.warm_steps} | N_hits={len(hits)} | L_hits={cfg.L_hits_fullbatch}")
    print(f"Stage2 objective: L_total = L1 + w2_const * L2_norm | w2_const={cfg.w2_const}")
    print("============================================================")

    net.train()
    for step in range(cfg.Nsteps):
        t = step / max(1, cfg.Nsteps - 1)
        blur = cfg.blur_start + (cfg.blur_end - cfg.blur_start) * t
        loss_sink = make_sinkhorn(blur)

        if step < cfg.warm_steps:
            stage = "stage1"
        else:
            cyc = cfg.stage2_steps_per_cycle + cfg.stage1_steps_per_cycle
            k = (step - cfg.warm_steps) % cyc
            stage = "stage2" if k < cfg.stage2_steps_per_cycle else "stage1"

        opt = opt1 if stage == "stage1" else opt2
        opt.zero_grad(set_to_none=True)

        L1 = compute_L1(loss_sink)

        if stage == "stage1":
            L2_norm = torch.tensor(0.0, device=device)
            L2_raw = torch.tensor(0.0, device=device)
            Ltot = L1
        else:
            L_hits = min(int(cfg.L_hits_fullbatch), len(hits))
            L_hits = max(3, L_hits)
            k0max = len(hits) - L_hits
            if k0max < 0:
                raise RuntimeError(f"Not enough hits for Stage-2: N_hits={len(hits)}, L_hits={L_hits}")

            l2n_list = []
            l2r_list = []
            for _ in range(int(cfg.BATCH_SIZE_STAGE2)):
                k0 = np.random.randint(0, k0max + 1)
                l2n, l2r = build_induced_block(k0, L_hits, loss_sink)
                l2n_list.append(l2n)
                l2r_list.append(l2r)

            L2_norm = torch.stack(l2n_list).mean()
            L2_raw = torch.stack(l2r_list).mean()
            Ltot = L1 + float(cfg.w2_const) * L2_norm

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
                    f"  --> [New BEST_TOTAL] Iter {step} | "
                    f"L_total={best_total:.6f} | L1={best_total_L1:.6f} | "
                    f"L2_norm={best_total_L2norm:.6f} | L2_raw={best_total_L2raw:.6f}"
                )

            if l2r_v < best_L2raw:
                best_L2raw = l2r_v
                best_L2_it = step
                best_L2_sd = cpu_state_dict(net)
                best_L2_at_best = ltot_v
                best_L1_at_best = l1_v
                best_L2norm_at_best = l2n_v
                print(
                    f"  --> [New BEST_L2]    Iter {step} | "
                    f"L2_raw={best_L2raw:.6f} | L_total={best_L2_at_best:.6f} | "
                    f"L1={best_L1_at_best:.6f} | L2_norm={best_L2norm_at_best:.6f}"
                )

        if step % 100 == 0:
            print(
                f"Iter {step:5d} [{stage}] | Ltot={ltot_v:.6f} | L1={l1_v:.6f} | "
                f"L2_norm={l2n_v:.6f} | L2_raw={l2r_v:.6f} | blur={blur:.3f}"
            )

    print(f"Training done for case={case_name}.")

    # -------------------------
    # Evaluation
    # -------------------------
    eval_final = evaluate_on_test(net, scaler, y_scaled, test, cfg, device)

    eval_best_total: Dict[str, float] = {}
    eval30_best_total: Dict[str, Any] = {}
    if best_total_sd is not None:
        net_best = build_net(d, width=cfg.width).to(device)
        net_best.load_state_dict(best_total_sd)
        net_best.eval()

        # Old diagnostic eval: single-start scaled NMSE.
        eval_best_total = evaluate_on_test(net_best, scaler, y_scaled, test, cfg, device)

        # Final appendix eval: fixed 30-start raw-space metrics.
        eval30_best_total = evaluate_30start_metrics(net_best, scaler, test, cfg, device)

        save_rollout_plot(
            net_best,
            scaler,
            test,
            cfg,
            device,
            os.path.join(case_dir, "rollout_test0_BEST_TOTAL.png"),
        )
    else:
        save_rollout_plot(
            net,
            scaler,
            test,
            cfg,
            device,
            os.path.join(case_dir, "rollout_test0_FINAL.png"),
        )

    # -------------------------
    # Save models and histories
    # -------------------------
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path_final = os.path.join(case_dir, f"{cfg.save_prefix}_FINAL_{stamp}.p")
    path_best = os.path.join(case_dir, f"{cfg.save_prefix}_BEST_TOTAL_{stamp}.p")
    path_best_l2 = os.path.join(case_dir, f"{cfg.save_prefix}_BEST_L2_{stamp}.p")

    config_dict = asdict(cfg)
    config_dict.update(
        dict(
            case_name=case_name,
            sub=int(sub),
            d=int(d),
            T_train=int(T),
            min_t=int(min_t),
            eps_in=float(section["eps_in"]),
            eps_out=float(section["eps_out"]),
            nn_med=float(section["nn_med"]),
            N_hits=int(len(hits)),
        )
    )

    base_payload = dict(
        config=config_dict,
        model=dict(transformer=scaler, state_dict=None),
        stats=dict(
            hits=hits,
            taus_steps=taus,
            visit_lens=visit_lens,
            C_dw=section["C_dw"],
            u_mean_c=section["u_mean_c"],
            u_std_c=section["u_std_c"],
            u_mean=u_mean,
            u_std=u_std,
            implicit_section_stats=section["stats"],
        ),
        train_hist=dict(
            stage=np.array(stage_hist, dtype=object),
            L_total=np.array(loss_tot_hist, dtype=np.float64),
            L1=np.array(loss_L1_hist, dtype=np.float64),
            L2_norm=np.array(loss_L2norm_hist, dtype=np.float64),
            L2_raw=np.array(loss_L2raw_hist, dtype=np.float64),
        ),
        eval=dict(final=eval_final, best_total=eval_best_total, best_total_30start=eval30_best_total),
    )

    payload_final = dict(base_payload)
    payload_final["model"] = dict(transformer=scaler, state_dict=cpu_state_dict(net))
    with open(path_final, "wb") as f:
        pickle.dump(payload_final, f)
    print("Saved FINAL:", path_final)

    if best_total_sd is not None:
        payload_best = dict(base_payload)
        payload_best["config"] = dict(config_dict)
        payload_best["config"].update(
            dict(
                best_total_iter=int(best_total_it),
                best_total=float(best_total),
                best_total_L1=float(best_total_L1),
                best_total_L2_norm=float(best_total_L2norm),
                best_total_L2_raw=float(best_total_L2raw),
            )
        )
        payload_best["model"] = dict(transformer=scaler, state_dict=best_total_sd)
        with open(path_best, "wb") as f:
            pickle.dump(payload_best, f)
        print(f"Saved BEST_TOTAL: {path_best}  (L_total={best_total:.6f} @ iter {best_total_it})")
    else:
        print("No BEST_TOTAL saved. Maybe Nsteps never entered Stage-2.")

    if best_L2_sd is not None:
        payload_best_l2 = dict(base_payload)
        payload_best_l2["config"] = dict(config_dict)
        payload_best_l2["config"].update(
            dict(
                best_l2_iter=int(best_L2_it),
                best_l2_L2_raw=float(best_L2raw),
                best_l2_L_total=float(best_L2_at_best),
                best_l2_L1=float(best_L1_at_best),
                best_l2_L2_norm=float(best_L2norm_at_best),
            )
        )
        payload_best_l2["model"] = dict(transformer=scaler, state_dict=best_L2_sd)
        with open(path_best_l2, "wb") as f:
            pickle.dump(payload_best_l2, f)
        print(f"Saved BEST_L2: {path_best_l2}  (L2_raw={best_L2raw:.6f} @ iter {best_L2_it})")
    else:
        print("No BEST_L2 saved. Maybe Nsteps never entered Stage-2.")

    # -------------------------
    # Plots
    # -------------------------
    plt.figure(figsize=(10, 3), dpi=160)
    plt.plot(loss_tot_hist, label="L_total")
    plt.plot(loss_L1_hist, label="L1")
    plt.legend(frameon=False)
    plt.title(f"Loss: {case_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "loss_total_L1.png"))
    plt.close()

    plt.figure(figsize=(10, 3), dpi=160)
    plt.plot(loss_L2norm_hist, label="L2_norm")
    plt.legend(frameon=False)
    plt.title(f"Stage-2 loss norm: {case_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "loss_L2_norm.png"))
    plt.close()

    plt.figure(figsize=(10, 3), dpi=160)
    plt.plot(loss_L2raw_hist, label="L2_raw")
    plt.legend(frameon=False)
    plt.title(f"Stage-2 loss raw: {case_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "loss_L2_raw.png"))
    plt.close()

    # hit interval diagnostic
    if len(taus) > 0:
        plt.figure(figsize=(7, 4), dpi=160)
        plt.hist(taus, bins=40)
        plt.xlabel("inter-hit interval / obs steps")
        plt.ylabel("count")
        plt.title(f"Implicit return intervals: {case_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, "hit_interval_hist.png"))
        plt.close()

    summary = dict(
        case=case_name,
        status="ok",
        **section["stats"],
        best_total_iter=int(best_total_it),
        best_total=safe_float(best_total),
        best_total_L1=safe_float(best_total_L1),
        best_total_L2_norm=safe_float(best_total_L2norm),
        best_total_L2_raw=safe_float(best_total_L2raw),
        best_l2_iter=int(best_L2_it),
        best_l2_raw=safe_float(best_L2raw),
    )

    for k, v in eval_final.items():
        summary[f"FINAL_{k}"] = safe_float(v)
    for k, v in eval_best_total.items():
        summary[f"BEST_TOTAL_{k}"] = safe_float(v)
    for k, v in eval30_best_total.items():
        if k == "eval_starts":
            summary["EVAL30_eval_starts"] = v
        else:
            summary[f"EVAL30_{k}"] = safe_float(v)

    with open(os.path.join(case_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================
# Ablation runner
# ============================================================

def parse_q_grid(s: str) -> List[float]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if len(vals) == 0:
        raise ValueError("q_grid is empty.")
    return vals


def write_summary_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if len(rows) == 0:
        return
    keys = []
    priority = [
        "case", "status", "q_speed", "seed", "pool_size", "pool_frac", "cand_size",
        "M_landmarks_actual", "eps_in", "eps_out", "N_hits", "tau_mean_steps",
        "tau_std_steps", "tau_median_steps", "best_total", "best_total_L2_raw",
        "best_l2_raw", "BEST_TOTAL_NMSE_100", "BEST_TOTAL_NMSE_200",
        "BEST_TOTAL_NMSE_400", "BEST_TOTAL_NMSE_500", "BEST_TOTAL_NMSE_full", "FINAL_NMSE_full",
        "EVAL30_n_rollouts_actual", "EVAL30_eval_horizon", "EVAL30_NMSE_500_mean",
        "EVAL30_NMSE_500_std_rollouts", "EVAL30_Attractor_W2_mean",
        "EVAL30_PSD_logL1_mean",
    ]
    for k in priority:
        if any(k in r for r in rows):
            keys.append(k)
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, None) for k in keys})


def plot_ablation_summary(rows: List[Dict[str, Any]], out_dir: str) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    if len(ok) == 0:
        return

    q = np.array([float(r["q_speed"]) for r in ok], dtype=np.float64)
    order = np.argsort(q)
    q = q[order]

    def arr(key: str) -> np.ndarray:
        vals = []
        for r in np.array(ok, dtype=object)[order]:
            v = r.get(key, np.nan)
            vals.append(np.nan if v is None else float(v))
        return np.asarray(vals, dtype=np.float64)

    # q_speed vs hit count
    plt.figure(figsize=(6.4, 4.0), dpi=160)
    plt.plot(q, arr("N_hits"), marker="o")
    plt.xlabel("q_speed")
    plt.ylabel("N_hits")
    plt.title("Implicit section: speed threshold vs hit count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_qspeed_vs_hits.png"))
    plt.close()

    # q_speed vs inter-hit mean
    plt.figure(figsize=(6.4, 4.0), dpi=160)
    plt.plot(q, arr("tau_mean_steps"), marker="o")
    plt.xlabel("q_speed")
    plt.ylabel("mean inter-hit interval / obs steps")
    plt.title("Implicit section: speed threshold vs return time")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_qspeed_vs_tau_mean.png"))
    plt.close()

    # q_speed vs NMSE if available
    for key, title in [
        ("BEST_TOTAL_NMSE_100", "Test rollout NMSE_100"),
        ("BEST_TOTAL_NMSE_400", "Test rollout NMSE_400"),
        ("BEST_TOTAL_NMSE_full", "Test rollout NMSE_full"),
        ("best_total_L2_raw", "Best Stage-2 raw loss"),
    ]:
        y = arr(key)
        if np.all(~np.isfinite(y)):
            continue
        plt.figure(figsize=(6.4, 4.0), dpi=160)
        plt.plot(q, y, marker="o")
        plt.xlabel("q_speed")
        plt.ylabel(key)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"ablation_qspeed_vs_{key}.png"))
        plt.close()



def mean_std_over_seeds(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return float(np.mean(arr)), 0.0
    # For appendix reporting across training seeds, use sample std.
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def aggregate_over_training_seeds(rows: List[Dict[str, Any]], root: str) -> List[Dict[str, Any]]:
    ok = [r for r in rows if r.get("status") == "ok"]
    groups: Dict[float, List[Dict[str, Any]]] = {}

    for r in ok:
        q = float(r["q_speed"])
        groups.setdefault(q, []).append(r)

    agg_rows: List[Dict[str, Any]] = []

    for q in sorted(groups.keys()):
        rs = groups[q]

        nmse_mu, nmse_sd = mean_std_over_seeds([r.get("EVAL30_NMSE_500_mean", np.nan) for r in rs])
        w2_mu, w2_sd = mean_std_over_seeds([r.get("EVAL30_Attractor_W2_mean", np.nan) for r in rs])
        psd_mu, psd_sd = mean_std_over_seeds([r.get("EVAL30_PSD_logL1_mean", np.nan) for r in rs])
        nh_mu, nh_sd = mean_std_over_seeds([r.get("N_hits", np.nan) for r in rs])
        tau_mu, tau_sd = mean_std_over_seeds([r.get("tau_mean_steps", np.nan) for r in rs])

        agg_rows.append(dict(
            q_speed=float(q),
            n_seeds_ok=int(len(rs)),
            seeds=",".join(str(r.get("seed", "")) for r in rs),
            N_hits_mean=nh_mu,
            N_hits_std=nh_sd,
            tau_mean_steps_mean=tau_mu,
            tau_mean_steps_std=tau_sd,
            NMSE_500_mean_over_seeds=nmse_mu,
            NMSE_500_std_over_seeds=nmse_sd,
            Attractor_W2_mean_over_seeds=w2_mu,
            Attractor_W2_std_over_seeds=w2_sd,
            PSD_logL1_mean_over_seeds=psd_mu,
            PSD_logL1_std_over_seeds=psd_sd,
            NMSE_500=f"{nmse_mu:.4g} ± {nmse_sd:.4g}" if np.isfinite(nmse_mu) else "nan",
            Attractor_W2=f"{w2_mu:.4g} ± {w2_sd:.4g}" if np.isfinite(w2_mu) else "nan",
            PSD_logL1=f"{psd_mu:.4g} ± {psd_sd:.4g}" if np.isfinite(psd_mu) else "nan",
        ))

    csv_path = os.path.join(root, "aggregate_over_training_seeds.csv")
    keys = [
        "q_speed", "n_seeds_ok", "seeds",
        "N_hits_mean", "N_hits_std", "tau_mean_steps_mean", "tau_mean_steps_std",
        "NMSE_500_mean_over_seeds", "NMSE_500_std_over_seeds",
        "Attractor_W2_mean_over_seeds", "Attractor_W2_std_over_seeds",
        "PSD_logL1_mean_over_seeds", "PSD_logL1_std_over_seeds",
        "NMSE_500", "Attractor_W2", "PSD_logL1",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in agg_rows:
            writer.writerow({k: r.get(k, "") for k in keys})

    md_path = os.path.join(root, "aggregate_over_training_seeds.md")
    with open(md_path, "w") as f:
        f.write("# Multi-seed implicit-section ablation\n\n")
        f.write("Metrics are aggregated over training seeds. Each seed-level value is first evaluated using fixed 30 rollout starts.\n\n")
        f.write("| q_speed | ok seeds | N_hits | tau_mean | NMSE_500 | Attractor_W2 | PSD_logL1 |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in agg_rows:
            f.write(
                f"| {r['q_speed']:.2f} | {r['n_seeds_ok']} | "
                f"{r['N_hits_mean']:.3g} ± {r['N_hits_std']:.3g} | "
                f"{r['tau_mean_steps_mean']:.3g} ± {r['tau_mean_steps_std']:.3g} | "
                f"{r['NMSE_500']} | {r['Attractor_W2']} | {r['PSD_logL1']} |\n"
            )

    if len(agg_rows) > 0:
        qs = np.array([r["q_speed"] for r in agg_rows], dtype=np.float64)
        for mean_key, std_key, ylabel, fname in [
            ("NMSE_500_mean_over_seeds", "NMSE_500_std_over_seeds", "NMSE_500", "aggregate_qspeed_vs_NMSE_500.png"),
            ("Attractor_W2_mean_over_seeds", "Attractor_W2_std_over_seeds", "Attractor_W2", "aggregate_qspeed_vs_Attractor_W2.png"),
            ("PSD_logL1_mean_over_seeds", "PSD_logL1_std_over_seeds", "PSD_logL1", "aggregate_qspeed_vs_PSD_logL1.png"),
        ]:
            y = np.array([r[mean_key] for r in agg_rows], dtype=np.float64)
            e = np.array([r[std_key] for r in agg_rows], dtype=np.float64)
            plt.figure(figsize=(6.4, 4.0), dpi=160)
            plt.errorbar(qs, y, yerr=e, marker="o", capsize=4)
            plt.xlabel("q_speed")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel}: mean ± std over training seeds")
            plt.tight_layout()
            plt.savefig(os.path.join(root, fname))
            plt.close()

    return agg_rows


def print_aggregate_table(agg_rows: List[Dict[str, Any]]) -> None:
    print("\n==================== FINAL APPENDIX TABLE: mean ± std over training seeds ====================")
    cols = ["q_speed", "n_seeds_ok", "NMSE_500", "Attractor_W2", "PSD_logL1"]
    colw = 20
    header = " | ".join([f"{c:>{colw}s}" for c in cols])
    print(header)
    print("-" * len(header))

    for r in agg_rows:
        vals = [
            f"{r['q_speed']:.2f}",
            str(r["n_seeds_ok"]),
            r["NMSE_500"],
            r["Attractor_W2"],
            r["PSD_logL1"],
        ]
        print(" | ".join([f"{v:>{colw}s}" for v in vals]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="train_data.npy")
    parser.add_argument("--test", type=str, default="test_data.npy")
    parser.add_argument("--out_dir", type=str, default="./implicit_poincare_ablation_runs")
    parser.add_argument("--q_grid", type=str, default="0.10,0.20,0.30,0.40")
    parser.add_argument("--seeds", type=str, default="13532,10000,100,200,3407")
    parser.add_argument("--seed", type=int, default=123, help="Legacy single seed; ignored unless --seeds is empty.")
    parser.add_argument("--dt_obs", type=float, default=0.1)
    parser.add_argument("--dt_base", type=float, default=0.1)
    parser.add_argument("--quick", action="store_true", help="Use short training for smoke test.")
    parser.add_argument("--Nsteps", type=int, default=None)
    parser.add_argument("--warm_steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu. Default auto-detects.")
    parser.add_argument("--eval_n_rollouts", type=int, default=30)
    parser.add_argument("--eval_struct_len", type=int, default=500)
    parser.add_argument("--eval_start_mode", type=str, default="linspace", choices=["linspace", "random"])
    parser.add_argument("--eval_start_seed", type=int, default=42)
    parser.add_argument("--psd_nfft", type=int, default=2048)
    parser.add_argument("--w2_blur", type=float, default=0.05)
    parser.add_argument("--w2_max_points", type=int, default=1500)
    args = parser.parse_args()

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_data = np.load(args.train)
    test_data = np.load(args.test)
    train_data = ensure_2d_array(train_data, "train_data")
    test_data = ensure_2d_array(test_data, "test_data")

    seed_values = parse_q_grid(args.seeds) if args.seeds.strip() else [int(args.seed)]
    seed_values = [int(s) for s in seed_values]

    base_cfg = TrainConfig(
        train_path=args.train,
        test_path=args.test,
        out_dir=args.out_dir,
        seed=int(seed_values[0]),
        dt_obs=float(args.dt_obs),
        dt_base=float(args.dt_base),
        eval_n_rollouts=int(args.eval_n_rollouts),
        eval_struct_len=int(args.eval_struct_len),
        eval_start_mode=str(args.eval_start_mode),
        eval_start_seed=int(args.eval_start_seed),
        psd_nfft=int(args.psd_nfft),
        w2_blur=float(args.w2_blur),
        w2_max_points=int(args.w2_max_points),
    )

    if args.quick:
        base_cfg = replace(
            base_cfg,
            Nsteps=300,
            warm_steps=200,
            stage1_batch=256,
            BATCH_SIZE_STAGE2=4,
            L_hits_fullbatch=16,
            eval_rollout_len=200,
            eval_struct_len=200,
            eval_n_rollouts=5,
        )
        # quick mode is only a smoke test, not the appendix experiment
        q_values_quick_note = True

    if args.Nsteps is not None:
        base_cfg = replace(base_cfg, Nsteps=int(args.Nsteps))
    if args.warm_steps is not None:
        base_cfg = replace(base_cfg, warm_steps=int(args.warm_steps))

    os.makedirs(base_cfg.out_dir, exist_ok=True)

    q_values = parse_q_grid(args.q_grid)
    rows: List[Dict[str, Any]] = []

    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(base_cfg.out_dir, f"ablation_{run_stamp}")
    os.makedirs(root, exist_ok=True)

    with open(os.path.join(root, "base_config.json"), "w") as f:
        json.dump(asdict(base_cfg), f, indent=2)

    print("============================================================")
    print("Implicit Poincare Section Ablation")
    print(f"train={args.train} shape={train_data.shape}")
    print(f"test ={args.test} shape={test_data.shape}")
    print(f"q_grid={q_values}")
    print(f"seeds={seed_values}")
    print(f"eval: n_rollouts={base_cfg.eval_n_rollouts}, struct_len={base_cfg.eval_struct_len}, start_mode={base_cfg.eval_start_mode}")
    print(f"root={root}")
    print("============================================================")

    for q in q_values:
        for seed in seed_values:
            cfg = replace(base_cfg, q_speed=float(q), seed=int(seed))
            q_name = f"qspeed_{q:.2f}".replace(".", "p")
            case_name = f"{q_name}_seed_{int(seed)}"
            case_dir = os.path.join(root, q_name, f"seed_{int(seed)}")
            try:
                row = train_one_case(case_name, train_data, test_data, cfg, device, case_dir)
                row["seed"] = int(seed)
            except Exception as e:
                print(f"[ERROR] case={case_name}: {repr(e)}")
                row = dict(case=case_name, status="error", q_speed=float(q), seed=int(seed), error=repr(e))
                os.makedirs(case_dir, exist_ok=True)
                with open(os.path.join(case_dir, "error.json"), "w") as f:
                    json.dump(row, f, indent=2)
            rows.append(row)
            write_summary_csv(rows, os.path.join(root, "per_seed_results_partial.csv"))
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    summary_csv = os.path.join(root, "per_seed_results.csv")
    write_summary_csv(rows, summary_csv)
    with open(os.path.join(root, "per_seed_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    agg_rows = aggregate_over_training_seeds(rows, root)
    print_aggregate_table(agg_rows)

    print("============================================================")
    print("Multi-seed ablation finished.")
    print("Per-seed CSV  :", summary_csv)
    print("Aggregate CSV :", os.path.join(root, "aggregate_over_training_seeds.csv"))
    print("Aggregate MD  :", os.path.join(root, "aggregate_over_training_seeds.md"))
    print("Root folder   :", root)
    print("============================================================")


if __name__ == "__main__":
    main()
