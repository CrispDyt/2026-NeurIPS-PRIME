#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computational-cost profiler for PRIME objective-controlled training.

Supported benchmarks:
  - Lorenz63 partial observation: 2D observations (u0,u2)
  - KS partial observation: 5D sensor observations

Supported objectives:
  - MSE
  - DIM
  - DIM+MSE
  - PRIME

The script trains each requested objective from scratch and reports:
  setup time, training time, total wall-clock time, average iteration time,
  peak CUDA allocated/reserved memory, CPU RSS, parameter count, and final loss.

Example:
  python profile_computational_cost.py \
      --benchmark all \
      --lorenz_train train_data.npy \
      --ks_train ks_train.npy \
      --methods mse,dim,dim_mse,prime \
      --seeds 123 \
      --out_dir cost_results

Quick smoke test:
  python profile_computational_cost.py --benchmark lorenz --lorenz_train train_data.npy --quick
"""

import argparse
import csv
import json
import os
import random
import time
import resource
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from sklearn.preprocessing import MaxAbsScaler
from geomloss import SamplesLoss


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()


def peak_cuda_mb(device: torch.device) -> Tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return float(alloc), float(reserved)


def current_rss_mb() -> float:
    # Linux ru_maxrss is in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def count_params(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def make_sinkhorn(blur: float):
    return SamplesLoss(loss="sinkhorn", p=2, blur=float(blur), backend="tensorized")


def build_net(d: int, width: int = 100) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, d),
    )


def load_array(path: str) -> np.ndarray:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Cannot find array: {p}")
    if p.suffix.lower() == ".npy":
        arr = np.load(p, allow_pickle=True)
    elif p.suffix.lower() == ".npz":
        obj = np.load(p, allow_pickle=True)
        for k in ["train", "train_data", "data", "X", "Y"]:
            if k in obj:
                arr = obj[k]
                break
        else:
            keys = list(obj.keys())
            if len(keys) == 1:
                arr = obj[keys[0]]
            else:
                raise KeyError(f"Cannot infer array key in {p}; keys={keys}")
    else:
        raise ValueError("Only .npy and .npz are supported for training arrays.")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr.shape} from {p}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Array contains non-finite values: {p}")
    return arr


# ============================================================
# Benchmark configuration
# ============================================================
@dataclass
class BenchConfig:
    name: str
    dt_obs: float
    dt_base: float
    tau_pair: int
    tau_delay: int
    dim_delay: int
    delay_ch: int
    delay_mode: str  # "onech" or "multich"
    width: int
    # PRIME section construction
    cand_frac: float
    q_speed: float
    M_landmarks: int
    L_sec: int
    eps_in_mult: float
    eps_out_mult: float
    # Training lengths
    mse_steps: int
    dim_steps: int
    dim_mse_steps: int
    prime_steps: int
    prime_warm_steps: int
    stage1_batch: int
    prime_L_hits: int
    prime_batch_blocks: int
    lr1: float
    lr2: float
    # Loss weights
    dim_w_mse: float
    dim_mse_w_mse: float
    prime_w2_const: float
    alpha_pair_state: float
    alpha_pair_delay: float
    w_marg_u0: float
    w_marg_u1: float
    w_defect: float
    blur_start: float
    blur_end: float
    stage2_steps_per_cycle: int
    stage1_steps_per_cycle: int


def get_bench_config(name: str, quick: bool = False) -> BenchConfig:
    name = name.lower()
    if name == "lorenz":
        cfg = BenchConfig(
            name="lorenz",
            dt_obs=0.1,
            dt_base=0.1,
            tau_pair=2,
            tau_delay=1,
            dim_delay=5,
            delay_ch=0,
            delay_mode="onech",
            width=100,
            cand_frac=0.5,
            q_speed=0.3,
            M_landmarks=200,
            L_sec=5,
            eps_in_mult=0.35,
            eps_out_mult=1.8,
            mse_steps=10000,
            dim_steps=10000,
            dim_mse_steps=10000,
            prime_steps=4000,
            prime_warm_steps=3000,
            stage1_batch=512,
            prime_L_hits=32,
            prime_batch_blocks=16,
            lr1=1e-3,
            lr2=1e-3,
            dim_w_mse=1.0,
            dim_mse_w_mse=0.1,
            prime_w2_const=0.5,
            alpha_pair_state=1.0,
            alpha_pair_delay=0.7,
            w_marg_u0=0.20,
            w_marg_u1=0.20,
            w_defect=0.0,
            blur_start=0.2,
            blur_end=0.1,
            stage2_steps_per_cycle=1,
            stage1_steps_per_cycle=1,
        )
    elif name == "ks":
        cfg = BenchConfig(
            name="ks",
            dt_obs=1.0,
            dt_base=1.0,
            tau_pair=2,
            tau_delay=10,
            dim_delay=3,
            delay_ch=0,
            delay_mode="multich",
            width=100,
            cand_frac=0.2,
            q_speed=0.3,
            M_landmarks=128,
            L_sec=5,
            eps_in_mult=0.35,
            eps_out_mult=1.8,
            mse_steps=10000,
            dim_steps=10000,
            dim_mse_steps=10000,
            prime_steps=12000,
            prime_warm_steps=10000,
            stage1_batch=512,
            prime_L_hits=16,
            prime_batch_blocks=16,
            lr1=1e-3,
            lr2=1e-3,
            dim_w_mse=0.1,      # DIM-style glue used in your KS DIM notebook.
            dim_mse_w_mse=1.0,  # DIM+MSE uses stronger pointwise term.
            prime_w2_const=3.0,
            alpha_pair_state=1.0,
            alpha_pair_delay=0.7,
            w_marg_u0=0.20,
            w_marg_u1=0.20,
            w_defect=0.0,
            blur_start=0.2,
            blur_end=0.1,
            stage2_steps_per_cycle=1,
            stage1_steps_per_cycle=0,
        )
    else:
        raise ValueError(f"Unknown benchmark: {name}")

    if quick:
        # Smoke-test values: useful for verifying the script before full profiling.
        cfg.mse_steps = 50
        cfg.dim_steps = 50
        cfg.dim_mse_steps = 50
        cfg.prime_steps = 80
        cfg.prime_warm_steps = 40
        cfg.stage1_batch = min(64, cfg.stage1_batch)
        cfg.prime_batch_blocks = min(4, cfg.prime_batch_blocks)
        cfg.prime_L_hits = min(8, cfg.prime_L_hits)
        cfg.M_landmarks = min(32, cfg.M_landmarks)
    return cfg


# ============================================================
# Delay helpers
# ============================================================
def build_delay_onech(y_scaled: np.ndarray, tau: int, dim: int, ch: int) -> Tuple[np.ndarray, int]:
    T, d = y_scaled.shape
    min_t = (dim - 1) * tau
    D = np.zeros((T, dim), dtype=np.float64)
    s = y_scaled[:, ch]
    for t in range(min_t, T):
        D[t] = np.asarray([s[t - i * tau] for i in range(dim)], dtype=np.float64)
    return D, min_t


def build_delay_multich(y_scaled: np.ndarray, tau: int, dim: int) -> Tuple[np.ndarray, int]:
    T, d = y_scaled.shape
    min_t = (dim - 1) * tau
    D = np.zeros((T, d * dim), dtype=np.float64)
    for t in range(min_t, T):
        blocks = [y_scaled[t - i * tau] for i in range(dim)]
        D[t] = np.concatenate(blocks, axis=0)
    return D, min_t


def build_delay_targets_onech(y_scaled: np.ndarray, tau: int, dim: int, ch: int) -> Tuple[np.ndarray, int]:
    T = len(y_scaled)
    H = (dim - 1) * tau
    out = np.zeros((T - H, dim), dtype=np.float32)
    s = y_scaled[:, ch]
    for t in range(T - H):
        out[t] = np.asarray([s[t + H - i * tau] for i in range(dim)], dtype=np.float32)
    return out, H


def build_delay_targets_multich(y_scaled: np.ndarray, tau: int, dim: int) -> Tuple[np.ndarray, int]:
    T, d = y_scaled.shape
    H = (dim - 1) * tau
    out = np.zeros((T - H, d * dim), dtype=np.float32)
    for t in range(T - H):
        blocks = [y_scaled[t + H - i * tau] for i in range(dim)]
        out[t] = np.concatenate(blocks, axis=0)
    return out, H


# ============================================================
# Section construction and PRIME block helpers
# ============================================================
def farthest_point_sampling(X: np.ndarray, M: int, start: int = 0) -> np.ndarray:
    N = int(X.shape[0])
    M = int(min(int(M), N))
    idx = np.zeros((M,), dtype=np.int64)
    idx[0] = int(np.clip(start, 0, N - 1))
    dist2 = np.sum((X - X[idx[0]]) ** 2, axis=1)
    for m in range(1, M):
        j = int(np.argmax(dist2))
        idx[m] = j
        d2_new = np.sum((X - X[j]) ** 2, axis=1)
        dist2 = np.minimum(dist2, d2_new)
    return idx


def extract_hits_from_landmarks(U_dw: np.ndarray, C_dw: np.ndarray, eps_in: float, eps_out: float, L_sec: int):
    T = int(U_dw.shape[0])
    hits = []
    visit_lens = []
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
    return np.asarray(hits, dtype=np.int64), np.asarray(visit_lens, dtype=np.int64)


def build_prime_section(y_scaled: np.ndarray, cfg: BenchConfig, seed: int):
    if cfg.delay_mode == "onech":
        D_full, min_t = build_delay_onech(y_scaled, cfg.tau_delay, cfg.dim_delay, cfg.delay_ch)
    else:
        D_full, min_t = build_delay_multich(y_scaled, cfg.tau_delay, cfg.dim_delay)

    U_full = np.concatenate([y_scaled, D_full], axis=1)
    T, d = y_scaled.shape
    T_eff = T - min_t
    idx_eff = np.arange(min_t, T)

    speed = np.linalg.norm(y_scaled[1:] - y_scaled[:-1], axis=1)
    speed = np.concatenate([[speed[0]], speed], axis=0)
    thr = np.quantile(speed[idx_eff], cfg.q_speed)
    pool = idx_eff[speed[idx_eff] >= thr]

    rng = np.random.default_rng(int(seed))
    cand_target = max(1, int(round(cfg.cand_frac * T_eff)))
    if len(pool) >= cand_target:
        idx_cand = rng.choice(pool, size=cand_target, replace=False)
    else:
        idx_cand = pool.copy()
    idx_cand = np.sort(idx_cand)

    Ucand = U_full[idx_cand]
    u_mean_c = Ucand.mean(axis=0)
    u_std_c = Ucand.std(axis=0) + 1e-6
    Ucand_dw = (Ucand - u_mean_c[None, :]) / u_std_c[None, :]
    Ufull_dw = (U_full - u_mean_c[None, :]) / u_std_c[None, :]

    start0 = int(rng.integers(0, len(idx_cand)))
    fps_idx = farthest_point_sampling(Ucand_dw, cfg.M_landmarks, start=start0)
    C_dw = Ucand_dw[fps_idx]

    nn = []
    for i in range(len(C_dw)):
        d2 = np.sum((C_dw - C_dw[i]) ** 2, axis=1)
        d2[i] = np.inf
        nn.append(np.sqrt(np.min(d2)))
    nn_med = float(np.median(nn))
    eps_in = cfg.eps_in_mult * nn_med
    eps_out = cfg.eps_out_mult * eps_in

    hits_raw, visit_lens = extract_hits_from_landmarks(Ufull_dw, C_dw, eps_in, eps_out, cfg.L_sec)
    hits = np.sort(hits_raw[hits_raw >= min_t])

    XU_data = y_scaled[hits] if len(hits) else np.zeros((1, d), dtype=np.float64)
    DU_data = D_full[hits] if len(hits) else np.zeros((1, D_full.shape[1]), dtype=np.float64)
    U_hits_data = np.concatenate([XU_data, DU_data], axis=1)

    return dict(
        D_full=D_full,
        min_t=min_t,
        U_full=U_full,
        hits=hits,
        visit_lens=visit_lens,
        XU_data=XU_data,
        DU_data=DU_data,
        U_hits_data=U_hits_data,
        N_hits=int(len(hits)),
    )


# ============================================================
# Training objective helpers
# ============================================================
class ObjectiveTrainer:
    def __init__(self, y_scaled: np.ndarray, cfg: BenchConfig, device: torch.device):
        self.y_scaled = np.asarray(y_scaled, dtype=np.float64)
        self.cfg = cfg
        self.device = device
        self.T, self.d = self.y_scaled.shape
        self.sub = int(round(cfg.dt_obs / cfg.dt_base))
        if self.sub < 1:
            raise ValueError("sub < 1; check dt_obs/dt_base")

        self.y_t = torch.tensor(self.y_scaled, dtype=torch.float32, device=device)

        x_mean = self.y_scaled.mean(axis=0)
        x_std = self.y_scaled.std(axis=0) + 1e-6
        self.x_mean_t = torch.tensor(x_mean, dtype=torch.float32, device=device)
        self.x_std_t = torch.tensor(x_std, dtype=torch.float32, device=device)

        if cfg.delay_mode == "onech":
            delay_targets, H = build_delay_targets_onech(self.y_scaled, cfg.tau_delay, cfg.dim_delay, cfg.delay_ch)
        else:
            delay_targets, H = build_delay_targets_multich(self.y_scaled, cfg.tau_delay, cfg.dim_delay)
        self.delay_targets = delay_targets
        self.delay_H = H
        d_mean = delay_targets.mean(axis=0)
        d_std = delay_targets.std(axis=0) + 1e-6
        self.d_mean_t = torch.tensor(d_mean, dtype=torch.float32, device=device)
        self.d_std_t = torch.tensor(d_std, dtype=torch.float32, device=device)

    def whiten_x(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.x_mean_t[None, :]) / self.x_std_t[None, :]

    def whiten_delay(self, D: torch.Tensor) -> torch.Tensor:
        return (D - self.d_mean_t[None, :]) / self.d_std_t[None, :]

    def stage1_loss(self, net: nn.Module, loss_sink, batch_size: int, w_mse: float = 0.0, glue: bool = True) -> torch.Tensor:
        cfg = self.cfg
        max_horizon = max(cfg.tau_pair, self.delay_H)
        Tmax = self.T - max_horizon - 1
        idx = np.random.randint(0, Tmax, size=int(batch_size))

        x0 = torch.tensor(self.y_scaled[idx], dtype=torch.float32, device=self.device)
        xT_d = torch.tensor(self.y_scaled[idx + cfg.tau_pair], dtype=torch.float32, device=self.device)

        x = x0.clone()
        for _ in range(int(cfg.tau_pair * self.sub)):
            x = x + float(cfg.dt_base) * net(x)
        xT_m = x

        x0w = self.whiten_x(x0)
        xTd = self.whiten_x(xT_d)
        xTm = self.whiten_x(xT_m)
        L_state = loss_sink(torch.cat([x0w, xTm], dim=1), torch.cat([x0w, xTd], dim=1))
        L_mse = torch.mean((xT_m - xT_d) ** 2)

        if cfg.delay_mode == "onech":
            # Full-trajectory one-channel delay cloud, matching your Lorenz PRIME/DIM logic.
            Kfine = cfg.tau_delay * cfg.dim_delay * self.sub
            yy = self.y_t.clone()
            sols = []
            for _ in range(Kfine):
                sols.append(yy)
                for _ in range(self.sub):
                    yy = yy + float(cfg.dt_base) * net(yy)
            sol = torch.stack(sols, dim=0)[:, self.delay_H:, :]
            step = cfg.tau_delay * self.sub
            sol_ch = sol[:, :, cfg.delay_ch].permute(1, 0).contiguous()
            D_model = torch.flip(sol_ch[:, ::step], dims=[1]).unsqueeze(0).contiguous()
            D_data = torch.tensor(
                self.delay_targets,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            L_delay = loss_sink(D_model, D_data).sum()
            L_glue = torch.mean((D_model[:, 0, :-1] - D_data[:, 0, :-1]) ** 2) if glue else 0.0
        else:
            # Batch multichannel delay cloud, matching your KS PRIME logic.
            y_delay_batch = torch.tensor(self.delay_targets[idx], dtype=torch.float32, device=self.device)
            x_d = torch.tensor(self.y_scaled[idx], dtype=torch.float32, device=self.device)
            sols = []
            n_dim_steps = cfg.tau_delay * cfg.dim_delay * self.sub
            for _ in range(n_dim_steps):
                sols.append(x_d)
                x_d = x_d + float(cfg.dt_base) * net(x_d)
            sol_d = torch.stack(sols, dim=0)
            step = cfg.tau_delay * self.sub
            delay_blocks = torch.flip(sol_d[::step], dims=[0])
            D_model = delay_blocks.permute(1, 0, 2).contiguous().view(len(idx), cfg.dim_delay * self.d)
            L_delay = loss_sink(self.whiten_delay(D_model), self.whiten_delay(y_delay_batch))
            L_glue = 0.0

        return L_state + L_delay + (float(w_mse) * L_mse) + L_glue


def build_induced_block_loss(
    net: nn.Module,
    loss_sink,
    y_scaled: np.ndarray,
    cfg: BenchConfig,
    section: Dict,
    whiten_u_fn,
    device: torch.device,
    k0: int,
    L_hits: int,
) -> torch.Tensor:
    d = y_scaled.shape[1]
    sub = int(round(cfg.dt_obs / cfg.dt_base))
    hits = section["hits"]
    hit_times = hits[k0:k0 + L_hits].astype(np.int64)
    XU_data = section["XU_data"]
    DU_data = section["DU_data"]

    X_blk_d = torch.tensor(XU_data[k0:k0 + L_hits], dtype=torch.float32, device=device)
    D_blk_d = torch.tensor(DU_data[k0:k0 + L_hits], dtype=torch.float32, device=device)

    t_start = int(hit_times[0] - (cfg.dim_delay - 1) * cfg.tau_delay)
    t_end = int(hit_times[-1])
    total_obs_steps = int(t_end - t_start)
    if t_start < 0:
        raise RuntimeError("negative t_start in PRIME block")

    x = torch.tensor(y_scaled[t_start], dtype=torch.float32, device=device)
    hit_map = {int(t): j for j, t in enumerate(hit_times.tolist())}
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
            x = x + float(cfg.dt_base) * net(x)
        x_hist.append(x.clone())

    if any(v is None for v in x_hits_m):
        raise RuntimeError("some hit states not recorded")

    X_blk_m = torch.stack(x_hits_m, dim=0)

    Dm_list = []
    for t_abs in hit_times:
        idx0 = int(t_abs - t_start)
        if cfg.delay_mode == "onech":
            lag_vals = []
            for i in range(cfg.dim_delay):
                lag_vals.append(x_hist[idx0 - i * cfg.tau_delay][cfg.delay_ch])
            Dm_list.append(torch.stack(lag_vals, dim=0))
        else:
            lag_states = []
            for i in range(cfg.dim_delay):
                lag_states.append(x_hist[idx0 - i * cfg.tau_delay])
            Dm_list.append(torch.stack(lag_states, dim=0).reshape(-1))
    D_blk_m = torch.stack(Dm_list, dim=0)

    U_blk_m = torch.cat([X_blk_m, D_blk_m], dim=1)
    U_blk_d = torch.cat([X_blk_d, D_blk_d], dim=1)
    Um = whiten_u_fn(U_blk_m)
    Ud = whiten_u_fn(U_blk_d)

    xm, dm = Um[:, :d], Um[:, d:]
    xd, dd = Ud[:, :d], Ud[:, d:]
    u_m = torch.cat([cfg.alpha_pair_state * xm, cfg.alpha_pair_delay * dm], dim=1)
    u_d = torch.cat([cfg.alpha_pair_state * xd, cfg.alpha_pair_delay * dd], dim=1)

    u0_m, u1_m = u_m[:-1], u_m[1:]
    u0_d, u1_d = u_d[:-1], u_d[1:]
    pair_m = torch.cat([u0_m, u1_m], dim=1)
    pair_d = torch.cat([u0_d, u1_d], dim=1)

    L_graph = loss_sink(pair_m, pair_d)
    L_marg0 = loss_sink(u0_m, u0_d)
    L_marg1 = loss_sink(u1_m, u1_d)
    defect = torch.mean((Um[:, :d] - Ud[:, :d]) ** 2)
    L_raw = L_graph + cfg.w_marg_u0 * L_marg0 + cfg.w_marg_u1 * L_marg1 + cfg.w_defect * defect
    return L_raw / float(L_hits)


# ============================================================
# Training routines
# ============================================================
def train_mse(y_scaled: np.ndarray, cfg: BenchConfig, device: torch.device, steps: int) -> Tuple[nn.Module, float]:
    T, d = y_scaled.shape
    sub = int(round(cfg.dt_obs / cfg.dt_base))
    net = build_net(d, cfg.width).to(device)
    opt = optim.Adam(net.parameters(), lr=cfg.lr1)
    Tmax = T - cfg.tau_pair - 1
    loss_val = float("nan")
    net.train()
    for _ in range(int(steps)):
        idx = np.random.randint(0, Tmax, size=cfg.stage1_batch)
        x0 = torch.tensor(y_scaled[idx], dtype=torch.float32, device=device)
        yT = torch.tensor(y_scaled[idx + cfg.tau_pair], dtype=torch.float32, device=device)
        x = x0.clone()
        for _ in range(int(cfg.tau_pair * sub)):
            x = x + float(cfg.dt_base) * net(x)
        loss = torch.mean((x - yT) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_val = float(loss.detach().cpu())
    return net, loss_val


def train_dim_like(y_scaled: np.ndarray, cfg: BenchConfig, device: torch.device, steps: int, w_mse: float) -> Tuple[nn.Module, float]:
    T, d = y_scaled.shape
    net = build_net(d, cfg.width).to(device)
    opt = optim.Adam(net.parameters(), lr=cfg.lr1)
    trainer = ObjectiveTrainer(y_scaled, cfg, device)
    loss_val = float("nan")
    net.train()
    for step in range(int(steps)):
        tfrac = step / max(1, int(steps) - 1)
        blur = cfg.blur_start + (cfg.blur_end - cfg.blur_start) * tfrac
        loss_sink = make_sinkhorn(blur)
        loss = trainer.stage1_loss(net, loss_sink, cfg.stage1_batch, w_mse=w_mse, glue=(cfg.delay_mode == "onech"))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        loss_val = float(loss.detach().cpu())
    return net, loss_val


def train_prime(y_scaled: np.ndarray, cfg: BenchConfig, device: torch.device, seed: int, steps: int, warm_steps: int) -> Tuple[nn.Module, float, int]:
    T, d = y_scaled.shape
    trainer = ObjectiveTrainer(y_scaled, cfg, device)
    section = build_prime_section(y_scaled, cfg, seed=seed)
    hits = section["hits"]
    if len(hits) < 4:
        raise RuntimeError(f"Too few recurrence events: {len(hits)}")

    u_mean = section["U_hits_data"].mean(axis=0)
    u_std = section["U_hits_data"].std(axis=0) + 1e-6
    u_mean_t = torch.tensor(u_mean, dtype=torch.float32, device=device)
    u_std_t = torch.tensor(u_std, dtype=torch.float32, device=device)

    def whiten_u(U: torch.Tensor) -> torch.Tensor:
        return (U - u_mean_t[None, :]) / u_std_t[None, :]

    net = build_net(d, cfg.width).to(device)
    opt1 = optim.Adam(net.parameters(), lr=cfg.lr1)
    opt2 = optim.Adam(net.parameters(), lr=cfg.lr2)
    loss_val = float("nan")
    net.train()
    for step in range(int(steps)):
        tfrac = step / max(1, int(steps) - 1)
        blur = cfg.blur_start + (cfg.blur_end - cfg.blur_start) * tfrac
        loss_sink = make_sinkhorn(blur)

        if step < int(warm_steps):
            stage = "stage1"
        else:
            cyc = int(cfg.stage2_steps_per_cycle + cfg.stage1_steps_per_cycle)
            if cyc <= 0:
                stage = "stage1"
            else:
                k = (step - int(warm_steps)) % cyc
                stage = "stage2" if k < cfg.stage2_steps_per_cycle else "stage1"

        opt = opt1 if stage == "stage1" else opt2
        opt.zero_grad(set_to_none=True)
        L1 = trainer.stage1_loss(net, loss_sink, cfg.stage1_batch, w_mse=0.0, glue=(cfg.delay_mode == "onech"))

        if stage == "stage1":
            loss = L1
        else:
            L_hits = min(int(cfg.prime_L_hits), len(hits))
            k0max = len(hits) - L_hits
            blocks = []
            for _ in range(int(cfg.prime_batch_blocks)):
                k0 = np.random.randint(0, k0max + 1)
                blocks.append(build_induced_block_loss(net, loss_sink, y_scaled, cfg, section, whiten_u, device, k0, L_hits))
            L2 = torch.stack(blocks).mean()
            loss = L1 + float(cfg.prime_w2_const) * L2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        loss_val = float(loss.detach().cpu())

    return net, loss_val, int(section["N_hits"])


# ============================================================
# Profiling wrapper
# ============================================================
def profile_one(
    benchmark: str,
    method: str,
    train_raw: np.ndarray,
    cfg: BenchConfig,
    seed: int,
    device: torch.device,
) -> Dict:
    set_seed(seed)
    reset_cuda_peak(device)
    start_rss = current_rss_mb()
    setup_t0 = time.perf_counter()

    scaler = MaxAbsScaler().fit(train_raw.astype(np.float32))
    y_scaled = scaler.transform(train_raw.astype(np.float32)).astype(np.float64)

    # Method-specific step count.
    if method == "mse":
        steps = cfg.mse_steps
    elif method == "dim":
        steps = cfg.dim_steps
    elif method == "dim_mse":
        steps = cfg.dim_mse_steps
    elif method == "prime":
        steps = cfg.prime_steps
    else:
        raise ValueError(f"Unknown method: {method}")

    setup_sec = time.perf_counter() - setup_t0
    sync_cuda(device)
    train_t0 = time.perf_counter()

    n_hits = None
    if method == "mse":
        net, final_loss = train_mse(y_scaled, cfg, device, steps)
    elif method == "dim":
        net, final_loss = train_dim_like(y_scaled, cfg, device, steps, w_mse=cfg.dim_w_mse)
    elif method == "dim_mse":
        net, final_loss = train_dim_like(y_scaled, cfg, device, steps, w_mse=cfg.dim_mse_w_mse)
    elif method == "prime":
        net, final_loss, n_hits = train_prime(y_scaled, cfg, device, seed=seed, steps=steps, warm_steps=cfg.prime_warm_steps)

    sync_cuda(device)
    train_sec = time.perf_counter() - train_t0
    alloc_mb, reserved_mb = peak_cuda_mb(device)
    end_rss = current_rss_mb()

    return {
        "benchmark": benchmark,
        "method": method,
        "seed": int(seed),
        "device": str(device),
        "n_train": int(train_raw.shape[0]),
        "dim": int(train_raw.shape[1]),
        "steps": int(steps),
        "setup_sec": float(setup_sec),
        "train_sec": float(train_sec),
        "total_sec": float(setup_sec + train_sec),
        "avg_iter_ms": float(1000.0 * train_sec / max(1, int(steps))),
        "peak_cuda_alloc_mb": float(alloc_mb),
        "peak_cuda_reserved_mb": float(reserved_mb),
        "rss_mb_start": float(start_rss),
        "rss_mb_end": float(end_rss),
        "params": int(count_params(net)),
        "final_loss": float(final_loss),
        "n_hits": int(n_hits) if n_hits is not None else "",
        "config": json.dumps(asdict(cfg)),
    }


def write_csv(rows: List[Dict], path: Path) -> None:
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


def summarize_rows(rows: List[Dict]) -> List[Dict]:
    groups = {}
    for r in rows:
        key = (r["benchmark"], r["method"])
        groups.setdefault(key, []).append(r)
    out = []
    for (bench, method), rs in groups.items():
        row = {"benchmark": bench, "method": method, "n_runs": len(rs)}
        for metric in ["setup_sec", "train_sec", "total_sec", "avg_iter_ms", "peak_cuda_alloc_mb", "peak_cuda_reserved_mb", "rss_mb_end"]:
            vals = np.asarray([float(r[metric]) for r in rs], dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals))
        row["steps"] = int(rs[0]["steps"])
        row["params"] = int(rs[0]["params"])
        out.append(row)
    return out


def write_markdown_table(summary: List[Dict], path: Path) -> None:
    with open(path, "w") as f:
        f.write("# Computational cost summary\n\n")
        f.write("Wall-clock time includes objective optimization only; setup time is reported separately. CUDA memory is peak allocated memory.\n\n")
        f.write("| Benchmark | Method | Steps | Train sec | Avg iter ms | Peak CUDA MB | Params |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for r in summary:
            f.write(
                f"| {r['benchmark']} | {r['method']} | {r['steps']} | "
                f"{r['train_sec_mean']:.2f} ± {r['train_sec_std']:.2f} | "
                f"{r['avg_iter_ms_mean']:.2f} ± {r['avg_iter_ms_std']:.2f} | "
                f"{r['peak_cuda_alloc_mb_mean']:.1f} ± {r['peak_cuda_alloc_mb_std']:.1f} | "
                f"{r['params']} |\n"
            )


# ============================================================
# CLI
# ============================================================
ROOT = Path(__file__).resolve().parents[2]

def parse_args():
    p = argparse.ArgumentParser(description="Profile computational cost for Lorenz/KS objective-controlled training.")
    p.add_argument("--benchmark", type=str, default="all", choices=["lorenz", "ks", "all"])
    p.add_argument("--methods", type=str, default="mse,dim,dim_mse,prime")
    p.add_argument("--seeds", type=str, default="1000")
    p.add_argument("--lorenz-train", "--lorenz_train", dest="lorenz_train", type=str, default=str(ROOT / "data/lorenz63/train_data.npy"))
    p.add_argument("--ks-train", "--ks_train", dest="ks_train", type=str, default=str(ROOT / "data/ks/train.npy"))
    p.add_argument("--output-dir", "--out_dir", dest="out_dir", type=str, default=str(ROOT / "runs/cost/training"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--quick", action="store_true", help="Run a tiny smoke test with greatly reduced training steps.")
    p.add_argument("--allow_missing", action="store_true", help="Skip missing dataset paths instead of raising an error.")
    return p.parse_args()


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = ["lorenz", "ks"] if args.benchmark == "all" else [args.benchmark]
    datasets = {}
    for bench in requested:
        path = args.lorenz_train if bench == "lorenz" else args.ks_train
        if not Path(path).exists():
            msg = f"Missing {bench} training data: {path}"
            if args.allow_missing:
                print("[skip]", msg)
                continue
            raise FileNotFoundError(msg)
        datasets[bench] = load_array(path)

    rows = []
    print("=" * 80)
    print("Computational cost profiler")
    print("device:", device)
    print("benchmarks:", list(datasets.keys()))
    print("methods:", methods)
    print("seeds:", seeds)
    print("quick:", args.quick)
    print("=" * 80)

    for bench, train_raw in datasets.items():
        cfg = get_bench_config(bench, quick=args.quick)
        for seed in seeds:
            for method in methods:
                print(f"\n[RUN] benchmark={bench} method={method} seed={seed} shape={train_raw.shape}")
                try:
                    row = profile_one(bench, method, train_raw, cfg, seed, device)
                    rows.append(row)
                    print(
                        f"  train_sec={row['train_sec']:.2f}s | avg_iter={row['avg_iter_ms']:.2f}ms | "
                        f"peak_cuda={row['peak_cuda_alloc_mb']:.1f}MB | final_loss={row['final_loss']:.4g}"
                    )
                except Exception as e:
                    print(f"  [ERROR] {bench}/{method}/seed={seed}: {repr(e)}")
                    rows.append({
                        "benchmark": bench,
                        "method": method,
                        "seed": int(seed),
                        "error": repr(e),
                    })

    raw_csv = out_dir / "computational_cost_raw.csv"
    write_csv(rows, raw_csv)

    ok_rows = [r for r in rows if "error" not in r]
    errors = [r for r in rows if "error" in r]
    (out_dir / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
    summary = summarize_rows(ok_rows)
    summary_csv = out_dir / "computational_cost_summary.csv"
    write_csv(summary, summary_csv)
    md_path = out_dir / "computational_cost_summary.md"
    write_markdown_table(summary, md_path)

    print("\nSaved:")
    print("  raw CSV    :", raw_csv)
    print("  summary CSV:", summary_csv)
    print("  summary MD :", md_path)

    print("\nCompact summary:")
    for r in summary:
        print(
            f"{r['benchmark']:7s} | {r['method']:7s} | steps={r['steps']:5d} | "
            f"train={r['train_sec_mean']:.2f}±{r['train_sec_std']:.2f}s | "
            f"iter={r['avg_iter_ms_mean']:.2f}ms | "
            f"cuda={r['peak_cuda_alloc_mb_mean']:.1f}MB"
        )

    if errors:
        raise RuntimeError(f"{len(errors)} cost profiling run(s) failed; see errors.json")


if __name__ == "__main__":
    main()
