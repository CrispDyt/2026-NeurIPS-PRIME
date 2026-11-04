#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data-side diagnostics for the data-driven recurrence section used by PRIME.

This script is designed to complement recurrence-clock robustness experiments.
It does NOT train a model. Instead, it tests whether a proposed data-driven
recurrence section produces stable, non-degenerate recurrence events and return
statistics before PRIME training.

Main outputs:
  1) section_diagnostics_per_seed.csv
  2) section_diagnostics_aggregate.csv
  3) section_diagnostics_table.md
  4) section_diagnostics_table.tex
  5) simple diagnostic plots

Typical Lorenz63 usage:
  python recurrence_section_diagnostics.py \
      --train_path train_data.npy \
      --test_path test_data.npy \
      --out_dir ./section_diagnostics \
      --preset lorenz \
      --seeds 33,123,1000,13532,123456

Minimal usage:
  python recurrence_section_diagnostics.py --train_path train_data.npy --test_path test_data.npy

The extractor matches the logic used in your PRIME scripts:
  - MaxAbs scaling fit on train data
  - one-channel delay embedding
  - augmented state U_t = [y_t, delay_t]
  - speed-quantile filtering
  - random candidate subsampling
  - farthest point sampling landmarks
  - adaptive entry/exit thresholds
  - hysteresis hit extraction with minimum gap L_sec

Diagnostics:
  - N_hits and return-time statistics: mean/std/CV/min/max of K_i
  - hit-speed diagnostics: avoids quasi-stationary events
  - landmark usage: whether hits cover many landmark neighborhoods
  - split-half stability of the empirical return-graph measure
  - split-half stability of return-time distribution
  - optional train-test stability using the same train-derived section on test_data
  - optional random-hit negative control with the same number of events

Interpretation:
  A good recurrence section should have enough hits, a non-degenerate return-time
  distribution, broad landmark usage, and low split-half/train-test return-graph
  distances. This does not prove exact Poincare-section recovery; it supports the
  practical assumption that stable recurrence events can be extracted from the
  observed trajectory.
"""

import argparse
import csv
import datetime
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

try:
    from sklearn.preprocessing import MaxAbsScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# ============================================================
# Basic utilities
# ============================================================
def parse_list_int(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_float(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def load_array(path: str) -> np.ndarray:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p, allow_pickle=True)
    elif suffix == ".npz":
        obj = np.load(p, allow_pickle=True)
        for key in ["train_data", "test_data", "train", "test", "data", "X", "Y"]:
            if key in obj:
                arr = obj[key]
                break
        else:
            keys = list(obj.keys())
            if len(keys) == 1:
                arr = obj[keys[0]]
            else:
                raise KeyError(f"Could not infer key in {p}. Available keys={keys}")
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .npy or .npz")

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Array contains non-finite values: {p}")
    return arr


class SimpleMaxAbsScaler:
    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.scale_ = np.max(np.abs(X), axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform(self, X: np.ndarray):
        return np.asarray(X, dtype=np.float64) / self.scale_[None, :]


def make_maxabs_scaler(train: np.ndarray):
    if HAS_SKLEARN:
        return MaxAbsScaler().fit(train)
    return SimpleMaxAbsScaler().fit(train)


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v))


def median_iqr(vals: Sequence[float]) -> Tuple[float, float]:
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    q25, q75 = np.quantile(v, [0.25, 0.75])
    return float(np.median(v)), float(q75 - q25)


# ============================================================
# Section construction logic
# ============================================================
def build_delay_full_onech(y_scaled: np.ndarray, tau: int, dim: int, ch: int) -> Tuple[np.ndarray, int]:
    """
    Build one-channel delay coordinates aligned with original time indices.
    D[t] = [y_ch(t), y_ch(t-tau), ..., y_ch(t-(dim-1)tau)] for t >= min_t.
    Earlier rows are zeros and should be ignored.
    """
    y_scaled = np.asarray(y_scaled, dtype=np.float64)
    T, d = y_scaled.shape
    if ch < 0 or ch >= d:
        raise ValueError(f"delay_ch={ch} out of range for d={d}")
    tau = int(tau)
    dim = int(dim)
    min_t = (dim - 1) * tau
    if min_t >= T:
        raise ValueError(f"Delay window too long: min_t={min_t}, T={T}")
    D = np.zeros((T, dim), dtype=np.float64)
    s = y_scaled[:, ch]
    for t in range(min_t, T):
        D[t] = np.asarray([s[t - i * tau] for i in range(dim)], dtype=np.float64)
    return D, min_t


def farthest_point_sampling(X: np.ndarray, M: int, start: int = 0) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    N = int(X.shape[0])
    M = int(min(max(1, int(M)), N))
    idx = np.zeros((M,), dtype=np.int64)
    idx[0] = int(np.clip(start, 0, N - 1))
    dist2 = np.sum((X - X[idx[0]]) ** 2, axis=1)
    for m in range(1, M):
        j = int(np.argmax(dist2))
        idx[m] = j
        d2_new = np.sum((X - X[j]) ** 2, axis=1)
        dist2 = np.minimum(dist2, d2_new)
    return idx


def nearest_landmarks(U_dw: np.ndarray, C_dw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return nearest landmark ids and distances for each row of U_dw."""
    U_dw = np.asarray(U_dw, dtype=np.float64)
    C_dw = np.asarray(C_dw, dtype=np.float64)
    ids = np.empty((len(U_dw),), dtype=np.int64)
    dists = np.empty((len(U_dw),), dtype=np.float64)
    for t in range(len(U_dw)):
        d2 = np.sum((C_dw - U_dw[t]) ** 2, axis=1)
        i = int(np.argmin(d2))
        ids[t] = i
        dists[t] = float(np.sqrt(d2[i]))
    return ids, dists


def extract_hits_from_landmarks(
    U_dw: np.ndarray,
    C_dw: np.ndarray,
    eps_in: float,
    eps_out: float,
    L_sec: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hysteresis hit extraction.
    Returns:
      hits: hit indices in original aligned time axis
      visit_lens: residence lengths before exit
      hit_landmark_ids: nearest landmark id at hit
      hit_dists: nearest landmark distance at hit
    """
    T = int(U_dw.shape[0])
    hits: List[int] = []
    visit_lens: List[int] = []
    hit_ids: List[int] = []
    hit_dists: List[float] = []
    last = -10**12
    active = False
    active_i = -1
    enter_t = -1

    for t in range(T):
        d2 = np.sum((C_dw - U_dw[t]) ** 2, axis=1)
        i = int(np.argmin(d2))
        d = float(np.sqrt(d2[i]))

        if not active:
            if d <= float(eps_in) and (t - last) >= int(L_sec):
                hits.append(t)
                hit_ids.append(i)
                hit_dists.append(d)
                last = t
                active = True
                active_i = i
                enter_t = t
        else:
            d_active = float(np.sqrt(np.sum((U_dw[t] - C_dw[active_i]) ** 2)))
            if d_active >= float(eps_out):
                active = False
                active_i = -1
                if enter_t >= 0:
                    visit_lens.append(t - enter_t)
                enter_t = -1

    return (
        np.asarray(hits, dtype=np.int64),
        np.asarray(visit_lens, dtype=np.int64),
        np.asarray(hit_ids, dtype=np.int64),
        np.asarray(hit_dists, dtype=np.float64),
    )


@dataclass
class SectionConfig:
    case: str
    family: str
    q_speed: float = 0.3
    cand_frac: float = 0.5
    M_landmarks: int = 200
    L_sec: int = 5
    tau_delay: int = 1
    dim_delay: int = 5
    delay_ch: int = 0
    eps_in_mult: float = 0.35
    eps_out_mult: float = 1.8
    random_control: bool = False


def construct_section(y_scaled: np.ndarray, cfg: SectionConfig, seed: int) -> Dict[str, Any]:
    """Construct a train-derived data-driven recurrence section."""
    rng = np.random.default_rng(int(seed))
    y_scaled = np.asarray(y_scaled, dtype=np.float64)
    T, d = y_scaled.shape

    D_full, min_t = build_delay_full_onech(
        y_scaled, tau=int(cfg.tau_delay), dim=int(cfg.dim_delay), ch=int(cfg.delay_ch)
    )
    U_full = np.concatenate([y_scaled, D_full], axis=1)
    T_eff = T - min_t
    if T_eff <= 10:
        raise ValueError(f"Too few effective samples: T={T}, min_t={min_t}")

    speed = np.linalg.norm(y_scaled[1:] - y_scaled[:-1], axis=1)
    speed = np.concatenate([[speed[0]], speed], axis=0)
    idx_eff = np.arange(min_t, T)
    speed_eff = speed[idx_eff]
    speed_thr = float(np.quantile(speed_eff, float(cfg.q_speed)))
    pool = idx_eff[speed_eff >= speed_thr]

    cand_target = max(1, int(round(float(cfg.cand_frac) * T_eff)))
    if len(pool) >= cand_target:
        idx_cand = rng.choice(pool, size=cand_target, replace=False)
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

    start0 = int(rng.integers(0, len(idx_cand)))
    fps_idx_local = farthest_point_sampling(Ucand_dw, M=int(cfg.M_landmarks), start=start0)
    C_dw = Ucand_dw[fps_idx_local]
    M_actual = int(len(C_dw))

    if M_actual >= 2:
        nn = []
        for i in range(M_actual):
            d2 = np.sum((C_dw - C_dw[i]) ** 2, axis=1)
            d2[i] = np.inf
            nn.append(float(np.sqrt(np.min(d2))))
        nn_med = float(np.median(nn))
    else:
        nn_med = float("nan")

    eps_in = float(cfg.eps_in_mult) * nn_med
    eps_out = float(cfg.eps_out_mult) * eps_in

    hits_raw, visit_lens, hit_lm_ids, hit_dists = extract_hits_from_landmarks(
        Ufull_dw, C_dw, eps_in, eps_out, L_sec=int(cfg.L_sec)
    )
    keep = hits_raw >= min_t
    hits = np.sort(hits_raw[keep])
    # Align landmark/distance arrays after sorting/keeping. hits_raw is already time-ordered.
    hit_lm_ids = hit_lm_ids[keep]
    hit_dists = hit_dists[keep]

    return {
        "cfg": cfg,
        "T": T,
        "d": d,
        "D_full": D_full,
        "U_full": U_full,
        "Ufull_dw": Ufull_dw,
        "speed": speed,
        "idx_eff": idx_eff,
        "idx_cand": idx_cand,
        "pool": pool,
        "C_dw": C_dw,
        "u_mean_c": u_mean_c,
        "u_std_c": u_std_c,
        "eps_in": eps_in,
        "eps_out": eps_out,
        "nn_med": nn_med,
        "min_t": min_t,
        "hits": hits,
        "visit_lens": visit_lens,
        "hit_lm_ids": hit_lm_ids,
        "hit_dists": hit_dists,
        "speed_thr": speed_thr,
        "pool_size": int(len(pool)),
        "pool_frac": float(len(pool) / max(1, T_eff)),
        "cand_size": int(len(idx_cand)),
        "T_eff": int(T_eff),
        "M_landmarks_actual": M_actual,
        "fps_start_local": start0,
    }


def apply_section_to_new_data(y_scaled: np.ndarray, section: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a train-derived section to another trajectory, e.g. test_data."""
    cfg: SectionConfig = section["cfg"]
    D_full, min_t = build_delay_full_onech(
        y_scaled, tau=int(cfg.tau_delay), dim=int(cfg.dim_delay), ch=int(cfg.delay_ch)
    )
    U_full = np.concatenate([y_scaled, D_full], axis=1)
    Ufull_dw = (U_full - section["u_mean_c"][None, :]) / section["u_std_c"][None, :]
    hits_raw, visit_lens, hit_lm_ids, hit_dists = extract_hits_from_landmarks(
        Ufull_dw,
        section["C_dw"],
        section["eps_in"],
        section["eps_out"],
        L_sec=int(cfg.L_sec),
    )
    keep = hits_raw >= min_t
    hits = np.sort(hits_raw[keep])
    hit_lm_ids = hit_lm_ids[keep]
    hit_dists = hit_dists[keep]
    speed = np.linalg.norm(y_scaled[1:] - y_scaled[:-1], axis=1)
    speed = np.concatenate([[speed[0]], speed], axis=0)
    return {
        "D_full": D_full,
        "U_full": U_full,
        "Ufull_dw": Ufull_dw,
        "speed": speed,
        "min_t": min_t,
        "hits": hits,
        "visit_lens": visit_lens,
        "hit_lm_ids": hit_lm_ids,
        "hit_dists": hit_dists,
    }


# ============================================================
# Diagnostics
# ============================================================
def sample_random_hits(T: int, min_t: int, n_hits: int, L_sec: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    candidates = np.arange(int(min_t), int(T), dtype=np.int64)
    if len(candidates) == 0 or n_hits <= 0:
        return np.asarray([], dtype=np.int64)
    n_hits = int(min(int(n_hits), len(candidates)))

    # Greedy min-gap selection over random permutation.
    for _ in range(200):
        perm = rng.permutation(candidates)
        selected: List[int] = []
        for t in perm:
            if all(abs(int(t) - int(s)) >= int(L_sec) for s in selected):
                selected.append(int(t))
                if len(selected) >= n_hits:
                    return np.asarray(sorted(selected), dtype=np.int64)

    # Fallback: evenly spaced hits.
    grid = np.linspace(min_t, T - 1, n_hits).astype(np.int64)
    return np.asarray(sorted(np.unique(grid)), dtype=np.int64)


def quantile_w1_1d(a: np.ndarray, b: np.ndarray, n_q: int = 512) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    n = min(len(a), len(b), int(n_q))
    if n <= 1:
        return float("nan")
    q = np.linspace(0.001, 0.999, n)
    aq = np.quantile(a, q)
    bq = np.quantile(b, q)
    return float(np.mean(np.abs(aq - bq)))


def sliced_w2_cloud(A: np.ndarray, B: np.ndarray, n_proj: int = 256, seed: int = 123, max_points: int = 5000) -> float:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A = A[np.all(np.isfinite(A), axis=1)] if A.ndim == 2 and len(A) else A
    B = B[np.all(np.isfinite(B), axis=1)] if B.ndim == 2 and len(B) else B
    if A.ndim != 2 or B.ndim != 2 or len(A) < 3 or len(B) < 3:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    if len(A) > max_points:
        A = A[rng.choice(len(A), size=max_points, replace=False)]
    if len(B) > max_points:
        B = B[rng.choice(len(B), size=max_points, replace=False)]
    d = A.shape[1]
    dirs = rng.normal(size=(int(n_proj), d))
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    qn = min(len(A), len(B), 1500)
    if qn < 3:
        return float("nan")
    qs = np.linspace(0.001, 0.999, qn)
    vals = []
    for v in dirs:
        aq = np.quantile(A @ v, qs)
        bq = np.quantile(B @ v, qs)
        vals.append(np.mean((aq - bq) ** 2))
    return float(np.sqrt(np.mean(vals)))


def build_return_graph_atoms(U_full: np.ndarray, hits: np.ndarray, whiten: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build return graph atoms (r_i, r_{i+1}) from augmented coordinates U_full.
    Returns graph atoms and return-time intervals K_i.
    """
    hits = np.asarray(hits, dtype=np.int64)
    if len(hits) < 3:
        return np.zeros((0, 2 * U_full.shape[1]), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    U_hits = U_full[hits]
    if whiten:
        mu = U_hits.mean(axis=0)
        sd = U_hits.std(axis=0) + 1e-6
        U_hits = (U_hits - mu[None, :]) / sd[None, :]
    graph = np.concatenate([U_hits[:-1], U_hits[1:]], axis=1)
    K = np.diff(hits).astype(np.float64)
    return graph, K


def build_return_graph_atoms_with_ref(
    U_full: np.ndarray,
    hits: np.ndarray,
    ref_mu: np.ndarray,
    ref_sd: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    hits = np.asarray(hits, dtype=np.int64)
    if len(hits) < 3:
        return np.zeros((0, 2 * U_full.shape[1]), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    U_hits = U_full[hits]
    U_hits = (U_hits - ref_mu[None, :]) / ref_sd[None, :]
    graph = np.concatenate([U_hits[:-1], U_hits[1:]], axis=1)
    K = np.diff(hits).astype(np.float64)
    return graph, K


def split_half_diagnostics(U_full: np.ndarray, hits: np.ndarray, seed: int) -> Dict[str, float]:
    hits = np.asarray(hits, dtype=np.int64)
    if len(hits) < 8:
        return {
            "split_graph_sw2": float("nan"),
            "split_tau_w1": float("nan"),
            "split_n_pairs_first": 0,
            "split_n_pairs_second": 0,
        }
    U_hits = U_full[hits]
    mu = U_hits.mean(axis=0)
    sd = U_hits.std(axis=0) + 1e-6
    U_hits_w = (U_hits - mu[None, :]) / sd[None, :]
    graph = np.concatenate([U_hits_w[:-1], U_hits_w[1:]], axis=1)
    K = np.diff(hits).astype(np.float64)
    m = len(graph) // 2
    g1, g2 = graph[:m], graph[m:]
    k1, k2 = K[:m], K[m:]
    return {
        "split_graph_sw2": sliced_w2_cloud(g1, g2, n_proj=256, seed=seed),
        "split_tau_w1": quantile_w1_1d(k1, k2),
        "split_n_pairs_first": int(len(g1)),
        "split_n_pairs_second": int(len(g2)),
    }


def bootstrap_stability(U_full: np.ndarray, hits: np.ndarray, seed: int, B: int = 20) -> Dict[str, float]:
    hits = np.asarray(hits, dtype=np.int64)
    if len(hits) < 8 or B <= 0:
        return {
            "bootstrap_graph_sw2_mean": float("nan"),
            "bootstrap_graph_sw2_std": float("nan"),
            "bootstrap_tau_w1_mean": float("nan"),
            "bootstrap_tau_w1_std": float("nan"),
        }
    rng = np.random.default_rng(int(seed))
    U_hits = U_full[hits]
    mu = U_hits.mean(axis=0)
    sd = U_hits.std(axis=0) + 1e-6
    U_hits_w = (U_hits - mu[None, :]) / sd[None, :]
    graph = np.concatenate([U_hits_w[:-1], U_hits_w[1:]], axis=1)
    K = np.diff(hits).astype(np.float64)
    n = len(graph)
    if n < 6:
        return {
            "bootstrap_graph_sw2_mean": float("nan"),
            "bootstrap_graph_sw2_std": float("nan"),
            "bootstrap_tau_w1_mean": float("nan"),
            "bootstrap_tau_w1_std": float("nan"),
        }
    graph_vals = []
    tau_vals = []
    size = max(3, n // 2)
    for b in range(int(B)):
        idx1 = rng.choice(n, size=size, replace=False)
        idx2 = rng.choice(n, size=size, replace=False)
        graph_vals.append(sliced_w2_cloud(graph[idx1], graph[idx2], n_proj=128, seed=seed + b))
        tau_vals.append(quantile_w1_1d(K[idx1], K[idx2]))
    gmu, gsd = mean_std(graph_vals)
    tmu, tsd = mean_std(tau_vals)
    return {
        "bootstrap_graph_sw2_mean": gmu,
        "bootstrap_graph_sw2_std": gsd,
        "bootstrap_tau_w1_mean": tmu,
        "bootstrap_tau_w1_std": tsd,
    }


def train_test_diagnostics(train_section: Dict[str, Any], test_applied: Optional[Dict[str, Any]], seed: int) -> Dict[str, float]:
    if test_applied is None:
        return {
            "test_N_hits": float("nan"),
            "train_test_graph_sw2": float("nan"),
            "train_test_tau_w1": float("nan"),
            "test_tau_mean_steps": float("nan"),
            "test_tau_cv": float("nan"),
        }
    train_hits = train_section["hits"]
    test_hits = test_applied["hits"]
    if len(train_hits) < 3 or len(test_hits) < 3:
        return {
            "test_N_hits": int(len(test_hits)),
            "train_test_graph_sw2": float("nan"),
            "train_test_tau_w1": float("nan"),
            "test_tau_mean_steps": float("nan"),
            "test_tau_cv": float("nan"),
        }
    U_train_hits = train_section["U_full"][train_hits]
    mu = U_train_hits.mean(axis=0)
    sd = U_train_hits.std(axis=0) + 1e-6
    g_train, k_train = build_return_graph_atoms_with_ref(train_section["U_full"], train_hits, mu, sd)
    g_test, k_test = build_return_graph_atoms_with_ref(test_applied["U_full"], test_hits, mu, sd)
    test_tau_mean = float(np.mean(k_test)) if len(k_test) else float("nan")
    test_tau_std = float(np.std(k_test)) if len(k_test) else float("nan")
    return {
        "test_N_hits": int(len(test_hits)),
        "train_test_graph_sw2": sliced_w2_cloud(g_train, g_test, n_proj=256, seed=seed),
        "train_test_tau_w1": quantile_w1_1d(k_train, k_test),
        "test_tau_mean_steps": test_tau_mean,
        "test_tau_cv": float(test_tau_std / (test_tau_mean + 1e-12)) if np.isfinite(test_tau_mean) else float("nan"),
    }


def compute_row_metrics(
    section: Dict[str, Any],
    seed: int,
    test_applied: Optional[Dict[str, Any]] = None,
    random_control: bool = False,
    bootstrap_B: int = 20,
) -> Dict[str, Any]:
    cfg: SectionConfig = section["cfg"]
    hits = section["hits"]
    speed = section["speed"]
    T = section["T"]
    min_t = section["min_t"]

    if random_control:
        # Same number of hits, no landmark event structure.
        hits_use = sample_random_hits(T=T, min_t=min_t, n_hits=len(hits), L_sec=cfg.L_sec, seed=seed + 99991)
        case_name = cfg.case + "__random_sameN"
        family = cfg.family + "/random_control"
        hit_lm_ids = np.asarray([], dtype=np.int64)
        hit_dists = np.asarray([], dtype=np.float64)
        visit_lens = np.asarray([], dtype=np.float64)
    else:
        hits_use = hits
        case_name = cfg.case
        family = cfg.family
        hit_lm_ids = section["hit_lm_ids"]
        hit_dists = section["hit_dists"]
        visit_lens = section["visit_lens"]

    K = np.diff(hits_use).astype(np.float64) if len(hits_use) >= 2 else np.asarray([], dtype=np.float64)
    N_hits = int(len(hits_use))
    N_pairs = int(max(0, len(hits_use) - 1))
    tau_mean = float(np.mean(K)) if len(K) else float("nan")
    tau_std = float(np.std(K)) if len(K) else float("nan")
    tau_cv = float(tau_std / (tau_mean + 1e-12)) if len(K) else float("nan")

    hit_speeds = speed[hits_use] if len(hits_use) else np.asarray([], dtype=np.float64)
    speed_eff = speed[section["idx_eff"]]
    hit_speed_mean = float(np.mean(hit_speeds)) if len(hit_speeds) else float("nan")
    hit_speed_quantile_mean = float(np.mean([np.mean(speed_eff <= s) for s in hit_speeds])) if len(hit_speeds) else float("nan")

    if len(hit_lm_ids) > 0:
        unique_lm = int(len(np.unique(hit_lm_ids)))
        unique_lm_frac = float(unique_lm / max(1, int(section["M_landmarks_actual"])))
        hit_dist_mean = float(np.mean(hit_dists)) if len(hit_dists) else float("nan")
    else:
        unique_lm = 0
        unique_lm_frac = float("nan")
        hit_dist_mean = float("nan")

    row: Dict[str, Any] = {
        "case": case_name,
        "family": family,
        "seed": int(seed),
        "q_speed": float(cfg.q_speed),
        "cand_frac": float(cfg.cand_frac),
        "M_landmarks": int(cfg.M_landmarks),
        "M_landmarks_actual": int(section["M_landmarks_actual"]),
        "L_sec": int(cfg.L_sec),
        "tau_delay": int(cfg.tau_delay),
        "dim_delay": int(cfg.dim_delay),
        "delay_ch": int(cfg.delay_ch),
        "eps_in_mult": float(cfg.eps_in_mult),
        "eps_out_mult": float(cfg.eps_out_mult),
        "eps_in": float(section["eps_in"]),
        "eps_out": float(section["eps_out"]),
        "nn_med": float(section["nn_med"]),
        "T_eff": int(section["T_eff"]),
        "pool_size": int(section["pool_size"]),
        "pool_frac": float(section["pool_frac"]),
        "cand_size": int(section["cand_size"]),
        "speed_thr": float(section["speed_thr"]),
        "N_hits": N_hits,
        "N_pairs": N_pairs,
        "tau_mean_steps": tau_mean,
        "tau_std_steps": tau_std,
        "tau_cv": tau_cv,
        "tau_median_steps": float(np.median(K)) if len(K) else float("nan"),
        "tau_min_steps": float(np.min(K)) if len(K) else float("nan"),
        "tau_max_steps": float(np.max(K)) if len(K) else float("nan"),
        "visit_len_mean": float(np.mean(visit_lens)) if len(visit_lens) else float("nan"),
        "visit_len_std": float(np.std(visit_lens)) if len(visit_lens) else float("nan"),
        "unique_landmarks_hit": unique_lm,
        "unique_landmark_frac": unique_lm_frac,
        "hit_dist_mean": hit_dist_mean,
        "hit_speed_mean": hit_speed_mean,
        "hit_speed_quantile_mean": hit_speed_quantile_mean,
    }

    row.update(split_half_diagnostics(section["U_full"], hits_use, seed=seed + 101))
    row.update(bootstrap_stability(section["U_full"], hits_use, seed=seed + 202, B=bootstrap_B))

    if (not random_control) and test_applied is not None:
        row.update(train_test_diagnostics(section, test_applied, seed=seed + 303))
    else:
        row.update({
            "test_N_hits": float("nan"),
            "train_test_graph_sw2": float("nan"),
            "train_test_tau_w1": float("nan"),
            "test_tau_mean_steps": float("nan"),
            "test_tau_cv": float("nan"),
        })

    # Simple pass flags. These are diagnostic, not theoretical guarantees.
    row["flag_enough_hits"] = int(N_hits >= 32)
    row["flag_nonconstant_tau"] = int(np.isfinite(tau_cv) and tau_cv >= 0.05)
    row["flag_landmark_coverage"] = int((not random_control) and np.isfinite(unique_lm_frac) and unique_lm_frac >= 0.10)
    row["flag_split_graph_finite"] = int(np.isfinite(row["split_graph_sw2"]))
    row["diagnostic_score"] = int(row["flag_enough_hits"] + row["flag_nonconstant_tau"] + row["flag_landmark_coverage"] + row["flag_split_graph_finite"])

    return row


# ============================================================
# Case generation and aggregation
# ============================================================
def unique_cases(cases: List[SectionConfig]) -> List[SectionConfig]:
    seen = set()
    out = []
    for c in cases:
        key = (c.case, c.family, c.q_speed, c.cand_frac, c.M_landmarks, c.L_sec, c.tau_delay,
               c.dim_delay, c.delay_ch, c.eps_in_mult, c.eps_out_mult)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def make_cases(args: argparse.Namespace) -> List[SectionConfig]:
    base = dict(
        q_speed=float(args.q_speed),
        cand_frac=float(args.cand_frac),
        M_landmarks=int(args.M_landmarks),
        L_sec=int(args.L_sec),
        tau_delay=int(args.tau_delay),
        dim_delay=int(args.dim_delay),
        delay_ch=int(args.delay_ch),
        eps_in_mult=float(args.eps_in_mult),
        eps_out_mult=float(args.eps_out_mult),
    )
    cases: List[SectionConfig] = []
    cases.append(SectionConfig(case="baseline", family="baseline", **base))

    preset = args.preset.lower().strip()
    if preset in {"minimal", "lorenz", "full"}:
        # speed threshold sweep
        for q in parse_list_float(args.q_sweep):
            cfg = dict(base)
            cfg["q_speed"] = float(q)
            cases.append(SectionConfig(case=f"q_speed={q:.2f}", family="q_speed", **cfg))

    if preset in {"lorenz", "full"}:
        # delay dimension sweep
        for m in parse_list_int(args.dim_delay_sweep):
            cfg = dict(base)
            cfg["dim_delay"] = int(m)
            cases.append(SectionConfig(case=f"dim_delay={m}", family="dim_delay", **cfg))

        # landmark count sweep
        for M in parse_list_int(args.M_sweep):
            cfg = dict(base)
            cfg["M_landmarks"] = int(M)
            cases.append(SectionConfig(case=f"M={M}", family="landmarks", **cfg))

        # threshold sweep
        for e in parse_list_float(args.eps_in_sweep):
            cfg = dict(base)
            cfg["eps_in_mult"] = float(e)
            cases.append(SectionConfig(case=f"eps_in_mult={e:.2f}", family="threshold", **cfg))

    if preset == "full":
        for cf in parse_list_float(args.cand_frac_sweep):
            cfg = dict(base)
            cfg["cand_frac"] = float(cf)
            cases.append(SectionConfig(case=f"cand_frac={cf:.2f}", family="cand_frac", **cfg))

        for L in parse_list_int(args.L_sec_sweep):
            cfg = dict(base)
            cfg["L_sec"] = int(L)
            cases.append(SectionConfig(case=f"L_sec={L}", family="L_sec", **cfg))

    return unique_cases(cases)


def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(rows) == 0:
        return []
    group_keys = [
        "case", "family", "q_speed", "cand_frac", "M_landmarks", "L_sec", "tau_delay",
        "dim_delay", "delay_ch", "eps_in_mult", "eps_out_mult",
    ]
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in rows:
        key = tuple(r.get(k, "") for k in group_keys)
        groups.setdefault(key, []).append(r)

    # Columns to aggregate as mean/std.
    metric_cols = [
        "N_hits", "N_pairs", "tau_mean_steps", "tau_std_steps", "tau_cv", "tau_median_steps",
        "tau_min_steps", "tau_max_steps", "pool_frac", "cand_size", "eps_in", "eps_out",
        "unique_landmarks_hit", "unique_landmark_frac", "hit_speed_quantile_mean", "hit_dist_mean",
        "visit_len_mean", "split_graph_sw2", "split_tau_w1", "bootstrap_graph_sw2_mean",
        "bootstrap_tau_w1_mean", "test_N_hits", "train_test_graph_sw2", "train_test_tau_w1",
        "test_tau_mean_steps", "test_tau_cv", "diagnostic_score",
    ]

    out = []
    for key, rs in groups.items():
        base = {k: v for k, v in zip(group_keys, key)}
        base["n_seeds"] = len(rs)
        for c in metric_cols:
            vals = [safe_float(r.get(c, float("nan"))) for r in rs]
            mu, sd = mean_std(vals)
            base[f"{c}_mean"] = mu
            base[f"{c}_std"] = sd
        out.append(base)

    def sort_key(r):
        fam_order = {
            "baseline": 0,
            "q_speed": 1,
            "dim_delay": 2,
            "landmarks": 3,
            "threshold": 4,
            "cand_frac": 5,
            "L_sec": 6,
        }
        fam = str(r.get("family", ""))
        return (fam_order.get(fam, 99), str(r.get("case", "")))

    return sorted(out, key=sort_key)


# ============================================================
# Output helpers
# ============================================================
def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
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
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})


def fmt(x: Any, nd: int = 3) -> str:
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(x):
        return "--"
    return f"{x:.{nd}f}"


def write_markdown_table(agg: List[Dict[str, Any]], path: str) -> None:
    cols = [
        "case", "N_hits_mean", "tau_mean_steps_mean", "tau_cv_mean",
        "unique_landmark_frac_mean", "split_graph_sw2_mean", "split_tau_w1_mean",
        "train_test_graph_sw2_mean", "train_test_tau_w1_mean",
    ]
    names = [
        "Case", "$N_{hit}$", "$\\bar K$", "CV($K$)",
        "Landmark cov.", "Split graph SW2", "Split $K$ W1",
        "Train-test graph SW2", "Train-test $K$ W1",
    ]
    with open(path, "w") as f:
        f.write("# Data-driven recurrence section diagnostics\n\n")
        f.write("| " + " | ".join(names) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for r in agg:
            vals = []
            for c in cols:
                if c == "case":
                    vals.append(str(r.get(c, "")))
                elif c == "N_hits_mean":
                    vals.append(f"{fmt(r.get(c), 1)} $\\pm$ {fmt(r.get('N_hits_std'), 1)}")
                elif c == "tau_mean_steps_mean":
                    vals.append(f"{fmt(r.get(c), 2)} $\\pm$ {fmt(r.get('tau_mean_steps_std'), 2)}")
                elif c == "tau_cv_mean":
                    vals.append(fmt(r.get(c), 3))
                elif c == "unique_landmark_frac_mean":
                    vals.append(fmt(r.get(c), 3))
                elif c == "split_graph_sw2_mean":
                    vals.append(f"{fmt(r.get(c), 3)} $\\pm$ {fmt(r.get('split_graph_sw2_std'), 3)}")
                elif c == "split_tau_w1_mean":
                    vals.append(f"{fmt(r.get(c), 2)} $\\pm$ {fmt(r.get('split_tau_w1_std'), 2)}")
                elif c == "train_test_graph_sw2_mean":
                    vals.append(f"{fmt(r.get(c), 3)} $\\pm$ {fmt(r.get('train_test_graph_sw2_std'), 3)}")
                elif c == "train_test_tau_w1_mean":
                    vals.append(f"{fmt(r.get(c), 2)} $\\pm$ {fmt(r.get('train_test_tau_w1_std'), 2)}")
                else:
                    vals.append(fmt(r.get(c), 3))
            f.write("| " + " | ".join(vals) + " |\n")


def write_latex_table(agg: List[Dict[str, Any]], path: str, max_rows: int = 30) -> None:
    with open(path, "w") as f:
        f.write(r"""
\begin{table*}[t]
\centering
\caption{Data-side diagnostics for the data-driven recurrence section on Lorenz63. Each row is computed before model training and is averaged over extractor seeds. The selected section should produce enough recurrence events, non-degenerate return intervals, broad landmark usage, and stable empirical return-graph statistics under split-half and train-test comparisons. Lower is better for graph and return-time distances.}
\label{tab:section_diagnostics_lorenz}
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccccccc}
\toprule
Case & $N_{\rm hit}$ & $\bar K$ & CV$(K)$ & Landmark cov. & Split graph & Split $K$ & Train-test graph & Train-test $K$ \\
\midrule
""")
        for r in agg[:max_rows]:
            case = str(r.get("case", "")).replace("_", r"\_")
            line = (
                f"{case} & "
                f"{fmt(r.get('N_hits_mean'), 1)} $\\pm$ {fmt(r.get('N_hits_std'), 1)} & "
                f"{fmt(r.get('tau_mean_steps_mean'), 2)} & "
                f"{fmt(r.get('tau_cv_mean'), 3)} & "
                f"{fmt(r.get('unique_landmark_frac_mean'), 3)} & "
                f"{fmt(r.get('split_graph_sw2_mean'), 3)} & "
                f"{fmt(r.get('split_tau_w1_mean'), 2)} & "
                f"{fmt(r.get('train_test_graph_sw2_mean'), 3)} & "
                f"{fmt(r.get('train_test_tau_w1_mean'), 2)} \\\\"
            )
            f.write(line + "\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table*}
""")


def plot_diagnostics(agg: List[Dict[str, Any]], out_dir: str) -> None:
    if not HAS_PLT or len(agg) == 0:
        return
    os.makedirs(out_dir, exist_ok=True)

    # Plot only cases in q_speed, dim_delay, landmarks, threshold families where x-axis is meaningful.
    specs = [
        ("q_speed", "q_speed", "q_speed"),
        ("dim_delay", "dim_delay", "dim_delay"),
        ("landmarks", "M_landmarks", "M_landmarks"),
        ("threshold", "eps_in_mult", "eps_in_mult"),
    ]
    y_metrics = [
        ("N_hits_mean", "$N_{hit}$"),
        ("tau_cv_mean", "CV(K)"),
        ("split_graph_sw2_mean", "split graph SW2"),
        ("train_test_graph_sw2_mean", "train-test graph SW2"),
    ]

    for fam, xkey, xlabel in specs:
        subset = [r for r in agg if str(r.get("family", "")) == fam]
        if len(subset) == 0:
            continue
        try:
            xs = np.asarray([float(r.get(xkey)) for r in subset], dtype=np.float64)
        except Exception:
            continue
        order = np.argsort(xs)
        subset = [subset[i] for i in order]
        xs = xs[order]
        for ykey, ylabel in y_metrics:
            ys = np.asarray([safe_float(r.get(ykey)) for r in subset], dtype=np.float64)
            if not np.any(np.isfinite(ys)):
                continue
            plt.figure(figsize=(5.5, 3.5), dpi=170)
            plt.plot(xs, ys, marker="o")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} vs {xlabel}")
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{fam}_{ykey}.png"))
            plt.close()


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, default="train_data.npy")
    p.add_argument("--test_path", type=str, default="test_data.npy")
    p.add_argument("--out_dir", type=str, default="./section_diagnostics")
    p.add_argument("--preset", type=str, default="lorenz", choices=["minimal", "lorenz", "full"])
    p.add_argument("--seeds", type=str, default="33,123,1000,13532,123456")

    # Base section config; matches your current Lorenz defaults.
    p.add_argument("--q_speed", type=float, default=0.3)
    p.add_argument("--cand_frac", type=float, default=0.5)
    p.add_argument("--M_landmarks", type=int, default=200)
    p.add_argument("--L_sec", type=int, default=5)
    p.add_argument("--tau_delay", type=int, default=1)
    p.add_argument("--dim_delay", type=int, default=5)
    p.add_argument("--delay_ch", type=int, default=0)
    p.add_argument("--eps_in_mult", type=float, default=0.35)
    p.add_argument("--eps_out_mult", type=float, default=1.8)

    # Sweeps. Keep small; this is a diagnostic, not a new benchmark.
    p.add_argument("--q_sweep", type=str, default="0.10,0.20,0.30,0.40")
    p.add_argument("--dim_delay_sweep", type=str, default="3,5,7,9")
    p.add_argument("--M_sweep", type=str, default="50,100,200,400")
    p.add_argument("--eps_in_sweep", type=str, default="0.25,0.35,0.50")
    p.add_argument("--cand_frac_sweep", type=str, default="0.25,0.50,0.75")
    p.add_argument("--L_sec_sweep", type=str, default="3,5,10")

    p.add_argument("--include_random_control", action="store_true",
                   help="Also report random same-N hit controls for each case. Not needed for the main table.")
    p.add_argument("--bootstrap_B", type=int, default=20)
    p.add_argument("--obs_cols", type=str, default="all",
                   help="Optional comma-separated observation columns. Use 'all' for all columns in the data file.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load_array(args.train_path)
    test_raw = None
    if args.test_path and Path(args.test_path).exists():
        test_raw = load_array(args.test_path)

    if args.obs_cols.strip().lower() != "all":
        cols = parse_list_int(args.obs_cols)
        train_raw = train_raw[:, cols]
        if test_raw is not None:
            test_raw = test_raw[:, cols]

    scaler = make_maxabs_scaler(train_raw)
    train = scaler.transform(train_raw).astype(np.float64)
    test = scaler.transform(test_raw).astype(np.float64) if test_raw is not None else None

    cases = make_cases(args)
    seeds = parse_list_int(args.seeds)

    print("=" * 88)
    print("Data-driven recurrence section diagnostics")
    print("=" * 88)
    print("train:", train_raw.shape, args.train_path)
    print("test :", None if test_raw is None else test_raw.shape, args.test_path)
    print("out_dir:", str(out_dir))
    print("preset:", args.preset)
    print("n_cases:", len(cases), "n_seeds:", len(seeds))
    print("include_random_control:", bool(args.include_random_control))
    print("=" * 88)

    per_seed_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for ci, cfg in enumerate(cases, 1):
        print(f"\n[{ci}/{len(cases)}] case={cfg.case} family={cfg.family}")
        for seed in seeds:
            try:
                section = construct_section(train, cfg, seed=seed)
                test_applied = apply_section_to_new_data(test, section) if test is not None else None
                row = compute_row_metrics(
                    section=section,
                    seed=seed,
                    test_applied=test_applied,
                    random_control=False,
                    bootstrap_B=int(args.bootstrap_B),
                )
                per_seed_rows.append(row)
                print(
                    f"  seed={seed:>6d} | hits={row['N_hits']:>4d} | "
                    f"Kmean={fmt(row['tau_mean_steps'], 2)} | CV={fmt(row['tau_cv'], 3)} | "
                    f"splitG={fmt(row['split_graph_sw2'], 3)} | testG={fmt(row['train_test_graph_sw2'], 3)}"
                )

                if args.include_random_control:
                    rrow = compute_row_metrics(
                        section=section,
                        seed=seed,
                        test_applied=None,
                        random_control=True,
                        bootstrap_B=int(args.bootstrap_B),
                    )
                    per_seed_rows.append(rrow)

            except Exception as e:
                msg = repr(e)
                print(f"  [ERROR] seed={seed}: {msg}")
                errors.append({"case": cfg.case, "family": cfg.family, "seed": seed, "error": msg, **asdict(cfg)})

    agg = aggregate_rows(per_seed_rows)

    per_seed_csv = out_dir / "section_diagnostics_per_seed.csv"
    agg_csv = out_dir / "section_diagnostics_aggregate.csv"
    md_path = out_dir / "section_diagnostics_table.md"
    tex_path = out_dir / "section_diagnostics_table.tex"
    err_path = out_dir / "section_diagnostics_errors.json"

    write_csv(per_seed_rows, str(per_seed_csv))
    write_csv(agg, str(agg_csv))
    write_markdown_table(agg, str(md_path))
    write_latex_table(agg, str(tex_path))
    with open(err_path, "w") as f:
        json.dump(errors, f, indent=2)

    plot_diagnostics(agg, str(out_dir))

    print("\n" + "=" * 88)
    print("Aggregate summary, compact")
    print("=" * 88)
    print(f"{'case':>22s} | {'N_hits':>14s} | {'Kmean':>8s} | {'CV(K)':>8s} | {'Lcov':>8s} | {'SplitG':>8s} | {'TestG':>8s}")
    print("-" * 88)
    for r in agg:
        print(
            f"{str(r.get('case','')):>22s} | "
            f"{fmt(r.get('N_hits_mean'),1)}±{fmt(r.get('N_hits_std'),1):<6s} | "
            f"{fmt(r.get('tau_mean_steps_mean'),2):>8s} | "
            f"{fmt(r.get('tau_cv_mean'),3):>8s} | "
            f"{fmt(r.get('unique_landmark_frac_mean'),3):>8s} | "
            f"{fmt(r.get('split_graph_sw2_mean'),3):>8s} | "
            f"{fmt(r.get('train_test_graph_sw2_mean'),3):>8s}"
        )

    print("\nSaved:")
    print("  per-seed CSV:", per_seed_csv)
    print("  aggregate CSV:", agg_csv)
    print("  markdown table:", md_path)
    print("  latex table:", tex_path)
    print("  errors:", err_path)
    print("Done.")


if __name__ == "__main__":
    main()
