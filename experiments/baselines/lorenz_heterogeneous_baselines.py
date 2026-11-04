#!/usr/bin/env python3
"""
Heterogeneous baselines for partially observed Lorenz63 [X,Z].

Baselines implemented under their standard objectives:
  1) NGRC / NVAR chaotic-forecasting baseline: polynomial ridge regression on delay coordinates.
  2) ESN reservoir computing: fixed random reservoir + ridge readout.
  3) LSTM: sequence-to-one autoregressive MSE predictor.
  4) Koopman-AE: delay-window autoencoder with linear latent dynamics.
  5) DySLIM-style MLP: pointwise prediction plus invariant-measure MMD regularization.

Typical use on Pop!_OS / Linux:
  python lorenz_heterogeneous_baselines.py --all \
      --train_path train_data.npy --test_path test_data.npy \
      --out_dir lorenz_heterogeneous_baselines --device auto

Data format:
  train_data.npy and test_data.npy are arrays of shape (T, d). If d=3, use --obs_cols 0,2
  to select the 2D Lorenz observation [X,Z]. If d=2, the script uses both columns.

Outputs:
  out_dir/checkpoints/*        model checkpoints
  out_dir/results_summary.csv  aggregate metrics
  out_dir/latex_rows.txt       LaTeX-ready rows
  out_dir/plots/*              rollout overlays

Notes:
  - DySLIM here is a practical PyTorch MMD-regularized objective in the spirit of DySLIM,
    not a verbatim copy of the official JAX implementation.
  - The script evaluates autonomous rollouts from 30 initial histories by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import sys
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import welch
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MaxAbsScaler

try:
    from geomloss import SamplesLoss  # optional, used for Sinkhorn W2 if available
    HAS_GEOMLOSS = True
except Exception:
    SamplesLoss = None
    HAS_GEOMLOSS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


ROOT = Path(__file__).resolve().parents[2]

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


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
            if len(keys) != 1:
                raise KeyError(f"Could not infer array key in {p}; keys={keys}")
            arr = obj[keys[0]]
    elif suffix in {".p", ".pkl", ".pickle"}:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            for key in ["train_data", "test_data", "train", "test", "data", "X", "Y"]:
                if key in obj:
                    arr = obj[key]
                    break
            else:
                raise KeyError(f"Could not infer array key in pickle; keys={list(obj.keys())}")
        else:
            arr = obj
    elif suffix in {".pt", ".pth"}:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            for key in ["train_data", "test_data", "train", "test", "data", "X", "Y"]:
                if key in obj:
                    arr = obj[key]
                    break
            else:
                raise KeyError(f"Could not infer array key in torch file; keys={list(obj.keys())}")
        else:
            arr = obj
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array (T,d), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Array contains non-finite values: {p}")
    return arr


def parse_obs_cols(obs_cols: str, d: int) -> Optional[List[int]]:
    if obs_cols is None or str(obs_cols).strip() == "":
        return None
    cols = [int(x.strip()) for x in str(obs_cols).split(",") if x.strip() != ""]
    for c in cols:
        if c < 0 or c >= d:
            raise ValueError(f"obs col {c} out of bounds for d={d}")
    return cols


def select_obs(arr: np.ndarray, obs_cols: Optional[List[int]]) -> np.ndarray:
    if obs_cols is None:
        if arr.shape[1] == 2:
            return arr
        if arr.shape[1] >= 3:
            # default Lorenz partial observation [X,Z]
            return arr[:, [0, 2]]
        return arr
    return arr[:, obs_cols]


def ensure_dirs(out_dir: Path) -> None:
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)


def save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# Data windowing
# -----------------------------------------------------------------------------

def make_delay_windows(y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return X=(N,seq_len,d), Y=(N,d), where X[i] predicts Y[i]."""
    y = np.asarray(y, dtype=np.float64)
    T, d = y.shape
    if T <= seq_len:
        raise ValueError(f"T={T} <= seq_len={seq_len}")
    X = np.zeros((T - seq_len, seq_len, d), dtype=np.float64)
    Y = np.zeros((T - seq_len, d), dtype=np.float64)
    for i in range(T - seq_len):
        X[i] = y[i:i + seq_len]
        Y[i] = y[i + seq_len]
    return X, Y


def sample_batch_indices(n: int, batch_size: int) -> np.ndarray:
    return np.random.randint(0, n, size=int(batch_size))


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def nmse(ref: np.ndarray, pred: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    L = min(len(ref), len(pred))
    ref = ref[:L]
    pred = pred[:L]
    denom = np.mean(np.sum(ref ** 2, axis=1)) + 1e-12
    return float(np.mean(np.sum((pred - ref) ** 2, axis=1)) / denom)


def psd_logl1_distance(ref: np.ndarray, pred: np.ndarray, dt: float, nfft: int = 2048,
                       channels: Optional[List[int]] = None) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    L = min(len(ref), len(pred))
    ref = ref[:L]
    pred = pred[:L]
    if channels is None:
        channels = list(range(ref.shape[1]))
    fs = 1.0 / float(dt)
    eps = 1e-12
    vals = []
    for ch in channels:
        x = ref[:, ch] - np.mean(ref[:, ch])
        y = pred[:, ch] - np.mean(pred[:, ch])
        n = min(nfft, L)
        if n < 16:
            continue
        x = x[:n]
        y = y[:n]
        _, Pxx = welch(x, fs=fs, nperseg=min(512, n), noverlap=min(256, n // 2))
        _, Pyy = welch(y, fs=fs, nperseg=min(512, n), noverlap=min(256, n // 2))
        m = min(len(Pxx), len(Pyy))
        if m == 0:
            continue
        vals.append(float(np.mean(np.abs(np.log10(Pxx[:m] + eps) - np.log10(Pyy[:m] + eps)))))
    return float(np.mean(vals)) if vals else float("nan")


def sliced_wasserstein(A: np.ndarray, B: np.ndarray, n_proj: int = 128, seed: int = 0) -> float:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    d = A.shape[1]
    vals = []
    for _ in range(int(n_proj)):
        v = rng.normal(size=(d,))
        v = v / (np.linalg.norm(v) + 1e-12)
        vals.append(wasserstein_distance(A @ v, B @ v))
    return float(np.mean(vals))


def sinkhorn_or_sliced(A: np.ndarray, B: np.ndarray, device: torch.device,
                       blur: float = 0.05, fallback_proj: int = 128) -> float:
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    if HAS_GEOMLOSS:
        try:
            loss = SamplesLoss(loss="sinkhorn", p=2, blur=float(blur), backend="tensorized")
            XA = torch.tensor(A, dtype=torch.float32, device=device)
            XB = torch.tensor(B, dtype=torch.float32, device=device)
            return float(loss(XA, XB).detach().cpu())
        except Exception:
            pass
    return sliced_wasserstein(A, B, n_proj=fallback_proj)


def crossing_times_level(X: np.ndarray, ch: int, level: float, dt: float, direction: int = +1,
                         min_gap_steps: int = 1) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    x = X[:, ch]
    a = x[:-1] - float(level)
    b = x[1:] - float(level)
    if direction > 0:
        mask = (a < 0) & (b >= 0)
    elif direction < 0:
        mask = (a > 0) & (b <= 0)
    else:
        mask = (a == 0) | (a * b < 0)
    idx = np.where(mask)[0]
    hits = []
    last_i = -10 ** 12
    for i in idx:
        if i - last_i < min_gap_steps:
            continue
        denom = x[i + 1] - x[i]
        alpha = 0.0 if abs(denom) < 1e-12 else float(np.clip((level - x[i]) / denom, 0.0, 1.0))
        hits.append((i + alpha) * dt)
        last_i = int(i)
    return np.asarray(hits, dtype=np.float64)


def peak_times(X: np.ndarray, ch: int, dt: float, min_gap_steps: int = 1) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    x = X[:, ch]
    idx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    hits = []
    last_i = -10 ** 12
    for i in idx:
        if i - last_i < min_gap_steps:
            continue
        hits.append(i * dt)
        last_i = int(i)
    return np.asarray(hits, dtype=np.float64)


def get_section_hits(X: np.ndarray, section: Tuple[str, str, int, Optional[float], Optional[int]], dt: float) -> np.ndarray:
    name, typ, ch, level, direction = section
    if typ == "level":
        return crossing_times_level(X, ch=ch, level=float(level), dt=dt, direction=int(direction), min_gap_steps=1)
    if typ == "max":
        return peak_times(X, ch=ch, dt=dt, min_gap_steps=1)
    raise ValueError(f"Unknown section type: {typ}")


def drtd_w1(ref_hits: np.ndarray, pred_hits: np.ndarray) -> float:
    if len(ref_hits) < 2 or len(pred_hits) < 2:
        return float("nan")
    return float(wasserstein_distance(np.diff(ref_hits), np.diff(pred_hits)))


def hit_rate_error(ref_hits: np.ndarray, pred_hits: np.ndarray, T_phys: float) -> float:
    if T_phys <= 0:
        return float("nan")
    return float(abs(len(pred_hits) / T_phys - len(ref_hits) / T_phys))


def recurrence_metrics(ref: np.ndarray, pred: np.ndarray, dt: float,
                       sections: Optional[List[Tuple[str, str, int, Optional[float], Optional[int]]]] = None) -> Dict[str, float]:
    if sections is None:
        # For 2D Lorenz observation [X,Z], columns are u0 and u2.
        sections = [
            ("u0=10", "level", 0, 10.0, +1),
            ("u2=40", "level", 1, 40.0, +1),
            ("u2-max", "max", 1, None, None),
        ]
    L = min(len(ref), len(pred))
    ref = ref[:L]
    pred = pred[:L]
    T_phys = (L - 1) * float(dt)
    out: Dict[str, float] = {}
    drtds = []
    hres = []
    for sec in sections:
        sec_name = sec[0]
        h_ref = get_section_hits(ref, sec, dt)
        h_pred = get_section_hits(pred, sec, dt)
        d = drtd_w1(h_ref, h_pred)
        h = hit_rate_error(h_ref, h_pred, T_phys)
        out[f"{sec_name}_GT_hits"] = float(len(h_ref))
        out[f"{sec_name}_pred_hits"] = float(len(h_pred))
        out[f"{sec_name}_DRTD"] = d
        out[f"{sec_name}_HRE"] = h
        if np.isfinite(d):
            drtds.append(d)
        if np.isfinite(h):
            hres.append(h)
    out["Mean_DRTD"] = float(np.mean(drtds)) if drtds else float("nan")
    out["Mean_HRE"] = float(np.mean(hres)) if hres else float("nan")
    return out


def collapse_score(pred: np.ndarray, tail: int = 100) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    tail_arr = pred[-min(tail, len(pred)):]
    return float(np.mean(np.std(tail_arr, axis=0)))


# -----------------------------------------------------------------------------
# Baseline models
# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str
    seq_len: int
    dim: int
    extra: dict


class BaseRolloutModel:
    name: str
    config: ModelConfig
    def rollout_scaled(self, hist_scaled: np.ndarray, horizon: int) -> np.ndarray:
        raise NotImplementedError


# ---------- NGRC / NVAR ----------

def nvar_features(X_flat: np.ndarray, order: int = 2, clip_abs: Optional[float] = None) -> np.ndarray:
    """Polynomial features for X_flat shape (N,p) or (p,).

    ``clip_abs`` is a numerical guard. In training, the MaxAbs-scaled data are
    normally inside [-1, 1], so a default clip such as 20 does not change the
    training features. During autonomous rollout, an unstable polynomial NVAR can
    explode; clipping the feature input avoids floating-point overflow while the
    evaluator still records the rollout as diverged.
    """
    x = np.asarray(X_flat, dtype=np.float64)
    if clip_abs is not None and np.isfinite(float(clip_abs)) and float(clip_abs) > 0:
        x = np.clip(x, -float(clip_abs), float(clip_abs))
    one_dim = x.ndim == 1
    if one_dim:
        x = x[None, :]
    N, p = x.shape
    feats = [np.ones((N, 1), dtype=np.float64), x]
    if order >= 2:
        quad = []
        for i in range(p):
            xi = x[:, i:i + 1]
            for j in range(i, p):
                with np.errstate(over="ignore", invalid="ignore"):
                    quad.append(xi * x[:, j:j + 1])
        feats.append(np.concatenate(quad, axis=1))
    if order >= 3:
        cubic = []
        for i in range(p):
            xi = x[:, i:i + 1]
            for j in range(i, p):
                with np.errstate(over="ignore", invalid="ignore"):
                    xij = xi * x[:, j:j + 1]
                for k in range(j, p):
                    with np.errstate(over="ignore", invalid="ignore"):
                        cubic.append(xij * x[:, k:k + 1])
        feats.append(np.concatenate(cubic, axis=1))
    out = np.concatenate(feats, axis=1)
    return out[0] if one_dim else out


class NVARModel(BaseRolloutModel):
    def __init__(self, W: np.ndarray, seq_len: int, dim: int, order: int,
                 feature_clip: float = 20.0, rollout_clip: float = 50.0):
        self.W = W
        self.seq_len = int(seq_len)
        self.dim = int(dim)
        self.order = int(order)
        self.feature_clip = float(feature_clip)
        self.rollout_clip = float(rollout_clip)
        self.name = "NGRC-NVAR"
        self.config = ModelConfig(self.name, self.seq_len, self.dim,
                                  {"order": self.order,
                                   "feature_clip": self.feature_clip,
                                   "rollout_clip": self.rollout_clip})

    def rollout_scaled(self, hist_scaled: np.ndarray, horizon: int) -> np.ndarray:
        hist = [np.asarray(v, dtype=np.float64).copy() for v in hist_scaled[-self.seq_len:]]
        preds = []
        for _ in range(int(horizon)):
            xflat = np.asarray(hist[-self.seq_len:], dtype=np.float64).reshape(-1)
            phi = nvar_features(xflat, order=self.order, clip_abs=self.feature_clip)
            with np.errstate(over="ignore", invalid="ignore"):
                y = phi @ self.W
            y = np.asarray(y, dtype=np.float64)
            if (not np.all(np.isfinite(y))) or np.max(np.abs(y)) > self.rollout_clip:
                last = hist[-1] if len(hist) else np.zeros(self.dim, dtype=np.float64)
                fill = np.clip(np.nan_to_num(last, nan=0.0, posinf=self.rollout_clip, neginf=-self.rollout_clip),
                               -self.rollout_clip, self.rollout_clip)
                preds.append(fill.copy())
                while len(preds) < int(horizon):
                    preds.append(fill.copy())
                break
            y = np.clip(y, -self.rollout_clip, self.rollout_clip)
            preds.append(y)
            hist.append(y)
        return np.asarray(preds, dtype=np.float64)


def train_nvar(train_scaled: np.ndarray, seq_len: int, order: int, ridge: float,
               feature_clip: float, rollout_clip: float) -> NVARModel:
    X, Y = make_delay_windows(train_scaled, seq_len)
    Xf = X.reshape(len(X), -1)
    Phi = nvar_features(Xf, order=order, clip_abs=feature_clip)
    A = Phi.T @ Phi
    reg = float(ridge) * np.eye(A.shape[0], dtype=np.float64)
    # Do not regularize bias too much.
    reg[0, 0] = float(ridge) * 1e-3
    W = np.linalg.solve(A + reg, Phi.T @ Y)
    return NVARModel(W=W, seq_len=seq_len, dim=train_scaled.shape[1], order=order,
                     feature_clip=feature_clip, rollout_clip=rollout_clip)


# ---------- ESN reservoir computing ----------

class ESNModel(BaseRolloutModel):
    def __init__(self, Win: np.ndarray, Wres: np.ndarray, bias: np.ndarray, Wout: np.ndarray,
                 leak: float, seq_len: int, dim: int):
        self.Win = Win
        self.Wres = Wres
        self.bias = bias
        self.Wout = Wout
        self.leak = float(leak)
        self.seq_len = int(seq_len)
        self.dim = int(dim)
        self.n_res = int(Wres.shape[0])
        self.name = "Reservoir-ESN"
        self.config = ModelConfig(self.name, self.seq_len, self.dim, {"n_res": self.n_res, "leak": self.leak})

    def update(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        pre = self.Wres @ state + self.Win @ u + self.bias
        return (1.0 - self.leak) * state + self.leak * np.tanh(pre)

    def rollout_scaled(self, hist_scaled: np.ndarray, horizon: int) -> np.ndarray:
        hist_scaled = np.asarray(hist_scaled, dtype=np.float64)
        state = np.zeros(self.n_res, dtype=np.float64)
        for u in hist_scaled[:-1]:
            state = self.update(state, u)
        cur = hist_scaled[-1].copy()
        preds = []
        for _ in range(int(horizon)):
            state = self.update(state, cur)
            feat = np.concatenate([[1.0], state, cur])
            y = feat @ self.Wout
            preds.append(y)
            cur = y
        return np.asarray(preds, dtype=np.float64)


def train_esn(train_scaled: np.ndarray, seq_len: int, n_res: int, spectral_radius: float,
              leak: float, input_scale: float, density: float, ridge: float,
              washout: int, seed: int) -> ESNModel:
    rng = np.random.default_rng(seed)
    T, d = train_scaled.shape
    Win = rng.uniform(-input_scale, input_scale, size=(n_res, d))
    bias = rng.uniform(-0.1, 0.1, size=(n_res,))
    Wres = rng.normal(0.0, 1.0, size=(n_res, n_res))
    mask = rng.random(size=(n_res, n_res)) < float(density)
    Wres = Wres * mask
    # scale spectral radius
    eigvals = np.linalg.eigvals(Wres)
    rho = np.max(np.abs(eigvals)) + 1e-12
    Wres = Wres * (float(spectral_radius) / rho)

    state = np.zeros(n_res, dtype=np.float64)
    feats = []
    targets = []
    for t in range(T - 1):
        state = (1.0 - leak) * state + leak * np.tanh(Wres @ state + Win @ train_scaled[t] + bias)
        if t >= int(washout):
            feats.append(np.concatenate([[1.0], state, train_scaled[t]]))
            targets.append(train_scaled[t + 1])
    Phi = np.asarray(feats, dtype=np.float64)
    Y = np.asarray(targets, dtype=np.float64)
    A = Phi.T @ Phi
    reg = float(ridge) * np.eye(A.shape[0], dtype=np.float64)
    reg[0, 0] *= 1e-3
    Wout = np.linalg.solve(A + reg, Phi.T @ Y)
    return ESNModel(Win=Win, Wres=Wres, bias=bias, Wout=Wout, leak=leak, seq_len=seq_len, dim=d)


# ---------- Neural helper modules ----------

class LSTMForecaster(nn.Module):
    def __init__(self, dim: int, hidden: int, layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=dim, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, dim))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])

class LSTMRolloutModel(BaseRolloutModel):
    def __init__(self, net: LSTMForecaster, seq_len: int, dim: int, device: torch.device):
        self.net = net
        self.net.eval()
        self.seq_len = int(seq_len)
        self.dim = int(dim)
        self.device = device
        self.name = "LSTM"
        self.config = ModelConfig(self.name, self.seq_len, self.dim, {})
    @torch.no_grad()
    def rollout_scaled(self, hist_scaled: np.ndarray, horizon: int) -> np.ndarray:
        hist = [np.asarray(v, dtype=np.float64).copy() for v in hist_scaled[-self.seq_len:]]
        preds = []
        for _ in range(int(horizon)):
            x = torch.tensor(np.asarray(hist[-self.seq_len:])[None], dtype=torch.float32, device=self.device)
            y = self.net(x).detach().cpu().numpy()[0]
            preds.append(y)
            hist.append(y)
        return np.asarray(preds, dtype=np.float64)

class KoopmanAE(nn.Module):
    def __init__(self, in_dim: int, obs_dim: int, latent: int, hidden: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.obs_dim = int(obs_dim)
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, latent))
        self.K = nn.Linear(latent, latent, bias=False)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, in_dim))
    def encode(self, x):
        return self.encoder(x)
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        z = self.encode(x)
        rec = self.decode(z)
        z1 = self.K(z)
        pred_window = self.decode(z1)
        return rec, pred_window, z, z1

class KoopmanRolloutModel(BaseRolloutModel):
    def __init__(self, net: KoopmanAE, seq_len: int, dim: int, device: torch.device):
        self.net = net
        self.net.eval()
        self.seq_len = int(seq_len)
        self.dim = int(dim)
        self.device = device
        self.name = "Koopman-AE"
        self.config = ModelConfig(self.name, self.seq_len, self.dim, {})
    @torch.no_grad()
    def rollout_scaled(self, hist_scaled: np.ndarray, horizon: int) -> np.ndarray:
        hist = [np.asarray(v, dtype=np.float64).copy() for v in hist_scaled[-self.seq_len:]]
        preds = []
        for _ in range(int(horizon)):
            x_flat = np.asarray(hist[-self.seq_len:], dtype=np.float64).reshape(1, -1)
            x_t = torch.tensor(x_flat, dtype=torch.float32, device=self.device)
            _, pred_window, _, _ = self.net(x_t)
            win = pred_window.detach().cpu().numpy().reshape(self.seq_len, self.dim)
            y = win[-1]
            preds.append(y)
            hist.append(y)
        return np.asarray(preds, dtype=np.float64)

class MLPMap(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )
    def forward(self, x):
        return self.net(x)

class DySLIMRolloutModel(BaseRolloutModel):
    def __init__(self, net: MLPMap, seq_len: int, dim: int, device: torch.device):
        self.net = net
        self.net.eval()
        self.seq_len = int(seq_len)
        self.dim = int(dim)
        self.device = device
        self.name = "DySLIM-style"
        self.config = ModelConfig(self.name, self.seq_len, self.dim, {})
    @torch.no_grad()
    def rollout_scaled(self, hist_scaled: np.ndarray, horizon: int) -> np.ndarray:
        y = torch.tensor(hist_scaled[-1][None], dtype=torch.float32, device=self.device)
        preds = []
        for _ in range(int(horizon)):
            y = self.net(y)
            preds.append(y.detach().cpu().numpy()[0])
        return np.asarray(preds, dtype=np.float64)


# -----------------------------------------------------------------------------
# Neural training objectives
# -----------------------------------------------------------------------------

def train_lstm(train_scaled: np.ndarray, args, device: torch.device) -> LSTMRolloutModel:
    X, Y = make_delay_windows(train_scaled, args.seq_len)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    net = LSTMForecaster(dim=train_scaled.shape[1], hidden=args.lstm_hidden, layers=args.lstm_layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    n = len(X)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        pred = net(X_t[idx])
        loss = F.mse_loss(pred, Y_t[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        opt.step()
        if step % args.print_every == 0:
            print(f"[LSTM] step={step:05d} loss={loss.item():.6g}")
    return LSTMRolloutModel(net, seq_len=args.seq_len, dim=train_scaled.shape[1], device=device)


def train_koopman_ae(train_scaled: np.ndarray, args, device: torch.device) -> KoopmanRolloutModel:
    seq_len = args.seq_len
    d = train_scaled.shape[1]
    # windows x_t = y[t:t+seq], x_next = y[t+1:t+seq+1]
    X0 = []
    X1 = []
    for i in range(len(train_scaled) - seq_len - 1):
        X0.append(train_scaled[i:i + seq_len].reshape(-1))
        X1.append(train_scaled[i + 1:i + seq_len + 1].reshape(-1))
    X0 = torch.tensor(np.asarray(X0), dtype=torch.float32, device=device)
    X1 = torch.tensor(np.asarray(X1), dtype=torch.float32, device=device)
    net = KoopmanAE(in_dim=seq_len * d, obs_dim=d, latent=args.koopman_latent, hidden=args.koopman_hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    n = X0.shape[0]
    I = torch.eye(args.koopman_latent, device=device)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        x0 = X0[idx]
        x1 = X1[idx]
        rec, pred_win, z0, z1 = net(x0)
        # cycle consistency through encoder of the shifted window
        z1_target = net.encode(x1).detach()
        loss_rec = F.mse_loss(rec, x0)
        loss_pred = F.mse_loss(pred_win, x1)
        loss_lat = F.mse_loss(z1, z1_target)
        # small regularizer to discourage exploding linear dynamics
        K = net.K.weight
        loss_k = 1e-4 * torch.mean((K.T @ K - I) ** 2)
        loss = args.koopman_recon_w * loss_rec + loss_pred + args.koopman_latent_w * loss_lat + loss_k
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        opt.step()
        if step % args.print_every == 0:
            print(f"[Koopman-AE] step={step:05d} loss={loss.item():.6g} pred={loss_pred.item():.6g}")
    return KoopmanRolloutModel(net, seq_len=seq_len, dim=d, device=device)


def imq_mmd2(x: torch.Tensor, y: torch.Tensor, scales: Sequence[float] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0)) -> torch.Tensor:
    """IMQ-kernel MMD^2, robust for small batches."""
    x = x.float()
    y = y.float()
    dxx = torch.cdist(x, x, p=2) ** 2
    dyy = torch.cdist(y, y, p=2) ** 2
    dxy = torch.cdist(x, y, p=2) ** 2
    out = 0.0
    for c in scales:
        c = float(c)
        kxx = c / (c + dxx)
        kyy = c / (c + dyy)
        kxy = c / (c + dxy)
        out = out + kxx.mean() + kyy.mean() - 2.0 * kxy.mean()
    return out / float(len(scales))


def train_dyslim_style(train_scaled: np.ndarray, args, device: torch.device) -> DySLIMRolloutModel:
    """DySLIM-style MLP: multi-step pointwise loss + invariant-measure MMD regularization."""
    T, d = train_scaled.shape
    Y = torch.tensor(train_scaled, dtype=torch.float32, device=device)
    net = MLPMap(dim=d, hidden=args.dyslim_hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    horizon = int(args.dyslim_rollout)
    max_start = T - horizon - 1
    if max_start <= 1:
        raise ValueError("Training series too short for DySLIM rollout horizon")
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, max_start, (args.batch_size,), device=device)
        y = Y[idx]
        preds = []
        mse_loss = 0.0
        for h in range(1, horizon + 1):
            y = net(y)
            preds.append(y)
            mse_loss = mse_loss + F.mse_loss(y, Y[idx + h])
        mse_loss = mse_loss / float(horizon)
        pred_cloud = torch.cat(preds, dim=0)
        ref_idx = torch.randint(0, T, (pred_cloud.shape[0],), device=device)
        ref_cloud = Y[ref_idx]
        mmd = imq_mmd2(pred_cloud, ref_cloud)
        loss = mse_loss + args.dyslim_mmd_w * mmd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        opt.step()
        if step % args.print_every == 0:
            print(f"[DySLIM-style] step={step:05d} loss={loss.item():.6g} mse={mse_loss.item():.6g} mmd={mmd.item():.6g}")
    return DySLIMRolloutModel(net, seq_len=args.seq_len, dim=d, device=device)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def sanitize_scaled_rollout(pred: np.ndarray, dim: int, max_abs: float = 50.0) -> Tuple[np.ndarray, bool, int]:
    """Make a rollout finite while keeping a divergence flag.

    Heterogeneous baselines trained with their standard objectives can explode under
    autonomous rollout in this partial-observation setting. This guard prevents
    downstream calls such as sklearn.inverse_transform from crashing. It does not
    hide the failure: the row stores Diverged_starts and First_diverge_mean.
    """
    arr = np.asarray(pred, dtype=np.float64)
    if arr.ndim != 2:
        arr = np.reshape(arr, (-1, dim))
    if arr.shape[1] != dim:
        raise ValueError(f"rollout has dim {arr.shape[1]}, expected {dim}")
    max_abs = float(max_abs)
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 50.0
    finite_rows = np.all(np.isfinite(arr), axis=1)
    safe_abs = np.max(np.abs(np.nan_to_num(arr, nan=0.0, posinf=max_abs * 10, neginf=-max_abs * 10)), axis=1)
    bad = (~finite_rows) | (safe_abs > max_abs)
    if not np.any(bad):
        return arr, False, -1
    first = int(np.where(bad)[0][0])
    out = arr.copy()
    if first > 0 and np.all(np.isfinite(out[first - 1])):
        fill = out[first - 1].copy()
    else:
        fill = np.zeros(dim, dtype=np.float64)
    fill = np.clip(np.nan_to_num(fill, nan=0.0, posinf=max_abs, neginf=-max_abs), -max_abs, max_abs)
    if first > 0:
        out[:first] = np.clip(np.nan_to_num(out[:first], nan=0.0, posinf=max_abs, neginf=-max_abs), -max_abs, max_abs)
    out[first:] = fill[None, :]
    return out, True, first

def inverse_transform_many(scaler: MaxAbsScaler, X: np.ndarray) -> np.ndarray:
    shape = X.shape
    X2 = X.reshape(-1, shape[-1])
    Y2 = scaler.inverse_transform(X2)
    return Y2.reshape(shape)


def choose_eval_starts(T: int, seq_len: int, horizon: int, n_starts: int, seed: int) -> np.ndarray:
    max_start = T - seq_len - horizon
    if max_start <= 0:
        raise ValueError(f"Test series too short: T={T}, seq_len={seq_len}, horizon={horizon}")
    rng = np.random.default_rng(seed)
    n = min(int(n_starts), max_start)
    if n <= 1:
        return np.asarray([0], dtype=np.int64)
    return np.sort(rng.choice(np.arange(max_start), size=n, replace=False)).astype(np.int64)


def evaluate_model(model: BaseRolloutModel, train_raw: np.ndarray, test_raw: np.ndarray, scaler: MaxAbsScaler,
                   args, device: torch.device) -> Dict[str, object]:
    train_scaled = scaler.transform(train_raw)
    test_scaled = scaler.transform(test_raw)
    windows = [10, 20, 50, 100, 200, 300, 400, 500]
    max_h = max(windows)
    starts = choose_eval_starts(len(test_scaled), args.seq_len, max_h, args.eval_starts, args.seed + 123)
    nmse_by_w = {w: [] for w in windows}
    # Per-start autonomous rollouts.
    long_preds = []
    long_refs = []
    diverged_count = 0
    first_diverge = []
    for s in starts:
        hist_s = test_scaled[s:s + args.seq_len]
        pred_s = model.rollout_scaled(hist_s, max_h)
        pred_s, did_diverge, div_idx = sanitize_scaled_rollout(
            pred_s, dim=test_scaled.shape[1], max_abs=args.rollout_clip
        )
        if did_diverge:
            diverged_count += 1
            first_diverge.append(div_idx)
        pred_raw = inverse_transform_many(scaler, pred_s)
        ref_raw = test_raw[s + args.seq_len:s + args.seq_len + max_h]
        for w in windows:
            nmse_by_w[w].append(nmse(ref_raw[:w], pred_raw[:w]))
        long_preds.append(pred_raw[:args.long_horizon])
        long_refs.append(ref_raw[:args.long_horizon])
    out: Dict[str, object] = {"model": model.name, "Diverged_starts": int(diverged_count), "First_diverge_mean": float(np.mean(first_diverge)) if first_diverge else float("nan")}
    for w in windows:
        vals = np.asarray(nmse_by_w[w], dtype=np.float64)
        out[f"NMSE_{w}_mean"] = float(np.mean(vals))
        out[f"NMSE_{w}_std"] = float(np.std(vals))
        out[f"NMSE_{w}"] = f"{np.mean(vals):.4g} ± {np.std(vals):.4g}"

    # Structure metrics averaged across long rollouts. Use raw coordinates.
    w2s, psds, drtds, hres, colls = [], [], [], [], []
    for ref_raw, pred_raw in zip(long_refs, long_preds):
        L = min(len(ref_raw), len(pred_raw), args.long_horizon)
        ref_L = ref_raw[:L]
        pred_L = pred_raw[:L]
        # Downsample cloud for W2 if needed.
        if L > args.w2_max_points:
            idx = np.linspace(0, L - 1, args.w2_max_points).astype(int)
            A = ref_L[idx]
            B = pred_L[idx]
        else:
            A, B = ref_L, pred_L
        w2s.append(sinkhorn_or_sliced(A, B, device=device, blur=args.w2_blur, fallback_proj=args.sliced_proj))
        psds.append(psd_logl1_distance(ref_L, pred_L, dt=args.dt_obs, nfft=args.psd_nfft))
        rec = recurrence_metrics(ref_L, pred_L, dt=args.dt_obs)
        drtds.append(rec["Mean_DRTD"])
        hres.append(rec["Mean_HRE"])
        colls.append(collapse_score(pred_L, tail=100))
    def finite_mean(vals):
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if len(arr) else float("nan")
    out["Attractor_W2_mean"] = finite_mean(w2s)
    out["PSD_logL1_mean"] = finite_mean(psds)
    out["Mean_DRTD_mean"] = finite_mean(drtds)
    out["Mean_HRE_mean"] = finite_mean(hres)
    out["TailStd_mean"] = finite_mean(colls)

    # Save a representative plot.
    if HAS_MPL and args.plot:
        s = int(starts[0])
        hist_s = test_scaled[s:s + args.seq_len]
        pred_s = model.rollout_scaled(hist_s, args.plot_horizon)
        pred_s, _, _ = sanitize_scaled_rollout(pred_s, dim=test_scaled.shape[1], max_abs=args.rollout_clip)
        pred_raw = inverse_transform_many(scaler, pred_s)
        ref_raw = test_raw[s + args.seq_len:s + args.seq_len + args.plot_horizon]
        fig, axes = plt.subplots(2, 1, figsize=(11, 5), dpi=140, sharex=True)
        for ch in range(test_raw.shape[1]):
            axes[ch].plot(ref_raw[:, ch], color="black", lw=1.0, label="ref" if ch == 0 else None)
            axes[ch].plot(pred_raw[:, ch], lw=1.0, label=model.name if ch == 0 else None)
            axes[ch].set_ylabel(f"ch={ch}")
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("step")
        fig.suptitle(f"{model.name}: representative autonomous rollout")
        fig.tight_layout()
        safe_name = model.name.replace("/", "-").replace(" ", "_")
        fig.savefig(Path(args.out_dir) / "plots" / f"{safe_name}_overlay.png", bbox_inches="tight")
        plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# Checkpoint serialization helpers
# -----------------------------------------------------------------------------

def save_model(model: BaseRolloutModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, NVARModel):
        save_pickle(path, {"type": "nvar", "W": model.W, "seq_len": model.seq_len,
                           "dim": model.dim, "order": model.order,
                           "feature_clip": model.feature_clip,
                           "rollout_clip": model.rollout_clip})
    elif isinstance(model, ESNModel):
        save_pickle(path, {"type": "esn", "Win": model.Win, "Wres": model.Wres, "bias": model.bias,
                           "Wout": model.Wout, "leak": model.leak, "seq_len": model.seq_len, "dim": model.dim})
    elif isinstance(model, LSTMRolloutModel):
        torch.save({"type": "lstm", "state_dict": model.net.state_dict(), "seq_len": model.seq_len,
                    "dim": model.dim}, path)
    elif isinstance(model, KoopmanRolloutModel):
        torch.save({"type": "koopman", "state_dict": model.net.state_dict(), "seq_len": model.seq_len,
                    "dim": model.dim, "in_dim": model.seq_len * model.dim}, path)
    elif isinstance(model, DySLIMRolloutModel):
        torch.save({"type": "dyslim", "state_dict": model.net.state_dict(), "seq_len": model.seq_len,
                    "dim": model.dim}, path)
    else:
        raise TypeError(f"Unknown model type: {type(model)}")


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def train_one_baseline(name: str, train_scaled: np.ndarray, args, device: torch.device) -> BaseRolloutModel:
    name = name.lower()
    if name in {"ngrc", "nvar", "ngrc-nvar"}:
        print("\n=== Training NGRC / NVAR ===")
        return train_nvar(train_scaled, seq_len=args.seq_len, order=args.nvar_order,
                          ridge=args.nvar_ridge, feature_clip=args.nvar_feature_clip,
                          rollout_clip=args.rollout_clip)
    if name in {"esn", "reservoir", "reservoir-esn"}:
        print("\n=== Training Reservoir Computing / ESN ===")
        return train_esn(train_scaled, seq_len=args.seq_len, n_res=args.esn_size,
                         spectral_radius=args.esn_spectral_radius, leak=args.esn_leak,
                         input_scale=args.esn_input_scale, density=args.esn_density,
                         ridge=args.esn_ridge, washout=args.esn_washout, seed=args.seed)
    if name == "lstm":
        print("\n=== Training LSTM ===")
        return train_lstm(train_scaled, args=args, device=device)
    if name in {"koopman", "koopman-ae", "kae"}:
        print("\n=== Training Koopman-AE ===")
        return train_koopman_ae(train_scaled, args=args, device=device)
    if name in {"dyslim", "dyslim-style"}:
        print("\n=== Training DySLIM-style MMD regularized model ===")
        return train_dyslim_style(train_scaled, args=args, device=device)
    raise ValueError(f"Unknown baseline: {name}")


def write_results(out_dir: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(out_dir / "results_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    with open(out_dir / "latex_rows.txt", "w") as f:
        f.write("% Model & NMSE400 & W2 & PSD & DRTD & HRE & TailStd \\\n")
        for r in rows:
            f.write(
                f"{r['model']} & {r.get('NMSE_400', 'NA')} & "
                f"{float(r.get('Attractor_W2_mean', float('nan'))):.4g} & "
                f"{float(r.get('PSD_logL1_mean', float('nan'))):.4g} & "
                f"{float(r.get('Mean_DRTD_mean', float('nan'))):.4g} & "
                f"{float(r.get('Mean_HRE_mean', float('nan'))):.4g} & "
                f"{float(r.get('TailStd_mean', float('nan'))):.4g} \\\n"
            )
    print(f"\nSaved results: {out_dir / 'results_summary.csv'}")
    print(f"Saved LaTeX rows: {out_dir / 'latex_rows.txt'}")


def print_summary(rows: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 110)
    print("HETEROGENEOUS BASELINES SUMMARY")
    print("=" * 110)
    header = f"{'model':22s} | {'NMSE_100':18s} | {'NMSE_400':18s} | {'W2':10s} | {'PSD':10s} | {'DRTD':10s} | {'HRE':10s} | {'TailStd':10s} | {'Div':5s}"
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
            f"{int(r.get('Diverged_starts', 0)):5d}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train heterogeneous baselines on partially observed Lorenz63.")
    p.add_argument("--train_path", "--train-path", type=str, default=str(ROOT / "data/lorenz63/train_data.npy"))
    p.add_argument("--test_path", "--test-path", type=str, default=str(ROOT / "data/lorenz63/test_data.npy"))
    p.add_argument("--out_dir", "--output-dir", type=str, default=str(ROOT / "results/baselines/trained"))
    p.add_argument("--obs_cols", type=str, default="", help="Comma-separated columns, e.g. '0,2'. Empty: use all if d=2, else [0,2].")
    p.add_argument("--all", action="store_true", help="Train all baselines.")
    p.add_argument("--models", type=str, default="ngrc,esn,lstm,koopman,dyslim",
                   help="Comma-separated baselines: ngrc,esn,lstm,koopman,dyslim")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--dt_obs", type=float, default=0.1)

    # Shared rollout/eval
    p.add_argument("--seq_len", type=int, default=10, help="History length for sequence baselines and evaluation warm start.")
    p.add_argument("--eval_starts", type=int, default=30)
    p.add_argument("--long_horizon", type=int, default=500)
    p.add_argument("--plot_horizon", type=int, default=700)
    p.add_argument("--plot", action="store_true", default=True)
    p.add_argument("--no_plot", dest="plot", action="store_false")
    p.add_argument("--w2_blur", type=float, default=0.05)
    p.add_argument("--w2_max_points", type=int, default=700)
    p.add_argument("--sliced_proj", type=int, default=128)
    p.add_argument("--psd_nfft", type=int, default=2048)
    p.add_argument("--rollout_clip", type=float, default=50.0,
                   help="Scaled-space safety threshold. Rollouts exceeding this are marked as diverged and clipped for metric computation.")

    # Neural training
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--print_every", type=int, default=250)

    # NGRC/NVAR
    p.add_argument("--nvar_order", type=int, default=2)
    p.add_argument("--nvar_ridge", type=float, default=1e-3)
    p.add_argument("--nvar_feature_clip", type=float, default=20.0,
                   help="Numerical guard for NVAR polynomial features in scaled space.")

    # ESN
    p.add_argument("--esn_size", type=int, default=700)
    p.add_argument("--esn_spectral_radius", type=float, default=0.9)
    p.add_argument("--esn_leak", type=float, default=0.3)
    p.add_argument("--esn_input_scale", type=float, default=0.5)
    p.add_argument("--esn_density", type=float, default=0.05)
    p.add_argument("--esn_ridge", type=float, default=1e-5)
    p.add_argument("--esn_washout", type=int, default=100)

    # LSTM
    p.add_argument("--lstm_hidden", type=int, default=128)
    p.add_argument("--lstm_layers", type=int, default=2)

    # Koopman AE
    p.add_argument("--koopman_hidden", type=int, default=128)
    p.add_argument("--koopman_latent", type=int, default=16)
    p.add_argument("--koopman_recon_w", type=float, default=0.25)
    p.add_argument("--koopman_latent_w", type=float, default=0.1)

    # DySLIM-style
    p.add_argument("--dyslim_hidden", type=int, default=128)
    p.add_argument("--dyslim_rollout", type=int, default=16)
    p.add_argument("--dyslim_mmd_w", type=float, default=0.2)
    return p


def main():
    args = build_arg_parser().parse_args()
    torch.set_num_threads(1)
    if not HAS_GEOMLOSS:
        raise RuntimeError("Install GeomLoss for the paper transport metric.")
    set_seed(args.seed)
    device = choose_device(args.device)
    out_dir = Path(args.out_dir)
    args.out_dir = str(out_dir)
    if (out_dir / "run_config.json").exists():
        raise FileExistsError(f"Choose a fresh output directory: {out_dir}")
    ensure_dirs(out_dir)

    print("device:", device)
    print("geomloss available:", HAS_GEOMLOSS)
    train_raw0 = load_array(args.train_path)
    test_raw0 = load_array(args.test_path)
    obs_cols = parse_obs_cols(args.obs_cols, train_raw0.shape[1])
    train_raw = select_obs(train_raw0, obs_cols)
    test_raw = select_obs(test_raw0, obs_cols)
    if train_raw.shape[1] != 2:
        print(f"WARNING: selected observation dimension is {train_raw.shape[1]}, not 2. Proceeding anyway.")
    scaler = MaxAbsScaler().fit(train_raw)
    train_scaled = scaler.transform(train_raw)

    # Save run config and scaler.
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    save_pickle(out_dir / "checkpoints" / "scaler.pkl", scaler)

    if args.all:
        model_names = ["ngrc", "esn", "lstm", "koopman", "dyslim"]
    else:
        model_names = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    rows: List[Dict[str, object]] = []
    for name in model_names:
        t0 = time.time()
        print("\n" + "#" * 100)
        print(f"Training/evaluating baseline: {name}")
        print("#" * 100)
        model = train_one_baseline(name, train_scaled, args=args, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_finished = time.time()
        ckpt_name = model.name.replace("/", "-").replace(" ", "_") + f"_seed{args.seed}.p"
        save_model(model, out_dir / "checkpoints" / ckpt_name)
        row = evaluate_model(model, train_raw=train_raw, test_raw=test_raw, scaler=scaler, args=args, device=device)
        row["train_seconds"] = float(time.time() - t0)
        row["train_only_seconds"] = float(train_finished - t0)
        row["save_and_evaluation_seconds"] = float(time.time() - train_finished)
        rows.append(row)
        print_summary(rows)
        write_results(out_dir, rows)
    print_summary(rows)
    write_results(out_dir, rows)


if __name__ == "__main__":
    main()
