#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-evaluate saved implicit-Poincare q_speed ablation checkpoints for ALL NMSE windows.

This script does NOT retrain. It recursively scans a finished ablation folder, loads each
seed-level BEST_TOTAL checkpoint, rolls out from fixed test starts, computes NMSE at all
requested windows, and aggregates:

  per-rollout: one row per (q_speed, seed, start)
  per-seed   : mean/std over rollout starts for each seed model
  per-q      : mean/std over seed-level means for each q_speed

Expected directory examples:
  implicit_poincare_multiseed_parallel_20260430_122402/
    q_0.10/ablation_.../qspeed_0p10/seed_123/Implicit_BEST_TOTAL_*.p
    q_0.10/ablation_.../qspeed_0p10/seed_456/Implicit_BEST_TOTAL_*.p
    ...

Run example:
  python reevaluate_implicit_ablation_all_nmse.py \
    --root ./implicit_poincare_multiseed_parallel_20260430_122402 \
    --test_path ../test_data.npy \
    --eval_horizon 500 \
    --n_rollouts 30 \
    --windows 10,20,50,100,200,300,400,500
"""

import argparse
import datetime as _dt
import glob
import io
import json
import os
import pickle
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MaxAbsScaler


# ============================================================
# Loading helpers
# ============================================================
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_p_cpu(path: str):
    with open(path, "rb") as f:
        return CPU_Unpickler(f).load()


def smart_load(path: str):
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


def strip_module_prefix(sd: Dict[str, torch.Tensor]):
    if isinstance(sd, dict) and len(sd) > 0 and all(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def infer_mlp_dims_from_state_dict(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    sd = strip_module_prefix(sd)
    wkeys = []
    for k in sd.keys():
        if k.endswith(".weight"):
            p = k.split(".")[0]
            if p.isdigit():
                wkeys.append((int(p), k))
    wkeys = sorted(wkeys, key=lambda x: x[0])
    if not wkeys:
        raise ValueError("Could not infer MLP dims from state_dict: no numeric '*.weight' keys.")
    W0 = sd[wkeys[0][1]]
    Wlast = sd[wkeys[-1][1]]
    return int(W0.shape[1]), int(W0.shape[0]), int(Wlast.shape[0])


def build_net(d: int, width: int = 100) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, d),
    )


def extract_state_dict_scaler_config(obj: Any, fallback_scaler_data: np.ndarray):
    sd = None
    scaler = None
    cfg = {}
    stats = {}

    if isinstance(obj, dict):
        if "config" in obj and isinstance(obj["config"], dict):
            cfg = dict(obj["config"])
        if "stats" in obj and isinstance(obj["stats"], dict):
            stats = obj["stats"]

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
        raise RuntimeError("Could not find usable state_dict in checkpoint.")
    sd = strip_module_prefix(sd)

    if scaler is None:
        print("[warn] checkpoint has no scaler; fitting MaxAbsScaler on provided test data as fallback.")
        scaler = MaxAbsScaler().fit(np.asarray(fallback_scaler_data, dtype=np.float64))

    return sd, scaler, cfg, stats


# ============================================================
# Evaluation helpers
# ============================================================
def ensure_2d_array(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return x


@torch.no_grad()
def rollout_raw_from_x0(
    net: nn.Module,
    scaler,
    x0_raw: np.ndarray,
    K_obs: int,
    dt_base: float,
    sub: int,
    device: str,
) -> np.ndarray:
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


def nmse_value_raw(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    L = min(len(gt), len(pred))
    if L <= 0:
        return np.nan
    gt = gt[:L]
    pred = pred[:L]
    return float(np.mean((gt - pred) ** 2) / (np.mean(gt ** 2) + eps))


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


def finite_mean_std(vals) -> Tuple[float, float, int]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, 0
    return float(np.mean(arr)), float(np.std(arr, ddof=0)), int(len(arr))


def fmt_mean_std(vals, nd: int = 4) -> str:
    mu, sd, n = finite_mean_std(vals)
    if not np.isfinite(mu):
        return "nan"
    return f"{mu:.{nd}g} ± {sd:.{nd}g}"


def parse_float_from_path(path: str, pattern: str) -> float:
    m = re.search(pattern, path)
    if not m:
        return np.nan
    s = m.group(1).replace("p", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_seed_from_path(path: str) -> int:
    m = re.search(r"seed[_-](\d+)", path)
    if m:
        return int(m.group(1))
    return -1


# ============================================================
# Checkpoint discovery
# ============================================================
def discover_checkpoints(root: str, checkpoint_kind: str = "best_total") -> List[str]:
    root = os.path.abspath(root)
    if checkpoint_kind == "best_total":
        patterns = [
            os.path.join(root, "**", "seed_*", "Implicit_BEST_TOTAL_*.p"),
            os.path.join(root, "**", "seed_*", "*_BEST_TOTAL_*.p"),
        ]
    elif checkpoint_kind == "best_l2":
        patterns = [
            os.path.join(root, "**", "seed_*", "Implicit_BEST_L2_*.p"),
            os.path.join(root, "**", "seed_*", "*_BEST_L2_*.p"),
        ]
    elif checkpoint_kind == "final":
        patterns = [
            os.path.join(root, "**", "seed_*", "Implicit_FINAL_*.p"),
            os.path.join(root, "**", "seed_*", "*_FINAL_*.p"),
        ]
    else:
        raise ValueError(f"Unknown checkpoint_kind={checkpoint_kind}")

    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))
    paths = sorted(set(paths))

    # One checkpoint per seed directory. If multiple, keep newest by modification time.
    by_dir: Dict[str, str] = {}
    for p in paths:
        d = os.path.dirname(p)
        if d not in by_dir or os.path.getmtime(p) > os.path.getmtime(by_dir[d]):
            by_dir[d] = p
    return sorted(by_dir.values())


# ============================================================
# Main evaluation
# ============================================================
def evaluate_checkpoint(path: str, test: np.ndarray, windows: List[int], args, device: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    obj = smart_load(path)
    sd, scaler, cfg, stats = extract_state_dict_scaler_config(obj, fallback_scaler_data=test)

    d_in, width, d_out = infer_mlp_dims_from_state_dict(sd)
    if d_in != d_out:
        raise ValueError(f"Model is not square: d_in={d_in}, d_out={d_out}")
    if d_in != test.shape[1]:
        raise ValueError(f"Model/data dimension mismatch: model d={d_in}, test d={test.shape[1]}")

    net = build_net(d_in, width=width).to(device)
    net.load_state_dict(sd)
    net.eval()

    dt_obs = float(cfg.get("dt_obs", args.dt_obs))
    dt_base = float(cfg.get("dt_base", args.dt_base if args.dt_base is not None else dt_obs))
    sub = int(cfg.get("sub", max(1, round(dt_obs / dt_base))))
    sub = max(1, sub)

    q_speed = cfg.get("q_speed", np.nan)
    try:
        q_speed = float(q_speed)
    except Exception:
        q_speed = parse_float_from_path(path, r"q[_-](\d+p\d+)")
    if not np.isfinite(q_speed):
        q_speed = parse_float_from_path(path, r"qspeed[_-](\d+p\d+)")

    seed = cfg.get("seed", parse_seed_from_path(path))
    try:
        seed = int(seed)
    except Exception:
        seed = parse_seed_from_path(path)

    horizon = min(int(args.eval_horizon), len(test) - 2)
    max_window = max(windows)
    if horizon < max_window:
        raise ValueError(f"eval_horizon={horizon} is smaller than requested max window={max_window}")

    starts = choose_eval_starts(
        T=len(test),
        horizon=horizon,
        n_rollouts=args.n_rollouts,
        mode=args.start_mode,
        seed=args.start_seed,
    )

    # Recurrence diagnostics from checkpoint payload.
    N_hits = cfg.get("N_hits", np.nan)
    try:
        N_hits = int(N_hits)
    except Exception:
        N_hits = np.nan
    taus = stats.get("taus_steps", None) if isinstance(stats, dict) else None
    if taus is not None:
        taus = np.asarray(taus, dtype=np.float64)
        tau_mean_steps = float(np.mean(taus)) if len(taus) else np.nan
        tau_std_steps = float(np.std(taus)) if len(taus) else np.nan
    else:
        tau_mean_steps = np.nan
        tau_std_steps = np.nan

    per_rollout_rows = []
    for s in starts:
        s = int(s)
        gt = test[s:s + horizon + 1]
        pred = rollout_raw_from_x0(
            net=net,
            scaler=scaler,
            x0_raw=test[s],
            K_obs=horizon,
            dt_base=dt_base,
            sub=sub,
            device=device,
        )
        row = {
            "q_speed": q_speed,
            "seed": seed,
            "start_idx": s,
            "checkpoint": path,
            "eval_horizon": horizon,
            "dt_obs": dt_obs,
            "dt_base": dt_base,
            "sub": sub,
            "N_hits": N_hits,
            "tau_mean_steps": tau_mean_steps,
            "tau_std_steps": tau_std_steps,
        }
        for w in windows:
            row[f"NMSE_{w}"] = nmse_value_raw(gt[:w + 1], pred[:w + 1])
        per_rollout_rows.append(row)

    seed_summary = {
        "q_speed": q_speed,
        "seed": seed,
        "checkpoint": path,
        "n_rollouts_actual": int(len(starts)),
        "eval_horizon": int(horizon),
        "eval_starts": json.dumps([int(x) for x in starts]),
        "dt_obs": dt_obs,
        "dt_base": dt_base,
        "sub": sub,
        "N_hits": N_hits,
        "tau_mean_steps": tau_mean_steps,
        "tau_std_steps": tau_std_steps,
    }
    for w in windows:
        vals = [r[f"NMSE_{w}"] for r in per_rollout_rows]
        mu, sd, n = finite_mean_std(vals)
        seed_summary[f"NMSE_{w}_mean_rollouts"] = mu
        seed_summary[f"NMSE_{w}_std_rollouts"] = sd
        seed_summary[f"NMSE_{w}_n_valid_rollouts"] = n
        seed_summary[f"NMSE_{w}"] = f"{mu:.6g} ± {sd:.6g}" if np.isfinite(mu) else "nan"

    return per_rollout_rows, seed_summary


def aggregate_over_q(seed_df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    rows = []
    for q, g in seed_df.groupby("q_speed", dropna=False):
        row = {
            "q_speed": float(q) if pd.notna(q) else np.nan,
            "n_seeds_ok": int(len(g)),
        }
        for diag in ["N_hits", "tau_mean_steps", "tau_std_steps"]:
            if diag in g.columns:
                row[diag] = fmt_mean_std(g[diag].to_numpy(), nd=4)
                mu, sd, n = finite_mean_std(g[diag].to_numpy())
                row[f"{diag}_mean"] = mu
                row[f"{diag}_std"] = sd
        for w in windows:
            col = f"NMSE_{w}_mean_rollouts"
            if col in g.columns:
                vals = g[col].to_numpy(dtype=np.float64)
                mu, sd, n = finite_mean_std(vals)
                row[f"NMSE_{w}"] = fmt_mean_std(vals, nd=4)
                row[f"NMSE_{w}_mean_over_seeds"] = mu
                row[f"NMSE_{w}_std_over_seeds"] = sd
                row[f"NMSE_{w}_n_valid_seeds"] = n
        rows.append(row)
    out = pd.DataFrame(rows)
    if "q_speed" in out.columns:
        out = out.sort_values("q_speed").reset_index(drop=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True, help="Finished implicit multiseed ablation root folder.")
    p.add_argument("--test_path", type=str, default="test_data.npy")
    p.add_argument("--checkpoint_kind", type=str, default="best_total", choices=["best_total", "best_l2", "final"])
    p.add_argument("--windows", type=str, default="10,20,50,100,200,300,400,500")
    p.add_argument("--eval_horizon", type=int, default=500)
    p.add_argument("--n_rollouts", type=int, default=30)
    p.add_argument("--start_mode", type=str, default="linspace", choices=["linspace", "random"])
    p.add_argument("--start_seed", type=int, default=42)
    p.add_argument("--dt_obs", type=float, default=0.1)
    p.add_argument("--dt_base", type=float, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)

    test = ensure_2d_array(np.load(args.test_path), "test_data")

    if args.out_dir is None:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join(args.root, f"reeval_all_nmse_{stamp}")
    os.makedirs(args.out_dir, exist_ok=True)

    ckpts = discover_checkpoints(args.root, checkpoint_kind=args.checkpoint_kind)
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoints found under {args.root} with checkpoint_kind={args.checkpoint_kind}")

    print("============================================================")
    print("Re-evaluate implicit ablation checkpoints for ALL NMSE windows")
    print("============================================================")
    print("device:", device)
    print("root:", os.path.abspath(args.root))
    print("test_path:", args.test_path, test.shape)
    print("checkpoint_kind:", args.checkpoint_kind)
    print("n_checkpoints:", len(ckpts))
    print("windows:", windows)
    print("eval_horizon:", args.eval_horizon)
    print("n_rollouts:", args.n_rollouts)
    print("out_dir:", args.out_dir)
    print("============================================================")

    all_rollout_rows = []
    all_seed_rows = []
    errors = []

    for i, ckpt in enumerate(ckpts, start=1):
        print(f"\n[{i}/{len(ckpts)}] {ckpt}")
        try:
            rollout_rows, seed_summary = evaluate_checkpoint(ckpt, test, windows, args, device)
            all_rollout_rows.extend(rollout_rows)
            all_seed_rows.append(seed_summary)
            print(
                f"  ok | q={seed_summary['q_speed']} | seed={seed_summary['seed']} | "
                f"n={seed_summary['n_rollouts_actual']} | "
                + " | ".join([f"NMSE_{w}={seed_summary[f'NMSE_{w}']}" for w in windows if f"NMSE_{w}" in seed_summary])
            )
        except Exception as e:
            print("  [ERROR]", repr(e))
            errors.append({"checkpoint": ckpt, "error": repr(e)})

    if not all_seed_rows:
        raise RuntimeError("No checkpoints were evaluated successfully.")

    rollout_df = pd.DataFrame(all_rollout_rows)
    seed_df = pd.DataFrame(all_seed_rows)
    q_df = aggregate_over_q(seed_df, windows)

    # Keep compact string columns first.
    compact_cols = ["q_speed", "n_seeds_ok", "N_hits", "tau_mean_steps"] + [f"NMSE_{w}" for w in windows]
    compact_cols = [c for c in compact_cols if c in q_df.columns]
    q_compact = q_df[compact_cols].copy()

    rollout_csv = os.path.join(args.out_dir, "per_rollout_all_nmse.csv")
    seed_csv = os.path.join(args.out_dir, "per_seed_all_nmse.csv")
    q_csv = os.path.join(args.out_dir, "aggregate_qspeed_all_nmse.csv")
    q_compact_csv = os.path.join(args.out_dir, "aggregate_qspeed_all_nmse_compact.csv")
    q_md = os.path.join(args.out_dir, "aggregate_qspeed_all_nmse_compact.md")
    err_json = os.path.join(args.out_dir, "reeval_errors.json")

    rollout_df.to_csv(rollout_csv, index=False)
    seed_df.to_csv(seed_csv, index=False)
    q_df.to_csv(q_csv, index=False)
    q_compact.to_csv(q_compact_csv, index=False)
    with open(q_md, "w") as f:
        f.write("# Implicit recurrence-hit ablation: all NMSE windows\n\n")
        f.write("Each q_speed is trained with multiple seeds. Each seed-level model is re-evaluated using fixed rollout starts. Values are mean ± std over seed-level rollout means.\n\n")
        f.write(q_compact.to_markdown(index=False))
    with open(err_json, "w") as f:
        json.dump(errors, f, indent=2)

    print("\n==================== Aggregate over q_speed: ALL NMSE ====================")
    print(q_compact.to_string(index=False))

    print("\nSaved:")
    print("  per-rollout:", rollout_csv)
    print("  per-seed   :", seed_csv)
    print("  per-q full :", q_csv)
    print("  per-q table:", q_compact_csv)
    print("  per-q MD   :", q_md)
    print("  errors     :", err_json)
    print("Done.")


if __name__ == "__main__":
    main()
