#!/usr/bin/env python3
"""Evaluate the archived Lorenz63 models using the original table protocols."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import metrics as m

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["PRIME", "MSE", "DIM", "DIM+MSE"], help="Omit to evaluate all four.")
    p.add_argument("--seeds", nargs="+", type=int, default=[33, 123, 1000, 13532, 123456],
                   help="Historical folder labels; PRIME folder 13532 internally records seed 13531.")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/lorenz63")
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/lorenz63")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/lorenz63/evaluation")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--n-starts", type=int, default=32)
    p.add_argument("--pred-horizon", type=int, default=500)
    p.add_argument("--poincare-horizon", type=int, default=-1, help="-1 uses the full 999-step test trajectory.")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable. Explicit --device cpu supports a smoke test, not identical historical trajectories.")
    if not m.HAS_GEOMLOSS or not m.HAS_SCIPY_SIGNAL:
        p.error("Paper evaluation requires geomloss and scipy.signal; metric fallbacks are disabled.")
    if args.n_starts < 1 or args.pred_horizon < 1 or args.poincare_horizon < -1:
        p.error("Invalid rollout count or horizon")
    train = np.asarray(np.load(args.data_dir / "train_data.npy"), dtype=np.float64)
    test = np.asarray(np.load(args.data_dir / "test_data.npy"), dtype=np.float64)
    if train.shape != (500, 2) or test.shape != (1000, 2):
        p.error(f"Expected train (500,2), test (1000,2); got {train.shape}, {test.shape}")
    starts = m.choose_starts(len(test), args.pred_horizon, args.n_starts, mode="linspace", seed=42)
    horizon = len(test) - 1 if args.poincare_horizon == -1 else min(args.poincare_horizon, len(test) - 1)
    args.w2_blur, args.w2_max_points, args.psd_nfft = 0.05, 1000, 2048
    args.melr_eps, args.melr_drop_zero_mode = 1e-12, False
    args.tcm_max_lag, args.tcm_truncate_first_nonpositive = 200, False
    args.x_section_value, args.z_section_value, args.section_direction = 10.0, 40.0, 1
    args.unimodal_prominence, args.unimodal_distance, args.no_refine_peak_time = 1.0, 0, False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "evaluation_manifest.json").exists():
        p.error("Output directory already contains an evaluation; choose a different --output-dir")
    predictions, sections, errors, ckpts = [], [], [], []
    models = [args.model] if args.model else ["PRIME", "MSE", "DIM", "DIM+MSE"]
    for seed in args.seeds:
        for model in models:
            path = args.checkpoint_dir / f"seed_{seed}" / m.MODEL_FILES[model]
            print(f"Evaluating {model} folder seed={seed} on {args.device}", flush=True)
            try:
                net, scaler, cfg, dt_obs, dt_base, sub = m.load_net_scaler_cfg(path, train, args.device, default_dt_obs=0.1)
                actual_seed = cfg.get("seed")
                if actual_seed is not None and actual_seed != seed:
                    print(f"WARNING: folder seed={seed}, stored training seed={actual_seed}; retaining archival identity.", flush=True)
                ckpts.append(dict(model=model, folder_seed=seed, stored_seed=actual_seed,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(), config=cfg))
                identity = dict(seed_folder=f"1-step MSE {seed}", model=model,
                                checkpoint=f"seed_{seed}/{path.name}")
                for start in starts:
                    pred = m.rollout_from_x0(net, scaler, test[start], args.pred_horizon,
                                             dt_obs, args.device, dt_base, sub)
                    rec = m.evaluate_one_prediction_rollout(test[start:start + args.pred_horizon + 1], pred, 0.1, args)
                    rec.update(identity, start_idx=int(start), horizon=args.pred_horizon)
                    predictions.append(rec)
                pred_long = m.rollout_from_x0(net, scaler, test[0], horizon, dt_obs, args.device, dt_base, sub)
                rec = m.evaluate_full_poincare_rollout(test[:horizon + 1], pred_long, 0.1, args, z_idx=1)
                rec.update(identity, start_idx=0, poincare_horizon=horizon)
                sections.append(rec)
                del net
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:
                errors.append(dict(model=model, seed=seed, error=repr(exc)))
                print(f"ERROR: {exc}", file=sys.stderr)
    (args.output_dir / "eval_errors.json").write_text(json.dumps(errors, indent=2) + "\n")
    if not predictions or not sections:
        raise RuntimeError("No complete prediction / Poincare results; see eval_errors.json")
    pred_df, sec_df = pd.DataFrame(predictions), pd.DataFrame(sections)
    summary = m.merge_summaries(m.aggregate_prediction_results(pred_df), m.aggregate_poincare_results(sec_df))
    pred_df.to_csv(args.output_dir / "per_rollout_prediction_metrics_5seeds_32starts.csv", index=False)
    sec_df.to_csv(args.output_dir / "per_seed_full_horizon_poincare_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary_combined_prediction_and_full_poincare.csv", index=False)
    manifest = dict(device=args.device, torch_version=torch.__version__, numpy_version=np.__version__,
                    n_starts=len(starts), start_indices=starts.tolist(), pred_horizon=args.pred_horizon,
                    poincare_horizon=horizon, population_std=True, checkpoints=ckpts,
                    data_sha256={name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
                                 for name in ["train_data.npy", "test_data.npy"]},
                    completed_without_errors=not errors,
                    paper_protocol=(len(starts) == 32 and args.pred_horizon == 500 and horizon == 999),
                    warning="CPU and other numerical runtimes can differ on long chaotic trajectories.")
    (args.output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary[["model", "TCM_mean", "TCM_std", "tauW1_x10_mean", "tauW1_z40_mean", "tauW1_unimodal_mean"]].to_string(index=False))
    if errors:
        raise SystemExit("Evaluation incomplete; inspect eval_errors.json before using outputs.")

if __name__ == "__main__":
    main()
