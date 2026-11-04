#!/usr/bin/env python3
"""Evaluate KS checkpoints with the published Table 2/3 rollout protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings
import numpy as np
import torch
import metrics as m

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [4242, 10000, 13531, 123123, 123123123]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=m.MODEL_ORDER, help="Omit to evaluate all four models.")
    p.add_argument("--seeds", nargs="+", default=[str(x) for x in SEEDS], help="Space or comma separated seeds.")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/ks")
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/ks")
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs/ks/evaluation")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-starts", type=int, default=32)
    p.add_argument("--pred-horizon", type=int, default=1000)
    p.add_argument("--poincare-rollout-len", type=int, default=1000)
    args = p.parse_args()
    if any((args.output_dir / name).exists() for name in (
        "evaluation_manifest.json", "per_rollout_prediction_metrics_KS.csv", "per_run_tauW1_original_logic.csv"
    )):
        p.error("Output directory already contains evaluation results; choose a new --output-dir")
    if args.n_starts < 1 or args.pred_horizon < 8 or args.poincare_rollout_len < 1:
        p.error("n-starts must be positive, pred-horizon >=8, and poincare-rollout-len positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. CPU can be explicitly selected, but chaotic long rollouts differ from the CUDA references.")
    seeds = [int(s) for token in args.seeds for s in token.split(",") if s]
    models = [args.model] if args.model else m.MODEL_ORDER
    args.nmse_windows = [20, 100, 400]
    args.w2_blur, args.w2_max_points, args.psd_nfft = .05, 1500, 2048
    args.tcm_max_lag, args.tcm_truncate_first_nonpositive = 200, False
    args.poincare_num_starts, args.section_direction = 1, 0
    train_path, test_path = args.data_dir / "train.npy", args.data_dir / "test_full.npy"
    train = np.load(train_path, allow_pickle=False)
    test = np.asarray(np.load(test_path, allow_pickle=False), dtype=np.float64)
    if train.shape != (500, 5) or test.shape != (2500, 5):
        raise ValueError(f"Expected train (500,5), test (2500,5), got {train.shape}, {test.shape}")
    if not np.isfinite(test).all():
        raise ValueError("Nonfinite test data")
    loaded, provenance = [], []
    for seed in seeds:
        for model in models:
            sf = f"partial_seed_{seed}"
            path = args.checkpoint_dir / sf / m.MODEL_FILE_CANDIDATES[model][0]
            obj, state, scaler = m.load_model_ckpt(path, train)
            actual_seed = obj.get("config", {}).get("seed", obj.get("seed"))
            if actual_seed != seed:
                warnings.warn(f"Archived checkpoint identity mismatch: folder {seed}, metadata seed {actual_seed}, model {model}. See docs/ks.md.")
            di, width, do = m.infer_mlp_dims_from_state_dict(state)
            if di != 5 or do != 5:
                raise ValueError(f"Invalid checkpoint dimensions: {path}")
            net = m.build_net(di, width).to(args.device)
            net.load_state_dict(state)
            net.eval()
            loaded.append(dict(model=model, seed_folder=sf, checkpoint=f"{sf}/{path.name}", net=net, scaler=scaler))
            provenance.append(dict(model=model, folder_seed=seed, metadata_seed=actual_seed,
                                   file=f"{sf}/{path.name}", sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pred, pred_errors = m.evaluate_prediction_metrics(loaded, test, 1., args, args.device)
    tau, tau_errors = m.evaluate_tauW1_original_logic(loaded, test, 1., args, args.device)
    pred_summary = m.summarize_prediction(pred, args.nmse_windows)
    tau_summary = m.summarize_tauW1(tau)
    for name, rows in [("per_rollout_prediction_metrics_KS.csv", pred),
                       ("per_run_tauW1_original_logic.csv", tau),
                       ("summary_prediction_metrics_KS.csv", pred_summary),
                       ("summary_tauW1_original_logic.csv", tau_summary)]:
        m.write_csv(args.output_dir / name, rows)
    m.write_markdown_summary(args.output_dir / "summary_KS_full_metrics_original_tauW1.md", pred_summary, tau_summary, args.nmse_windows)
    errors = pred_errors + tau_errors
    (args.output_dir / "eval_errors.json").write_text(json.dumps(errors, indent=2) + "\n")
    manifest = dict(device=args.device, torch=torch.__version__, numpy=np.__version__,
                    elapsed_seconds=time.perf_counter() - started, dt_obs=1., n_starts=args.n_starts,
                    pred_horizon=args.pred_horizon, poincare_rollout_len=args.poincare_rollout_len,
                    checkpoints=provenance, test_sha256=hashlib.sha256(test_path.read_bytes()).hexdigest(),
                    prediction_rows=len(pred), drtd_rows=len(tau), errors=len(errors),
                    paper_section_source_ids=["Sigma1_u0_0", "Sigma3_u5_0", "Sigma2_u3_0"])
    (args.output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if errors:
        raise RuntimeError(f"Evaluation has {len(errors)} error(s); see eval_errors.json")


if __name__ == "__main__":
    main()
