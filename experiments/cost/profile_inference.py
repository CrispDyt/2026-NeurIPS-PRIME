#!/usr/bin/env python3
"""Table 20: time core single-state Lorenz63 Euler inference, excluding I/O."""
import argparse
import csv
import gc
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FILES = {"PRIME": "Implicit_BEST.p", "DIM": "DIM_BEST.pt", "DIM+MSE": "DIM_MSE_BEST.pt", "NODE": "NODE_BEST.p"}


@torch.inference_mode()
def rollout_final_state(network, initial_state, number_of_steps, dt_value):
    state = initial_state.clone()
    for _ in range(int(number_of_steps)):
        state = state + float(dt_value) * network(state)
    return state


def benchmark(network, initial_state, dt, steps, warmup, repeats, device):
    network.eval()
    for p in network.parameters():
        p.requires_grad_(False)
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    for _ in range(warmup):
        rollout_final_state(network, initial_state, steps, dt)
    synchronize()
    times = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        final = rollout_final_state(network, initial_state, steps, dt)
        synchronize()
        times.append((time.perf_counter() - started) * 1000.)
        _ = float(final[0, 0].detach().cpu())
    return dict(median_ms=float(np.median(times)), mean_ms=float(np.mean(times)), std_ms=float(np.std(times, ddof=0)),
                p10_ms=float(np.quantile(times, .1)), p90_ms=float(np.quantile(times, .9)))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/lorenz63")
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/lorenz63")
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs/cost/inference")
    p.add_argument("--seeds", nargs="+", type=int, default=[33, 123, 1000, 13532, 123456])
    p.add_argument("--methods", nargs="+", choices=list(FILES), default=list(FILES))
    p.add_argument("--device", default="cuda")
    p.add_argument("--rollout-steps", type=int, default=1000)
    p.add_argument("--warmup-runs", type=int, default=5)
    p.add_argument("--timed-repeats", type=int, default=20)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--quick", action="store_true", help="10-step rollout, one warmup, two repeats; not a paper timing.")
    args = p.parse_args()
    if args.quick:
        args.rollout_steps, args.warmup_runs, args.timed_repeats = 10, 1, 2
    if min(args.rollout_steps, args.timed_repeats, args.cpu_threads) < 1 or args.warmup_runs < 0:
        p.error("steps, repeats and threads must be positive; warmup nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu for CPU timing")
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    spec = importlib.util.spec_from_file_location("lorenz_metrics", ROOT / "experiments/lorenz63/metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    train = np.asarray(np.load(args.data_dir / "train_data.npy", allow_pickle=False), dtype=np.float64)
    test = np.asarray(np.load(args.data_dir / "test_data.npy", allow_pickle=False), dtype=np.float64)
    rows = []
    for seed in args.seeds:
        for method in args.methods:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            path = args.checkpoint_dir / f"seed_{seed}" / FILES[method]
            net, scaler, cfg, dt, _, _ = metrics.load_net_scaler_cfg(path, train, device, .1)
            initial = torch.as_tensor(scaler.transform(test[0].reshape(1, -1)).astype(np.float32), device=device)
            timing = benchmark(net, initial, dt, args.rollout_steps, args.warmup_runs, args.timed_repeats, device)
            median = timing["median_ms"]
            params = sum(p.numel() for p in net.parameters())
            rows.append(dict(seed=seed, method=method, parameter_count=params, rollout_steps=args.rollout_steps,
                             **timing, microseconds_per_step=median * 1000 / args.rollout_steps,
                             steps_per_second=args.rollout_steps * 1000 / median, checkpoint=f"seed_{seed}/{path.name}"))
            print(f"seed={seed} {method}: params={params}, {median * 1000 / args.rollout_steps:.4f} us/step", flush=True)
            del net, initial
    primes = {r['seed']: r['median_ms'] for r in rows if r['method'] == 'PRIME'}
    for row in rows:
        row['relative_to_PRIME'] = row['median_ms'] / primes[row['seed']] if row['seed'] in primes else np.nan
    rows.sort(key=lambda r: (r['seed'], r['method']))
    summaries = []
    for method in sorted(args.methods):
        group = [r for r in rows if r['method'] == method]
        values = [r['median_ms'] for r in group]
        ratios = [r['relative_to_PRIME'] for r in group if np.isfinite(r['relative_to_PRIME'])]
        summaries.append(dict(method=method, n_seeds=len(group), parameter_count_min=min(r['parameter_count'] for r in group),
                              parameter_count_max=max(r['parameter_count'] for r in group),
                              rollout_steps=args.rollout_steps, time_rollout_ms_mean=float(np.mean(values)),
                              time_rollout_ms_std=float(np.std(values, ddof=0)),
                              microseconds_per_step_mean=float(np.mean([r['microseconds_per_step'] for r in group])),
                              microseconds_per_step_std=float(np.std([r['microseconds_per_step'] for r in group], ddof=0)),
                              relative_to_PRIME_mean=float(np.mean(ratios)) if ratios else np.nan))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "inference_time_per_checkpoint.csv", rows)
    write_csv(args.output_dir / "inference_time_summary_by_method.csv", summaries)
    metadata = dict(device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                    torch_version=torch.__version__, python_version=sys.version, platform=platform.platform(),
                    cpu_threads=args.cpu_threads if device.type == "cpu" else None, batch_size=1,
                    rollout_steps=args.rollout_steps, warmup_runs=args.warmup_runs, timed_repeats=args.timed_repeats,
                    timing_scope="Core Euler rollout only; checkpoint/data loading, scaling and metrics excluded.",
                    source="pP3L_Rebuttal.ipynb cell 5; median over timed repetitions, mean/population std across seeds")
    (args.output_dir / "inference_time_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
