#!/usr/bin/env python3
"""Generate a NEW noisy Lorenz63 realization using the source Euler simulator.

The default full-state long-data split follows Lorenz/1-step MSE/Lorenz.ipynb
cell1: dt0.01,100000 Euler states, noise std1, stride10,5000 train and5000 test.
That cell does not establish the provenance of the bundled two-channel paper
split. This generator never claims bitwise regeneration of that archived data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def lorenz_system(state, sigma=10., rho=28., beta=8./3.):
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def euler_method(func, initial_state, dt, steps):
    initial_state = np.asarray(initial_state, dtype=float)
    states = np.zeros((steps, initial_state.size))
    states[0] = initial_state
    for i in range(1, steps):
        states[i] = states[i - 1] + func(states[i - 1]) * dt
    return states


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--noise-seed", type=int, default=0, help="New realization; historical noise RNG state unknown.")
    p.add_argument("--noise-std", type=float, default=1.)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--dt", type=float, default=.01)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--train-count", type=int, default=5000)
    p.add_argument("--test-count", type=int, default=5000)
    p.add_argument("--channels", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()
    if args.steps < 2 or args.dt <= 0 or args.stride < 1 or min(args.train_count, args.test_count) < 1 or args.noise_std < 0:
        p.error("steps>=2,dt>0,stride/counts>=1,noise-std>=0 required")
    if any(c not in [0, 1, 2] for c in args.channels):
        p.error("channels must be chosen from0,1,2")
    paths = [args.output_dir / "train_data.npy", args.output_dir / "test_data.npy"]
    if any(path.exists() for path in paths):
        raise FileExistsError("Choose a new --output-dir to preserve existing arrays")
    states = euler_method(lorenz_system, [.1, 0., 0.], args.dt, args.steps)
    if not np.isfinite(states).all():
        raise ValueError("Integration diverged; choose a smaller time step")
    noisy = states + np.random.RandomState(args.noise_seed).normal(0., args.noise_std, states.shape)
    observed = noisy[::args.stride][:, args.channels]
    if len(observed) < args.train_count + args.test_count:
        p.error("Insufficient downsampled observations for requested split")
    train = observed[:args.train_count]
    test = observed[args.train_count:args.train_count + args.test_count]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, array in zip(paths, [train, test]):
        np.save(path, array)
    manifest = dict(source="NIPS/Lorenz/1-step MSE/Lorenz.ipynb cell1 (one-based)",
                    initial_state=[.1, 0., 0.], sigma=10., rho=28., beta=8./3.,
                    dt=args.dt, steps=args.steps, stride=args.stride, dt_observed=args.dt * args.stride,
                    train_shape=list(train.shape), test_shape=list(test.shape), channels=args.channels,
                    noise_std=args.noise_std, noise_seed=args.noise_seed, historical_noise_seed="unknown",
                    paper_exact_realization=False, files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (args.output_dir / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
