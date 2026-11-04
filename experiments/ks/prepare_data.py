#!/usr/bin/env python3
"""Generate a new KS trajectory following KS_test.ipynb cells 1,4.

The bundled arrays are the exact archived inputs. Regenerated chaotic data
can differ across JAX versions/devices; this command writes to a new directory.
Simulator adapted from Felix Koehler's MIT-licensed machine-learning-and-simulation
implementation; see experiments/ks/THIRD_PARTY.md.
"""
import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--noise-std", type=float, default=0.25)
    p.add_argument("--noise-seed", type=int, default=0, help="Seed for newly generated training noise; historical noise RNG state was not saved.")
    args = p.parse_args()
    if any((args.output_dir / name).exists() for name in ["train.npy", "test_full.npy"]):
        raise FileExistsError("Select a fresh output directory to preserve existing arrays.")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_platform_name", args.device)
    N, L, dt = 200, 100., .1
    mesh = jnp.linspace(0., L, N, endpoint=False)
    k = jnp.fft.rfftfreq(N, d=L / (N * 2 * jnp.pi))
    derivative = 1j * k
    linear = -derivative**2 - derivative**4
    exp = jnp.exp(dt * linear)
    coef = jnp.where(linear == 0., dt, (exp - 1.) / linear)
    mask = k < 2 / 3 * jnp.max(k)

    @jax.jit
    def step(u):
        nonlinear = jnp.fft.rfft(-.5 * u**2) * mask
        return jnp.fft.irfft(exp * jnp.fft.rfft(u) + coef * derivative * nonlinear, n=N)

    u = jnp.sin(2 * jnp.pi * mesh / L) + 1e-2 * jax.random.normal(jax.random.PRNGKey(0), shape=(N,))
    observations = []
    # Only the first 40,000 samples enter the notebook's published-data slice.
    for i in range(40000):
        if i >= 10000 and i % 10 == 0:
            observations.append(np.asarray(u)[[0, 50, 80, 100, 150]])
        u = step(u)
    observations = np.asarray(observations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    noise = np.random.RandomState(args.noise_seed).normal(0., args.noise_std, size=observations[:500].shape)
    np.save(args.output_dir / "train.npy", (observations[:500] + noise).astype(np.float32))
    np.save(args.output_dir / "test_full.npy", observations[500:])
    print(f"Saved {observations[:500].shape} train and {observations[500:].shape} test; dt_obs=1.0")


if __name__ == "__main__":
    main()
