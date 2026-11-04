#!/usr/bin/env python3
"""Train the four KS methods using verified notebook losses and settings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from sklearn.preprocessing import MaxAbsScaler
from geomloss import SamplesLoss
from metrics import build_net, MODEL_ORDER, MODEL_FILE_CANDIDATES
from prime_training import train_prime

ROOT = Path(__file__).resolve().parents[2]


def train_baseline(train, model, seed, steps, device, output_dir):
    """Cells 13 (MSE), 15 (DIM), 17 (DIM+MSE); dt=1, tau_pair=2."""
    batch, tau_pair, tau_delay, dim_delay = 512, 2, 10, 3
    scaler = MaxAbsScaler().fit(train.astype(np.float32))
    y = scaler.transform(train.astype(np.float32))
    T, d = y.shape
    net = build_net(d, width=100).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    energy = SamplesLoss(loss="energy")
    delay_targets = np.concatenate([y[20 - lag:T - lag] for lag in (0, 10, 20)], axis=1)
    Tmax = T - (2 if model == "MSE" else 20)
    history = []
    best_loss, best_iteration = float("inf"), -1
    filename = MODEL_FILE_CANDIDATES[model][0]
    config = dict(seed=seed, steps=steps, batch=batch, lr=1e-3, width=100,
                  Kmax=500, tau_pair=2, micro_steps=2, dt_base=1, stride=1,
                  dt_obs=1, obs_dim=d)
    if model != "MSE":
        config.update(tau_delay=10, dim_delay=3, delay_mode="multichannel_concat",
                      delay_dim_total=15, w_push=1., w_dim=1.)
        # Cell 15 stores w_glue=0.1 but DOES NOT add it to its objective.
        config.update({"w_glue": 0.1} if model == "DIM" else {"w_mse": 1.})
    net.train()
    for it in range(steps):
        idx = np.random.randint(0, Tmax, size=batch)
        x = torch.tensor(y[idx], dtype=torch.float32, device=device)
        target = torch.tensor(y[idx + tau_pair], dtype=torch.float32, device=device)
        if model == "MSE":
            for _ in range(2):
                x = x + net(x)
            loss = ((x - target) ** 2).mean()
            components = {}
        else:
            target_delay = torch.tensor(delay_targets[idx], dtype=torch.float32, device=device)
            sols = []
            for j in range(30):
                sols.append(x)
                x = x + net(x)
                if j == 1:
                    x_pair = x
            sol = torch.stack(sols, dim=0)
            pred_delay = torch.flip(sol[::10], dims=[0]).permute(1, 0, 2).contiguous().view(batch, 15)
            loss_push = energy(x_pair, target)
            loss_dim = energy(pred_delay, target_delay)
            loss = loss_push + loss_dim
            components = {"push": float(loss_push.detach().cpu()), "dim": float(loss_dim.detach().cpu())}
            if model == "DIM+MSE":
                loss_mse = ((x_pair - target) ** 2).mean()
                loss = loss + loss_mse
                components["mse"] = float(loss_mse.detach().cpu())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        if not np.isfinite(value):
            raise RuntimeError(f"Nonfinite training loss at iteration {it}")
        history.append(dict(iteration=it, loss=value, **components))
        if value < best_loss:
            best_loss, best_iteration = value, it
            # Preserve historical ordering: select on pre-update loss; save post-update weights.
            payload = dict(iter=it, model_state_dict=net.state_dict(),
                           optimizer_state_dict=optimizer.state_dict(), scaler=scaler, config=config,
                           best_total_loss=best_loss)
            torch.save(payload, output_dir / filename)
        if it % 500 == 0 or it == steps - 1:
            print(f"{model} iteration={it} loss={value:.8g} best={best_loss:.8g}", flush=True)
    torch.save(dict(iter=steps - 1, model_state_dict=net.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(), scaler=scaler, config=config,
                    best_total_loss=best_loss), output_dir / filename.replace("BEST", "FINAL"))
    with (output_dir / f"{model.replace('+', '_')}_training_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return dict(best_loss=best_loss, best_iteration=best_iteration, steps=steps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=MODEL_ORDER)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--steps", type=int, help="Default 10000 for baselines; 12000 for PRIME.")
    p.add_argument("--warmup-steps", type=int, default=10000, help="PRIME DIM warmup; use 1 with --steps 2 for a two-stage smoke test.")
    p.add_argument("--device", default="cuda", help="cuda reproduces the archived backend; cpu is suitable for smoke tests.")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/ks")
    p.add_argument("--output-dir", type=Path, help="Seed checkpoint directory; default runs/ks/train/partial_seed_SEED.")
    args = p.parse_args()
    steps = args.steps if args.steps is not None else (12000 if args.model == "PRIME" else 10000)
    if steps < 1 or args.warmup_steps < 0:
        p.error("steps must be positive and warmup-steps nonnegative")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select --device cpu explicitly for a CPU run.")
    out = args.output_dir or ROOT / f"runs/ks/train/partial_seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / MODEL_FILE_CANDIDATES[args.model][0]).exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint in {out}; select a new --output-dir")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train_path = args.data_dir / "train.npy"
    train = np.load(train_path, allow_pickle=False)
    if train.shape != (500, 5) or not np.isfinite(train).all():
        raise ValueError(f"Expected finite (500,5) train data, got {train.shape}")
    started = time.perf_counter()
    if args.model == "PRIME":
        result = train_prime(train, args.seed, steps, args.warmup_steps, args.device, out)
    else:
        result = train_baseline(train, args.model, args.seed, steps, args.device, out)
    result.update(model=args.model, seed=args.seed, device=args.device,
                  elapsed_seconds=time.perf_counter() - started, numpy=np.__version__, torch=torch.__version__,
                  train_sha256=hashlib.sha256(train_path.read_bytes()).hexdigest(),
                  profile="KS_test.ipynb current training cells; see docs/ks.md")
    (out / f"{args.model.replace('+', '_')}_training_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
