#!/usr/bin/env python3
"""Train the four Lorenz63 methods from the original main-table experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MaxAbsScaler

ROOT = Path(__file__).resolve().parents[2]
FILES = {"PRIME": "Implicit_BEST.p", "MSE": "NODE_BEST.p",
         "DIM": "DIM_BEST.pt", "DIM+MSE": "DIM_MSE_BEST.pt"}


def build_net(d):
    return nn.Sequential(nn.Linear(d, 100), nn.Tanh(), nn.Linear(100, 100),
                         nn.Tanh(), nn.Linear(100, 100), nn.Tanh(), nn.Linear(100, d))


def train_baseline(args, data, run_dir):
    """Notebook cells 4/6/8; retain the distinct DIM anchor and DIM+MSE pair losses."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    scaler = MaxAbsScaler().fit(data.astype(np.float32))
    train_s = scaler.transform(data.astype(np.float32))
    T, d = train_s.shape
    steps = 10000 if args.steps is None else args.steps
    config = dict(seed=args.seed, dt_base=0.1, stride=1, dt_obs=0.1, tau_pair=2,
                  micro_steps=2, steps=steps, batch=512, lr=1e-3, width=100,
                  Kmax=500, obs_dim=d)
    net = build_net(d).to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    if args.model != "MSE":
        from geomloss import SamplesLoss
        measure_loss = SamplesLoss(loss="energy")
        config.update(tau_delay=1, dim_delay=5, DELAY_CH=0, w_im=1.0, w_dim=1.0,
                      w_mse=1.0 if args.model == "DIM" else 0.1)
        delay_targets = np.asarray([train_s[t:t + 5, 0][::-1]
                                    for t in range(T - 4)], dtype=np.float32)
        y_true_anchor = torch.tensor(train_s[:5], dtype=torch.float32, device=args.device)
        x_anchor = torch.tensor(train_s[0], dtype=torch.float32, device=args.device)
    best_loss, best_iter = float("inf"), -1
    history = []
    best_path = run_dir / FILES[args.model]
    net.train()
    for it in range(steps):
        idx = np.random.randint(0, T - (2 if args.model == "MSE" else 4), size=512)
        x0 = torch.tensor(train_s[idx], dtype=torch.float32, device=args.device)
        yT = torch.tensor(train_s[idx + 2], dtype=torch.float32, device=args.device)
        if args.model == "MSE":
            x = x0
            for _ in range(2):
                x = x + 0.1 * net(x)
            loss = ((x - yT) ** 2).mean()
            parts = {}
        else:
            y_delay = torch.tensor(delay_targets[idx], dtype=torch.float32, device=args.device)
            x, sols, xT_m = x0.clone(), [], None
            for j in range(5):
                sols.append(x)
                x = x + 0.1 * net(x)
                if j == 1:
                    xT_m = x
            sol = torch.stack(sols, dim=0)
            dim_model = torch.flip(sol[:, :, 0].T, dims=[1])
            l_im = measure_loss(xT_m, yT)
            l_dim = measure_loss(dim_model, y_delay)
            if args.model == "DIM":
                anchors, curr_x = [x_anchor], x_anchor.clone()
                for _ in range(4):
                    curr_x = curr_x + 0.1 * net(curr_x)
                    anchors.append(curr_x)
                l_mse = torch.mean((torch.stack(anchors, dim=0) - y_true_anchor) ** 2)
            else:
                # Cell 8 computes an unused anchor rollout; omitting it changes no RNG or gradient.
                l_mse = torch.mean((xT_m - yT) ** 2)
            loss = l_dim + l_im + config["w_mse"] * l_mse
            parts = {"mse": float(l_mse.detach().cpu()), "im": float(l_im.detach().cpu()),
                     "dim": float(l_dim.detach().cpu())}
        opt.zero_grad()
        loss.backward()
        opt.step()
        cur_loss = float(loss.detach().cpu())
        history.append(dict(iteration=it, loss=cur_loss, **parts))
        if not np.isfinite(cur_loss):
            raise RuntimeError(f"Non-finite training loss at iteration {it}")
        # The source compares the pre-update batch loss and saves the post-update weights.
        if cur_loss < best_loss:
            best_loss, best_iter = cur_loss, it
            payload = dict(iter=it, model_state_dict=net.state_dict(),
                           optimizer_state_dict=opt.state_dict(), scaler=scaler, config=config)
            payload["best_train_loss" if args.model == "MSE" else "best_total_loss"] = best_loss
            torch.save(payload, best_path)
        if it % 500 == 0 or it == steps - 1:
            print(f"{args.model} seed={args.seed} iteration={it} loss={cur_loss:.9g}", flush=True)
    final_path = run_dir / FILES[args.model].replace("BEST", "FINAL")
    torch.save(dict(iter=steps - 1, model_state_dict=net.state_dict(),
                    optimizer_state_dict=opt.state_dict(), scaler=scaler, config=config), final_path)
    with (run_dir / f"{args.model.replace('+', '_')}_training_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    return dict(effective_seed=args.seed, best_loss=best_loss, best_iteration=best_iter,
                config=config, best_checkpoint=str(best_path), final_checkpoint=str(final_path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(FILES), required=True)
    p.add_argument("--seed", type=int, required=True, help="Actual random seed; never remapped.")
    p.add_argument("--steps", type=int, help="Override 10000 baseline / 4000 PRIME iterations.")
    p.add_argument("--warm-steps", type=int, help="PRIME only: override historical 3000 warm-up steps.")
    p.add_argument("--profile", choices=["standard", "legacy13532"], default="standard",
                   help="legacy13532: PRIME cfg .2/128/.3; use --seed 13531 for the stored historical run.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/lorenz63")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/lorenz63/trained")
    args = p.parse_args()
    if args.steps is not None and args.steps < 1:
        p.error("--steps must be positive")
    if args.warm_steps is not None and args.warm_steps < 0:
        p.error("--warm-steps must be nonnegative")
    if args.model != "PRIME" and (args.profile != "standard" or args.warm_steps is not None):
        p.error("--profile legacy13532 and --warm-steps are PRIME-only options")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable; --device cpu is supported for smoke tests but changes numerical trajectories.")
    data_path = args.data_dir / "train_data.npy"
    data = np.load(data_path)
    if data.shape != (500, 2) or not np.isfinite(data).all():
        p.error(f"Expected finite Lorenz training array (500, 2), received {data.shape}")
    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if any((run_dir / FILES[args.model].replace("BEST", suffix)).exists() for suffix in ["BEST", "FINAL"]):
        p.error(f"Run outputs already exist in {run_dir}; choose a different --output-dir")
    started = time.monotonic()
    if args.model == "PRIME":
        from prime_training import train_prime
        result = train_prime(args, data, run_dir)
    else:
        result = train_baseline(args, data, run_dir)
    result.update(model=args.model, requested_seed=args.seed, device=args.device,
                  train_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                  elapsed_seconds=time.monotonic() - started, torch_version=torch.__version__,
                  numpy_version=np.__version__, profile=args.profile)
    manifest = run_dir / f"{args.model.replace('+', '_')}_training_manifest.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved run manifest: {manifest}")

if __name__ == "__main__":
    main()
