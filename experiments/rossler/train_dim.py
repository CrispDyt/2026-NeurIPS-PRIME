#!/usr/bin/env python3
"""Portable extraction of the original notebook plotting code; see docs/figures.md."""
import argparse
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def main():
    import os
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/figures/train_dim')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--channel', type=int, choices=[0,2], default=0)
    p.add_argument('--steps', type=int, default=10000)
    args = p.parse_args()
    if args.steps < 1:
        p.error('--steps must be positive')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import pandas  # initialize before legacy NumPy pickle aliases
    import numpy as np
    import sys
    import torch
    torch.set_num_threads(1)
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; --device cpu is a numerical smoke test.')
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    os.chdir(args.output_dir)
    train_data = np.load(ROOT/'data/rossler/train_data.npy')
    test_data = np.load(ROOT/'data/rossler/test_data.npy')
    if (args.output_dir/f'DIM_channel{args.channel}_seed{args.seed}.pt').exists():
        raise FileExistsError('Checkpoint exists; choose a new output directory.')
    import os
    import datetime
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    from torch import optim
    from sklearn.preprocessing import MaxAbsScaler
    import matplotlib.pyplot as plt
    from geomloss import SamplesLoss

    # =========================================================
    # random seed
    # =========================================================
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # =========================================================
    # basic config
    # =========================================================
    device  = args.device
    dt_base = 1
    stride  = 1
    dt_obs  = dt_base * stride

    tau_pair = 2
    micro_steps = tau_pair * stride

    tau_delay = 1
    dim_delay = 5
    DELAY_CH = args.channel

    # ---------------------------------------------------------
    # three-term weights
    # 1) pushforward measure loss
    # 2) DIM loss
    # 3) glue loss
    # ---------------------------------------------------------
    w_push = 1.0
    w_dim  = 1.0
    w_glue = 1.0

    steps = args.steps
    batch = 512
    lr    = 1e-3
    width = 100
    Kmax  = 500

    # =========================================================
    # save-best config
    # =========================================================
    save_dir = str(args.output_dir)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_ckpt_path = os.path.join(save_dir, f"DIM_channel{DELAY_CH}_seed{seed}.pt")

    print(
        "seed", seed,
        "train", train_data.shape,
        "test", test_data.shape,
        "dt_obs", dt_obs,
        "tau_pair", tau_pair
    )

    # =========================================================
    # data scaling
    # =========================================================
    scaler = MaxAbsScaler().fit(train_data.astype(np.float32))
    train_s = scaler.transform(train_data.astype(np.float32))
    test_s  = scaler.transform(test_data.astype(np.float32))

    T, d = train_s.shape

    # =========================================================
    # delay target constructor (single channel)
    # =========================================================
    def time_delay(Y, tau_delay=1, dim_delay=5, ch=0):
        Y = np.asarray(Y, dtype=np.float32)
        s = Y[:, ch]
        H = (dim_delay - 1) * tau_delay
        D = np.zeros((len(Y) - H, dim_delay), dtype=np.float32)
        for t in range(len(Y) - H):
            D[t] = np.array([s[t + H - i * tau_delay] for i in range(dim_delay)], dtype=np.float32)
        return D

    delay_targets = time_delay(
        train_s, tau_delay=tau_delay, dim_delay=dim_delay, ch=DELAY_CH
    )

    max_horizon = max(tau_pair, (dim_delay - 1) * tau_delay)
    Tmax = T - max_horizon

    # =========================================================
    # model
    # =========================================================
    net = nn.Sequential(
        nn.Linear(d, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, d),
    ).to(device)

    opt = optim.Adam(net.parameters(), lr=lr)

    # energy distance
    measure_loss = SamplesLoss(loss="energy")
    net.train()

    # =========================================================
    # history
    # =========================================================
    loss_hist = []
    loss_push_hist = []
    loss_dim_hist = []
    loss_glue_hist = []

    best_loss = float("inf")
    best_iter = -1

    # number of steps needed to build delay embedding
    n_dim_steps = tau_delay * dim_delay * stride
    # total rollout steps needed to support both tau_pair and delay construction
    n_total_steps = max(micro_steps, n_dim_steps)
    step_fine = tau_delay * stride

    # =========================================================
    # training
    # =========================================================
    for it in range(steps):
        idx = np.random.randint(0, Tmax, size=batch)

        x0 = torch.tensor(train_s[idx], dtype=torch.float32, device=device)
        yT = torch.tensor(train_s[idx + tau_pair], dtype=torch.float32, device=device)
        y_delay = torch.tensor(delay_targets[idx], dtype=torch.float32, device=device)

        x = x0.clone()
        sols = []
        xT_m = None

        for j in range(n_total_steps):
            if j < n_dim_steps:
                sols.append(x)
            x = x + dt_base * net(x)
            if j == micro_steps - 1:
                xT_m = x

        # sol: [n_dim_steps, batch, d]
        sol = torch.stack(sols, dim=0)

        # single-channel delay embedding, shape [batch, dim_delay]
        DIM_model = torch.flip(sol[::step_fine, :, DELAY_CH].T, dims=[1])

        # -----------------------------------------------------
        # three losses
        # -----------------------------------------------------
        # 1) pushforward measure loss
        L_push = measure_loss(xT_m, yT)

        # 2) DIM loss
        L_dim = measure_loss(DIM_model, y_delay)

        # 3) glue loss
        # keep your current "local/single-channel" style:
        # use pointwise MSE on the delay window except the last coordinate
        L_glue = ((DIM_model[:, :-1] - y_delay[:, :-1]) ** 2).mean()

        # total loss
        loss = w_push * L_push + w_dim * L_dim + w_glue * L_glue

        opt.zero_grad()
        loss.backward()
        opt.step()

        cur_loss = float(loss.detach().cpu())
        cur_push = float(L_push.detach().cpu())
        cur_dim  = float(L_dim.detach().cpu())
        cur_glue = float(L_glue.detach().cpu())

        loss_hist.append(cur_loss)
        loss_push_hist.append(cur_push)
        loss_dim_hist.append(cur_dim)
        loss_glue_hist.append(cur_glue)

        # =====================================================
        # save BEST model
        # =====================================================
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_iter = it

            payload = {
                "iter": it,
                "best_total_loss": best_loss,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler": scaler,
                "config": {
                    "seed": seed,
                    "dt_base": dt_base,
                    "stride": stride,
                    "dt_obs": dt_obs,
                    "tau_pair": tau_pair,
                    "micro_steps": micro_steps,
                    "tau_delay": tau_delay,
                    "dim_delay": dim_delay,
                    "DELAY_CH": DELAY_CH,
                    "w_push": w_push,
                    "w_dim": w_dim,
                    "w_glue": w_glue,
                    "steps": steps,
                    "batch": batch,
                    "lr": lr,
                    "width": width,
                    "Kmax": Kmax,
                    "obs_dim": d,
                },
            }
            torch.save(payload, best_ckpt_path)

            print(
                f"[BEST] iter {it:5d} | "
                f"L_total = {best_loss:.6f} | "
                f"L_push = {cur_push:.6f} | "
                f"L_dim = {cur_dim:.6f} | "
                f"L_glue = {cur_glue:.6f} | "
                f"saved to {best_ckpt_path}"
            )

        if it % 500 == 0:
            print(
                f"iter {it:5d} | "
                f"L_total = {cur_loss:.6f} | "
                f"L_push = {cur_push:.6f} | "
                f"L_dim = {cur_dim:.6f} | "
                f"L_glue = {cur_glue:.6f}"
            )

    print("\nTraining finished.")
    print(f"Best total loss = {best_loss:.6f} at iter = {best_iter}")
    print(f"Best checkpoint saved at:\n{best_ckpt_path}")

    # =========================================================
    # reload BEST model
    # =========================================================
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    print(
        f"\nReloaded BEST model from iter {ckpt['iter']} "
        f"with best_total_loss = {ckpt['best_total_loss']:.6f}"
    )

    # =========================================================
    # rollout
    # =========================================================
    @torch.no_grad()
    def rollout_from_x0(net, x0_s, Kmax):
        x = torch.tensor(x0_s.reshape(1, -1), dtype=torch.float32, device=device)
        out = np.zeros((Kmax + 1, d), dtype=np.float32)
        out[0] = x.detach().cpu().numpy()[0]

        for k in range(1, Kmax + 1):
            for _ in range(stride):
                x = x + dt_base * net(x)
            out[k] = x.detach().cpu().numpy()[0]

        return out

    # keep your current rollout start
    Kmax_eff = min(Kmax, len(test_s) - 1)
    pr_s = rollout_from_x0(net, test_s[0], Kmax_eff)
    pr   = scaler.inverse_transform(pr_s)
    gt   = test_data[:Kmax_eff + 1]

    def mse_k(k):
        return float(np.mean((gt[:k+1] - pr[:k+1]) ** 2))

    print("\nMSE windows (rollout from test[0])")
    for k in [1, 2, 3, 10, 20, 30, 50, 100, 200, 300, 500]:
        if k <= Kmax_eff:
            print(f"MSE_{k:4d}  (T={k*dt_obs:7.2f}s) = {mse_k(k):.6f}")

    t = np.arange(Kmax_eff + 1) * dt_obs

    plt.figure(figsize=(10, 3))
    for j in range(d):
        plt.plot(t, gt[:, j], label=f"GT[{j}]")
        plt.plot(t, pr[:, j], "--", label=f"PRED[{j}]")
    plt.legend(ncol=2, fontsize=8)
    plt.title(f"DIM rollout from test[0] | dt_obs={dt_obs}")
    plt.tight_layout()
    plt.savefig(f'figure_{len(list(args.output_dir.glob("figure_*.png"))):02d}.png', dpi=180, bbox_inches='tight'); plt.close()

    plt.figure(figsize=(10, 3))
    plt.plot(loss_hist, label="L_total")
    plt.plot(loss_push_hist, label="L_push")
    plt.plot(loss_dim_hist, label="L_dim")
    plt.plot(loss_glue_hist, label="L_glue")
    plt.legend()
    plt.title("Training losses")
    plt.tight_layout()
    plt.savefig(f'figure_{len(list(args.output_dir.glob("figure_*.png"))):02d}.png', dpi=180, bbox_inches='tight'); plt.close()

if __name__ == '__main__':
    main()
