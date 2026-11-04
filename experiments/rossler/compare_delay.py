#!/usr/bin/env python3
"""Compare fresh Rössler DIM models. This does not recover the missing original Figure 4 pair."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-y0', type=Path, required=True)
    p.add_argument('--checkpoint-y2', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--n-starts', type=int, default=30)
    p.add_argument('--horizon', type=int, default=500)
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/rossler/comparison')
    args = p.parse_args()
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    torch.set_num_threads(1)
    test = np.load(ROOT/'data/rossler/test_data.npy')
    if not 1 <= args.horizon < len(test) or args.n_starts < 2:
        p.error('horizon must fit the test data and n-starts must be at least two')
    starts = np.unique(np.linspace(0, len(test)-args.horizon-1, args.n_starts).astype(int))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, predictions = [], {}
    fig, ax = plt.subplots(figsize=(6, 4))
    for channel, path in [(0, args.checkpoint_y0), (2, args.checkpoint_y2)]:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        cfg, scaler = ckpt['config'], ckpt['scaler']
        if cfg['DELAY_CH'] != channel:
            raise ValueError(f'{path}: expected delay channel {channel}, got {cfg["DELAY_CH"]}')
        width = cfg['width']
        net = torch.nn.Sequential(torch.nn.Linear(3, width), torch.nn.Tanh(),
                                  torch.nn.Linear(width, width), torch.nn.Tanh(),
                                  torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, 3))
        net.load_state_dict(ckpt['model_state_dict'])
        net.to(args.device).eval()
        curves = []
        for start in starts:
            state = torch.tensor(scaler.transform(test[start:start+1]).astype(np.float32), device=args.device)
            trajectory = []
            with torch.no_grad():
                for _ in range(args.horizon):
                    state = state + cfg['dt_base'] * net(state)
                    trajectory.append(state.cpu().numpy()[0])
            pred = scaler.inverse_transform(np.asarray(trajectory))
            mse = np.mean((pred-test[start+1:start+args.horizon+1])**2, axis=1)
            curves.append(np.cumsum(mse)/np.arange(1, args.horizon+1))
        curves = np.asarray(curves)
        predictions[f'channel{channel}_cumulative_mse'] = curves
        mean = curves.mean(axis=0)
        ci = 1.96*curves.std(axis=0, ddof=1)/np.sqrt(len(curves))
        time = np.arange(1, args.horizon+1)*cfg['dt_obs']
        ax.plot(time, mean, label=f'Delay channel {channel}')
        ax.fill_between(time, mean-ci, mean+ci, alpha=.2)
        for t, m, c in zip(time, mean, ci):
            rows.append(dict(delay_channel=channel, time=t, mean_cumulative_mse=m, ci95_half_width=c))
    ax.set(xlabel='Time', ylabel='Cumulative mean squared error')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir/'delay_comparison.pdf')
    fig.savefig(args.output_dir/'delay_comparison.png', dpi=180)
    np.savez(args.output_dir/'rollout_errors.npz', starts=starts, **predictions)
    with (args.output_dir/'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output_dir/'protocol.json').write_text(json.dumps(dict(
        original_figure4_recovered=False, starts=starts.tolist(), horizon=args.horizon,
        confidence_interval='Normal approximation across starts: mean +/-1.96 sample std/sqrt(n); starts are correlated samples from one trajectory.',
        error='Mean over 3 state coordinates, then cumulative mean over lead times 1..horizon',
        checkpoints=[str(args.checkpoint_y0), str(args.checkpoint_y2)], device=args.device), indent=2)+'\n')


if __name__ == '__main__':
    main()
