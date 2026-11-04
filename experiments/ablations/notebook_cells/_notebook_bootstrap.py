"""Portable inputs for extracted notebook cells; no numerical experiment changes."""
import argparse
from pathlib import Path
import os
import sys


def prepare(script):
    repo = Path(__file__).resolve().parents[3]
    task = Path(script).stem
    cpu_original = task in {'variable_lag_eval', 'prime_only_eval', 'prime_w2_eval',
                            'dim_mse_w2_eval', 'graph_specificity_eval'}
    p = argparse.ArgumentParser(description=f'Extracted {task}; historical numerical defaults, portable paths.')
    p.add_argument('--data-dir', type=Path, default=repo / 'data/lorenz63')
    p.add_argument('--checkpoint-dir', type=Path, default=repo / 'checkpoints/ablations')
    p.add_argument('--output-dir', type=Path, default=repo / 'results/ablations' / task)
    p.add_argument('--seed', type=int)
    p.add_argument('--steps', type=int, help='Training override; PRIME-only means stage-2 steps.')
    p.add_argument('--warm-steps', type=int)
    p.add_argument('--fixed-lag', type=int, choices=[3,6,9,12], default=12)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu' if cpu_original else 'cuda')
    p.add_argument('--plots', action='store_true')
    args = p.parse_args()
    if args.steps is not None and args.steps < 1: p.error('--steps must be positive')
    if args.warm_steps is not None and args.warm_steps < 0: p.error('--warm-steps must be nonnegative')
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scipy.signal
    import sklearn.preprocessing
    import geomloss
    # Import compiled extensions before making legacy NumPy pickle aliases available.
    if np.__version__.startswith('1.'):
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable. Explicit CPU use changes historical GPU trajectories.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('Output directory is not empty; choose a fresh --output-dir')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if not args.plots:
        plt.show = lambda *a, **k: plt.close('all')
    train = np.load(args.data_dir / 'train_data.npy')
    test = np.load(args.data_dir / 'test_data.npy')
    return dict(PORT_REPO=repo, PORT_ARGS=args, PORT_DATA=args.data_dir,
                PORT_CHECKPOINTS=args.checkpoint_dir, PORT_OUTPUT=args.output_dir,
                PORT_DEVICE=args.device, train_data=train, test_data=test,
                dt_obs=.1, dt_base=.1, sub=1, os=os, np=np, pd=pd, torch=torch, plt=plt,
                display=lambda value: print(value.to_string() if hasattr(value, "to_string") else value))
