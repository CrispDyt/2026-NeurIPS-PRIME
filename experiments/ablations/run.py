#!/usr/bin/env python3
"""Run a curated ablation implementation with repository-relative data and outputs.

Use `python experiments/ablations/run.py TASK --help` for that task's original options.
Notebook task names end in _train or _eval and share portable --data-dir,
--checkpoint-dir, --output-dir and explicit --device options.
"""
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LEGACY = {
 'schedule_train': ('Ab_Lorenz_1.py', 'lorenz_ablation_K'),
 'section_train': ('Ab_Lorenz_2.py', 'lorenz_ablation_section'),
 'explicit_band_train': ('Explicit40.py', 'lorenz_ablation_section'),
 'loss_terms_train': ('Ab_Lorenz_3.py', 'lorenz_ablation_loss_terms'),
 'clock_train': ('recurrence_clock_ablation.py', 'lorenz_clock_robustness'),
 'clock_eval': ('evaluate_clock_robustness_results.py', 'lorenz_clock_robustness'),
 'speed_eval': ('parameter_abalation.py', 'speed_threshold'),
 'speed_train': ('implicit_poincare_multiseed_train_eval_v2.py', 'speed_threshold'),
 'section_diagnostics': ('recurrence_section_diagnostics.py', 'section_diagnostics'),
}
NOTEBOOK = [p.stem for p in (HERE/'notebook_cells').glob('*.py') if not p.name.startswith('_')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',choices=sorted(list(LEGACY)+NOTEBOOK))
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    args=p.parse_args()
    if args.task in NOTEBOOK:
        path=HERE/'notebook_cells'/f'{args.task}.py'
        sys.path.insert(0,str(path.parent));sys.argv=[str(path)]+args.arguments
        runpy.run_path(str(path),run_name='__main__');return
    filename,subdir=LEGACY[args.task];path=HERE/'legacy'/filename
    data=ROOT/'data/lorenz63';output=ROOT/'results/ablations'/subdir
    if args.task=='speed_eval':
        defaults=['--root',str(ROOT/'checkpoints/ablations/speed_threshold'),
                  '--test_path',str(data/'test_data.npy'),'--device','cuda','--out_dir',str(output/'evaluation')]
    elif args.task=='speed_train':
        defaults=['--train',str(data/'train_data.npy'),'--test',str(data/'test_data.npy'),
                  '--out_dir',str(output),'--seeds','123,456,789,2024,3407']
    elif args.task=='section_diagnostics':
        defaults=['--train_path',str(data/'train_data.npy'),'--test_path',str(data/'test_data.npy'),'--out_dir',str(output)]
    elif args.task=='clock_eval':
        ckpts=ROOT/'checkpoints/ablations/lorenz_clock_robustness'
        defaults=['--train_path',str(data/'train_data.npy'),'--test_path',str(data/'test_data.npy'),
                  '--run_root',str(ckpts),'--summary_csv',str(ROOT/'reference/ablations/lorenz_clock_robustness/clockrobust_summary_seed1000.csv'),
                  '--output_dir',str(output/'evaluation')]
    else:
        defaults=['--train_path',str(data/'train_data.npy'),'--run_root',str(output)]
    # Initialize optional compiled dependencies before enabling old NumPy checkpoint aliases.
    import numpy as np
    import pandas
    import scipy.signal
    import sklearn.preprocessing
    import geomloss
    import torch
    if np.__version__.startswith('1.'):
        sys.modules.setdefault('numpy._core',np.core)
        sys.modules.setdefault('numpy._core.multiarray',np.core.multiarray)
        sys.modules.setdefault('numpy._core.numeric',np.core.numeric)
    sys.path.insert(0,str(path.parent));sys.argv=[str(path)]+defaults+args.arguments
    runpy.run_path(str(path),run_name='__main__')

if __name__=='__main__':main()
