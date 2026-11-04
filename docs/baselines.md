# Heterogeneous Lorenz63 forecasting baselines

This subset reproduces the source protocols for Table 4 and Figure 9. It includes NGRC/NVAR, reservoir ESN, LSTM, Koopman autoencoder, and the author's DySLIM-style MMD objective. DySLIM-style is an adapted PyTorch baseline, not the official DySLIM JAX implementation.

## Re-evaluate Table 4

```bash
python experiments/baselines/evaluate.py --device cuda --output-dir results/baselines
```

Defaults load the bundled five seed1000 baseline checkpoints and the **Table 4 PRIME checkpoint with internal seed13531**. Thirty shared forecast starts are sampled with seed1123; horizon=500, dt=.1; NMSE windows=100/400. Sequence baselines receive a ten-step history ending at t_start−1. PRIME uses only the observation at t_start−1. This is a comparison at common forecast times; the models do not receive equal history lengths. DRTD is averaged over the original three sections (u0=10, u2=40, unimodal-map events) within each rollout. It differs from the five-training-seed protocol of the main tables.

Results, LaTeX rows and an evaluation manifest are written to `results/baselines/combined_eval/`; generated rollout arrays and optional plots go into sibling directories. `--checkpoint-dir` changes the input location without changing the output directory. `--no_plot` skips plots. For a smoke check, use `--eval_starts 1 --horizon 400 --no_plot`; this does not reproduce Table 4.

The archived Table 4 source is `reference/baselines/combined_results_with_PRIME.csv`. The separate `results_summary.csv` is the training script's earlier evaluation: notably its ESN safety clipping/protocol differs from the combined evaluator. Do not use it as the Table 4 source. Original transport, PSD, recurrence, divergence and clipping routines are preserved; GeomLoss is required at startup. The original transport helper also retains its runtime-error sliced-distance fallback; an environment that triggers it does not reproduce the same transport metric.

## Figure 9

```bash
python experiments/baselines/plot.py --device cuda --output-dir results/baselines/figure9
```

This ports the actual `plot_lorenz.ipynb` first cell into a standalone CLI. Defaults: start=0, max history=10, shared origin at test index9, future horizon700. The same observed origin is prepended to all plotted predictions. Output is the original 2×3 phase-trajectory layout as PDF and PNG, with the full GT attractor and local GT segment.

The Figure 9 notebook used **a different PRIME checkpoint, internal seed1000**. It is deliberately saved separately as `checkpoints/baselines/PRIME_figure9_seed1000.p`; `Implicit_BEST.p` is the Table 4 seed13531 checkpoint. Their weights and recurrence hyperparameters differ. The plotting CLI accepts `--prime-checkpoint` for an explicit alternative but does not silently substitute one for the other.

The notebook referenced an undefined `GenericSequentialMap`. Its tanh-MLP helper was recovered from the adjacent `plot_recurrence_section.ipynb`, and loading now requires an exact state-dictionary match. Dead duplicate rollout wrappers and IPython display output were removed. Numerical rollout and plotting logic are retained.

## Fresh baseline training

```bash
python experiments/baselines/train.py --all --device cuda --output-dir results/baselines/trained
```

Use `--all` to preserve the original model order and shared RNG stream. Running an individual `--models lstm` starts the RNG stream anew and therefore is not guaranteed to reproduce the checkpoint from the original combined training run. Neural models use 3000 updates, batch 256, lr1e-3 and gradient clipping 1. NVAR and ESN use their original direct ridge-regression fits; `--steps` only changes neural optimization. Full hyperparameters are in `reference/baselines/run_config.json`; `train.py --help` exposes them. The common data lives in `data/lorenz63/` and is byte-identical to the baseline source data. The training scaler is fitted only on the training split.

Outputs include five new checkpoints, scaler, run config, evaluation CSV and optional plots. Existing output runs are not overwritten. To evaluate new checkpoints, use `--checkpoint-dir results/baselines/trained/checkpoints --prime_ckpt checkpoints/baselines/Implicit_BEST.p`. PRIME training is handled by the Lorenz experiment code, not this five-baseline trainer.

Timing caveat: the historical `results_summary.csv` column `train_seconds` includes training **and** the subsequent evaluation. The adapted trainer preserves that field but also records `train_only_seconds` and `save_and_evaluation_seconds`. None of these fields should be substituted for Table 21's per-update timing experiment; that experiment has its own benchmark/source CSVs.

## Provenance and validation

`reference/baselines/provenance.json` lists original source paths and SHA-256 hashes for code, notebook cells' container files, checkpoints and data. The source is `NIPS/Lorenz/`; only selected checkpoints (about 5.5MB) and small reference CSV/config files are copied. No original notebooks, large rollout archives or pre-rendered figures are bundled. No independent upstream license was found for these local implementations; release licensing is a repository-owner decision.

Validated in Python 3.11, NumPy 1.26, PyTorch 2.1.2+cu121 with SciPy, scikit-learn and GeomLoss:

- Syntax and train/evaluate/plot help passed.
- All five fresh baselines completed a smoke run (two updates for neural models; original direct solves for NVAR/ESN), including one-start 400-step evaluation.
- The bundled six-model Table 4 evaluator completed a one-start 400-step CPU smoke run.
- Figure 9's six-model plot completed a 20-step CPU smoke run; the generated PDF/PNG and identical-start assertions were checked.

These smoke runs validate execution and paths, not statistical agreement of complete reruns. The published full 30-start aggregate has been retained as an auditable reference. CPU/GPU/library changes can alter long autonomous rollouts; use the historical CUDA configuration for paper comparisons.
