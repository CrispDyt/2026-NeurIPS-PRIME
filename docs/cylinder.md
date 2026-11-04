# Flow past a cylinder (CFD)

This subset reproduces the CFD portions of the prediction and temporal-consistency tables from the supplied checkpoints. It contains the fixed nine-sensor dataset, 20 checkpoints (four methods × five seeds), the original metric routines with a portable command-line wrapper, and fresh-training entry points. No raw flow frames, notebooks, plots, or redundant experiment logs are bundled.

## Evaluation

From the repository root, with the environment in the root README:

```bash
python experiments/cylinder/prepare_data.py
python experiments/cylinder/evaluate.py --device cuda --output-dir results/cylinder
```

Defaults: seeds `100,200,13532,123456,123123123`; all four models; 32 evenly spaced starts per model/seed; prediction horizon 500 coarse steps, NMSE windows 20/100/400; DRTD horizon 1000 from test[0]. Use `--model PRIME --seeds 100` for one checkpoint. `--data-dir` and `--checkpoint-dir` accept relocated input directories. Output includes raw per-rollout/per-seed CSVs, summary CSVs, errors, and a provenance manifest. The unmodified historical CSVs are in `reference/cylinder/`.

The model uses forward Euler at dt=0.01; predictions are interpolated to the common dt=0.1 grid. DRTD sections are **zero-based channels 6, 4, 5**, each at physical value 4, with both crossing directions. The historical CSV section identifiers (`Sigma1_u0_4`, `Sigma2_u3_4`, `Sigma3_u2_4`) are retained for joins, but their old text labels were incorrect; new output labels state the actual channels. DRTD retains the historical sorted/truncated return-time comparison, rather than replacing it with another Wasserstein implementation. TCM and prediction summaries pool 160 rollouts per model; DRTD summaries use valid seeds; standard deviations use ddof=0. `Attractor_W2` corresponds to SD in the paper.

SciPy Welch and GeomLoss are required. The CLI refuses to substitute another PSD or transport metric. The old transport helper's exception-to-sliced-distance fallback has been removed; a GeomLoss failure is reported as an evaluation error. Successful Sinkhorn calculations retain the original numerics. CUDA and CPU can produce different long trajectories; use CUDA for comparison with the supplied historical tables and record the output manifest.

## Fresh training

```bash
python experiments/cylinder/train.py --model PRIME --seed 100 --device cuda
python experiments/cylinder/train.py --model MSE --seed 100 --device cpu
python experiments/cylinder/train.py --model DIM --seed 100 --device cuda
python experiments/cylinder/train.py --model 'DIM+MSE' --seed 100 --device cuda
```

Results default to `results/cylinder/trained/Seed100/`; evaluate them with `--checkpoint-dir results/cylinder/trained`. Existing checkpoints are not overwritten. Repeat for each of the five seeds. `--steps` overrides the training budget and consequently the blur schedule; `--steps 2` is a smoke test, not a paper run. Initialization is fresh; no bundled checkpoint is loaded for training.

The training-source audit found historical differences that the CLI preserves explicitly:

| Model | Default updates | Recipe / checkpoint selection |
|---|---:|---|
| MSE | 10000 | Two-hidden-layer 9→100→100→9 tanh; Adam lr=1e-3; pointwise MSE on four-coordinate delay blocks; minimum pre-update loss selects post-update weights |
| DIM | 3000 | Three-hidden-layer 9→100→100→100→9 tanh; Stage 1 only; delay marginal + state graph loss; final weights |
| DIM+MSE | 3000 | Same Stage 1 plus MSE weight 2; final weights |
| PRIME, seeds 100/200/123456 | 5000 | 3000 warmup updates, then one recurrence update and three Stage 1 updates per cycle; MSE weight 0; minimum recurrence loss selects pre-update weights |
| PRIME, seed 13532 | 4000 | Same schedule and MSE weight 0, shorter historical budget |
| PRIME, seed 123123123 | 5000 | Same schedule but historical MSE weight **2** |

All use StandardScaler fitted only to the 200 training observations, four delay coordinates at lag 2, and 10 microsteps per observation. Measure training uses 512 Stage 1 samples, blur 0.20→0.07, learning rates 1e-3/1e-5, gradient clipping 1, recurrence blocks of up to 16 hits, and eight recurrence blocks per update. The training section is channel 8 within one standardized unit of its training mean, positive derivative, separated by at least three coarse steps. This training selector is distinct from the evaluation sections above. Random-projection generators retain their original device and seed offsets. Diagnostic loss calls every 50 steps are retained because they consume random samples.

The alpha-MSE differences are inferred from saved histories and checked against the original computation: seed100 initial loss is 16.4622516632 with weight0 and 38.6487350464 with weight2; the latter exactly matches DIM+MSE. `new.py` gives 24.1174850464 and is therefore not the source of these selected models. The selected training source is `new_try.py`, with the historical seed/budget/weight changes above. This is evidence for the recovered recipe, not a claim that every historical optimization step has been replayed. `--alpha-mse` permits an explicit controlled override recorded in the manifest. Do not describe the archived five PRIME checkpoints as having identical hyperparameters.

## Data and provenance

`data/cylinder/flow_sensors.p` is the original 2.1MB sensor dataset: noisy training `(200,9)`; clean test `(15000,9)`; noisy test `(15000,9)`; raw dt .01; training/common dt .1. SHA-256: `b9c4e3ec0dfd1ad9b633e32a394a2b439069b2350b47fac5ace199d8544d6262`.

All source paths, SHA-256 hashes, copy sizes, and selected checkpoint hashes are in `reference/cylinder/provenance.json`. Source project: `Entropy/Entropy-1/Cylinder Case/`. `evaluate.py` was adapted by preserving numerical functions and replacing its Poincare-only main with a complete evaluation CLI. The comparison notebook supplied plotting/evaluation evidence; it does not contain the authoritative training routine. No experiment source files were modified.

Optional extraction from an existing trusted raw simulation file:

```bash
python experiments/cylinder/prepare_data.py --raw /path/to/flow_full70.p --seed 0 --output-dir data/cylinder_new_realization
```

The original raw file is approximately 1.2GB and is not bundled. The preprocessing code discards 3000 frames, extracts the same nine probes, applies the original spatial perturbation/interpolation scheme, and constructs the same split. The original perturbation RNG seed was not saved; re-extraction produces a **new** noise realization. Use bundled `flow_sensors.p` to reproduce the tables. Raw-field visualizations require the unbundled raw file. No verified public raw-data download URL was found in the local project.

The original simulator header credits Felix Köhler (2021), MIT-licensed code from `https://github.com/Ceyron/machine-learning-and-simulation`; its referenced CREDITS.md was absent locally. The optional portable simulator is `experiments/data_generation/simulate_cylinder.py`; the recovered upstream MIT license is retained in `licenses/Ceyron-MIT.txt`. It needs JAX separately and was not rerun at the full 200000-step budget. No separate license for the author's sensor dataset/checkpoints was found locally; repository owners must choose their release terms rather than treating the upstream simulator's license as a dataset license.

## Validation performed

Validated in Python 3.11.15, NumPy 1.26.0, PyTorch 2.1.2+cu121, scikit-learn 1.1.3, SciPy, and GeomLoss on an RTX 4060 Laptop GPU. Historical StandardScaler objects report scikit-learn 1.6.1; the loader adds NumPy `_core` aliases for compatibility with the saved pickles.

- Syntax and CLI help checks passed.
- Two-update fresh training passed for all four methods (CUDA measure losses; CPU MSE). Training compute time was approximately 0.4–1.6 seconds per smoke run, excluding imports.
- One-start/20-step CPU evaluation completed with one prediction row and three DRTD rows, no evaluation errors.
- A CUDA PRIME seed100 replay at the **full historical horizons** (500 prediction steps, 1000 DRTD steps) matched original NMSE, PSD, and all three DRTD values exactly; TCM differed by 3.1e-16, transport by 2.8e-9.

A complete 20-checkpoint evaluation and full fresh training were not rerun during repository assembly. Small smoke horizons do not reproduce the paper tables.
