# Lorenz63 reproduction

This subset contains the code and frozen inputs for the four-method Lorenz63
experiments in Tables 2 and 3. It uses the observed `(x, z)` coordinates, not the
full three-dimensional Lorenz state. The saved training array is `(500, 2)` and
the test array is `(1000, 2)`; observation and Euler integration steps are both
`0.1`. The arrays are copied without changes. They are the authoritative data
for reproduction: the notebook's later data-generation cell uses a different
long-data configuration and an unrecorded noise RNG state, so running that cell
would not regenerate these exact arrays.

## Evaluate the archived checkpoints

Run from the repository root in the environment described by the root README:

```bash
python experiments/lorenz63/evaluate.py --device cuda
```

The defaults evaluate all four methods and historical folder labels `33, 123,
1000, 13532, 123456`. Outputs go to `results/lorenz63/evaluation`; the checked-in
`reference/lorenz63` CSVs are never overwritten. A single-model command is:

```bash
python experiments/lorenz63/evaluate.py --model PRIME --seeds 33 --device cuda \
  --output-dir results/lorenz63/prime_seed33
```

`--data-dir`, `--checkpoint-dir` and `--output-dir` accept alternative directories.
Checkpoints are expected at `CHECKPOINT_DIR/seed_<folder label>/<filename>`.
The selected files are exactly `Implicit_BEST.p` (PRIME), `NODE_BEST.p` (MSE),
`DIM_BEST.pt` (DIM), and `DIM_MSE_BEST.pt` (DIM+MSE). Their original bytes,
including scaler objects and original training metadata, are preserved.
Only load these trusted bundled checkpoints; their historical format uses
Python pickle.

## Train from scratch

```bash
python experiments/lorenz63/train.py --model MSE --seed 33 --device cuda
python experiments/lorenz63/train.py --model DIM --seed 33 --device cuda
python experiments/lorenz63/train.py --model 'DIM+MSE' --seed 33 --device cuda
python experiments/lorenz63/train.py --model PRIME --seed 33 --device cuda
```

Fresh outputs go to `results/lorenz63/trained/seed_<actual seed>`. The scripts
refuse to replace an existing model's run. To evaluate freshly trained models,
pass `--checkpoint-dir results/lorenz63/trained`. Repeat the commands for the
other desired actual seeds. `--seed` always sets the requested actual RNG seed;
it never remaps a seed to a different number.

All models use the original three hidden layers of width 100 with Tanh
activations and Adam at `1e-3`. The baseline defaults are 10,000 iterations,
batch 512, two Euler steps for the pair loss, and the original float32
MaxAbsScaler fitting. DIM uses energy-distance matching of the future state
cloud and five-coordinate delays of `x`, plus a fixed initial-trajectory anchor
loss with weight 1. DIM+MSE uses the same two energy losses plus the paired
future-state MSE with weight 0.1. These are distinct losses despite both
checkpoint configurations historically naming the coefficient `w_mse`.

PRIME fits its scaler to the float64 training array, builds recurrence events
from training only, and runs 4,000 iterations: 3,000 stage-1 warm-up iterations,
then alternating stage-2 and stage-1 iterations. Stage 2 also includes the
stage-1 objective. It uses the preserved Sinkhorn graph/delay losses,
32-event blocks, 16 sampled blocks per stage-2 update, and a blur schedule
from 0.2 to 0.1. `Implicit_BEST.p` is the minimum total-loss checkpoint among
stage-2 updates, not the minimum stage-2-only loss. The alternate
`Implicit_BEST_L2.p` and final checkpoint are also saved for inspection.

Checkpoint selection preserves the historical convention for every method:
the scalar loss is evaluated before `optimizer.step()`, and the corresponding
saved weights are taken after that update. No validation or test loss selects
the checkpoint. Training histories and a runtime/data-hash manifest are saved.

## Historical metadata differences

The archived PRIME file under **folder label 13532 records actual seed 13531**.
It also records a different recurrence configuration:

| Configuration | PRIME labels 33, 123, 1000, 123456 | PRIME label 13532 |
|---|---:|---:|
| Stored actual seed | same as label | 13531 |
| Candidate fraction | 0.5 | 0.2 |
| Requested landmarks | 200 | 128 |
| Stage-2 weight | 0.5 | 0.3 |
| Warm-up / total iterations | 3000 / 4000 | 3000 / 4000 |

The reference tables retain the historical folder identity so rows can be
traced back to the original records. The evaluator emits a warning and records
both folder label and internal seed in its manifest. The exceptional historical
configuration is explicitly available as:

```bash
python experiments/lorenz63/train.py --model PRIME --seed 13531 \
  --profile legacy13532 --device cuda \
  --output-dir results/lorenz63/legacy_prime
```

This writes `seed_13531`, accurately reflecting the requested random seed. A
fresh `--seed 13532` run under the standard profile is a new experiment, not a
recreation of that archived file. This discrepancy was preserved and documented;
no new training was substituted into the reference table during repository
assembly. The baseline archives have no internal seed field, so their seed
identity is established only by source-folder provenance. New runs record it.

The current `Lorenz.ipynb` cell 10 has `warm_steps=0` and a `NoWarmUp` save prefix,
whereas all five selected PRIME checkpoints record `warm_steps=3000`. This port
uses the checkpoint-recorded 3000-step default. It does not use the later
`PRIME.ipynb` or `Generated PRIME.ipynb` experiments, which concern different
long-data / generative-roof variants.

## Exact evaluation conventions

`metrics.py` retains the numerical functions of the original `evaluate.py`.
The new entry point changes paths, model/seed selection, device selection,
error handling and metadata only; dependencies must be present rather than
silently selecting the old fallback metrics.

- Prediction metrics and TCM use 32 deterministic starts from
  `linspace(0, 498, 32).astype(int)`, a 500-step rollout (501 states), and the
  same starts for every seed and method. NMSE windows include the initial
  state and their endpoint. The original NMSE normalization is unchanged.
- `Attractor_W2` is the archived GeomLoss Sinkhorn quantity reported as SD in
  the paper (blur 0.05, at most 1000 points). `PSD_logL1` is Log-PSD, with the
  original SciPy Welch settings and `nfft=2048`.
- TCM uses both observed channels, lag cap 200, and the original correlation
  sum without first-nonpositive-lag truncation.
- DRTD uses one 999-step rollout from `test_data[0]` per model. Paper sections
  are `Sigma1 = x=10` positive crossings, `Sigma2 = z=40` positive crossings,
  and `Sigma3 = successive local maxima of z`. The third section retains
  prominence 1.0 and quadratic peak-time refinement. Return-time W1 retains
  the source implementation.
- Means and population standard deviations for prediction/TCM pool 5×32
  rollouts; DRTD aggregates finite values across model seeds. Missing DRTD is
  not zero: MSE has 2 valid Sigma1 results and 1 valid Sigma2 result;
  DIM+MSE has 3 valid Sigma2 results. Other DRTD cells have 5 valid seeds.
- The filenames retain `5seeds_32starts` for compatibility even in a reduced
  smoke evaluation. Always check the accompanying runtime manifest before
  interpreting a result as the full paper protocol.

The default device is explicitly CUDA. A missing GPU fails clearly; it is not
silently replaced by CPU. Historical long chaotic trajectories can diverge
under a different backend, PyTorch/library version, or hardware. The archived
raw CSVs remain the record of the reported experiment; rerunning training on a
new numerical runtime is not guaranteed to recover every archived weight or
long-trajectory metric bit for bit. The port was tested with Python 3.11,
PyTorch 2.1.2+cu121 and NumPy 1.26.0. Its loader supports old NumPy 2 pickle
module names when running NumPy 1.x.

## Lightweight verification performed

```bash
python experiments/lorenz63/train.py --model MSE --seed 33 --steps 2 \
  --device cpu --output-dir /tmp/lorenz63_mse_smoke
python experiments/lorenz63/train.py --model PRIME --seed 33 --steps 2 \
  --warm-steps 1 --device cpu --output-dir /tmp/lorenz63_prime_smoke
python experiments/lorenz63/evaluate.py --model PRIME --seeds 33 --device cpu \
  --n-starts 1 --pred-horizon 20 --poincare-horizon 20 \
  --output-dir /tmp/lorenz63_eval_smoke
```

All four methods passed a two-iteration CPU smoke run, including both PRIME
stages. The baseline models' resulting weights and best losses matched
execution of their original notebook cells bit for bit. All 38 retained numeric
metric/model helper definitions matched the original AST; only the checkpoint
loader adds compatibility aliases. PRIME's seed-33 recurrence geometry matched
the archived hit indices and calibration constants. The 640 prediction and 20
Poincare reference rows reaggregated to every archived numeric summary metric.
The short evaluation completed without errors. A CUDA replay of archived PRIME
seed 33, start 0, reproduced the 500-step prediction and 999-step DRTD
metrics (DRTD exactly; prediction differences below 1e-5). These tests do not constitute a
new full five-seed training run.

`reference/lorenz63/port_validation.json` records these checks.
`reference/lorenz63/provenance.json` records source-relative paths, SHA-256 hashes
and byte sizes of all bundled arrays, checkpoints and raw results, as well as
hashes of the source notebook and evaluator. The original NIPS files were not
modified. Only the main-table data/checkpoints were selected; notebook outputs,
plots, timestamped duplicate models and experiment logs are excluded.
