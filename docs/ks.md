# Kuramoto–Sivashinsky reproduction

This subset contains the exact archived five-sensor data, the 20 selected checkpoints used by the current Table 2/3 result files, an evaluator, and standalone training for all four methods. The corrected MSE checkpoint for seed `123123` is included; the older duplicated checkpoint is excluded.

## Evaluate the archived models

From the repository root, with the reproduction environment activated:

```bash
python experiments/ks/evaluate.py --device cuda
```

This evaluates the four methods and folder seeds `4242 10000 13531 123123 123123123`, writing results under `runs/ks/evaluation`. To evaluate a subset:

```bash
python experiments/ks/evaluate.py --model MSE --seeds 123123 --device cuda --output-dir runs/ks/mse_123123
```

Input paths are explicit and independent of the working directory: `--data-dir` defaults to `data/ks`, and `--checkpoint-dir` to `checkpoints/ks`. Fresh training uses the same `partial_seed_SEED` checkpoint directory convention, so a training output tree can be supplied to `--checkpoint-dir`.

The default is CUDA because the archived reference evaluation used CUDA. CPU is supported with `--device cpu`; rounding differences in chaotic long trajectories can substantially change the metrics, so CPU results should not be mixed into the archived CUDA summaries. The NumPy compatibility aliases are only for loading NumPy 2 pickle names in archived files. Some archived scalers were saved with scikit-learn 1.6.1; the tested environment uses 1.1.3 and preserves their saved scale values.

## Train from scratch

```bash
python experiments/ks/train.py --model MSE --seed 123123 --device cuda
python experiments/ks/train.py --model DIM --seed 4242 --device cuda
python experiments/ks/train.py --model 'DIM+MSE' --seed 4242 --device cuda
python experiments/ks/train.py --model PRIME --seed 4242 --device cuda
```

The default output is `runs/ks/train/partial_seed_SEED`, so the shipped checkpoints remain intact. Existing BEST checkpoints are never overwritten by training; select a new `--output-dir` for another run. `--seed` always controls the actual RNG seed without folder-specific substitutions.

The scripts port `KS_test.ipynb` code cells 13, 15, 17 and 19 (one-based) from `Entropy/Entropy-1/KS`. All networks are 5→100→100→100→5 Tanh MLPs with a training-fitted MaxAbsScaler and Adam at 0.001. The observed interval and Euler step are both 1.0. The `dt_obs=0.1` field in the older NPZ is superseded by these notebook training cells and is not used.

| Method | Default steps | Historical training objective |
|---|---:|---|
| MSE | 10,000 | Pointwise MSE after two Euler steps, batch 512 |
| DIM | 10,000 | Energy distance of the two-step pushforward and multichannel delay clouds |
| DIM+MSE | 10,000 | DIM objective plus the two-step pointwise MSE with weight 1 |
| PRIME | 12,000 | 10,000 warmup steps, then 2,000 steps of the joint DIM/PRIME objective, PRIME weight 3 |

DIM uses delay lag 10, dimension 3, all five channels and the notebook's reversed delay order. The DIM source stores `w_glue=0.1` in its checkpoint metadata, but never adds a glue term to its loss; the port preserves the actual loss. PRIME uses Sinkhorn blur 0.2→0.1, 16 recurrence blocks per update, up to 16 events per block, state/delay weights 1/0.7 and two marginal weights of 0.2. Both baseline and PRIME BEST selection preserve the historical convention: the measured loss precedes the optimizer update, and saved weights follow it. PRIME BEST selection is restricted to stage 2.

For a minimal local installation check:

```bash
python experiments/ks/train.py --model MSE --steps 2 --device cpu --output-dir runs/ks/smoke/mse
python experiments/ks/train.py --model DIM --steps 2 --device cpu --output-dir runs/ks/smoke/dim
python experiments/ks/train.py --model 'DIM+MSE' --steps 2 --device cpu --output-dir runs/ks/smoke/dim_mse
python experiments/ks/train.py --model PRIME --steps 2 --warmup-steps 1 --device cpu --output-dir runs/ks/smoke/prime
python experiments/ks/evaluate.py --model MSE --seeds 4242 --n-starts 1 --pred-horizon 40 --poincare-rollout-len 40 --device cpu --output-dir runs/ks/smoke/eval
```

A shortened horizon is a smoke test, not a paper result: NMSE windows longer than that horizon are clipped by the historical metric implementation. A PRIME run shorter than its warmup produces FINAL only, with no stage-2 BEST; use `--warmup-steps 1 --steps 2` to exercise both stages.

## Archived training provenance limitations

Evaluating archived checkpoints reproduces the recorded experiment, subject to backend/version numerical sensitivity. Retraining the current notebook profile is a separate operation. The checkpoint archive has two material inconsistencies that must not be hidden:

1. PRIME does not have one common archived training profile across five seeds:

   | Folder seed | Total steps | Warmup | PRIME weight | Blur start→end |
   |---|---:|---:|---:|---|
   | 4242 | 12,000 | 10,000 | 3 | 0.2→0.1 |
   | 10000 | 4,000 | 3,000 | 1 | 0.2→0.1 |
   | 13531 | 11,000 | 10,000 | 1 | 0.1→0.05 |
   | 123123 | 12,000 | 10,000 | 3 | 0.2→0.1 |
   | 123123123 | 12,000 | 10,000 | 3 | 0.2→0.1 |

   The current notebook and its available repository revision contain the 12,000-step profile. The exact older loss implementations are not recoverable from metadata alone, so the training CLI implements the verified current notebook profile for every requested seed and does not claim to recreate the older two models exactly.

2. `partial_seed_13531/DIM_MSE_BEST.pt` has internal metadata seed **200**. The historical tables grouped this artifact by folder label 13531. Its bytes and reference results are preserved, and evaluation emits a warning. The CLI does not rewrite its metadata or silently change a requested training seed. The cause of the mismatch remains unresolved.

All 20 artifacts, their scalar training configuration, actual metadata seed, SHA-256 and source path are recorded in `reference/ks/provenance.json`. The correct newly trained MSE seed123123 has BEST iteration 9879 and training loss 0.0011262656189501286.

## Evaluation definitions and sections

- `train.npy`: 500×5 float32; `test_full.npy`: 2500×5 float32.
- Prediction/TCM: 32 starts `linspace(0,1499,32,dtype=int)`, 1,000 single Euler steps per rollout, including the initial state. This is 160 rollouts per method across five folder seeds.
- `NMSE_20/100/400`: the original mean-square-error divided by the corresponding reference mean-square magnitude, including time zero.
- `Attractor_W2`: the archived column name for the paper's Sinkhorn divergence (SD), GeomLoss `SamplesLoss(loss='sinkhorn',p=2,blur=0.05,backend='tensorized')`. It is not a square-root Wasserstein distance.
- `PSD_logL1`: mean absolute log10 Welch spectral discrepancy over five channels. `TCM`: difference in channel-averaged integrated autocorrelation times, lag cap 200, with no first-nonpositive truncation.
- DRTD: test start zero, 1,000 Euler steps, linearly interpolated zero crossings in both directions; return intervals are differences between consecutive crossing times.
- DRTD distance retains the historical `wasserstein_1d_empirical`: sort each interval array, truncate both to the smaller length, then average absolute differences. This is not replaced by SciPy's unequal-sample empirical W1 implementation.
- Missing intervals yield NaN and are excluded from aggregates. Means and standard deviations use finite values and population standard deviation (`ddof=0`).

Paper order and historical CSV IDs:

| Paper column | Sensor channel | Threshold | Historical CSV section ID |
|---|---:|---:|---|
| Σ1 | 0 | 0 | `Sigma1_u0_0` |
| Σ2 | 1 | 0 | `Sigma3_u5_0` |
| Σ3 | 2 | 0 | `Sigma2_u3_0` |

The source IDs contain stale coordinate names. They are retained for joining references; actual channel indices above control computation. New labels are corrected and rows are ordered as in the paper. Original reference CSVs are copied byte-for-byte and therefore retain older labels.

Metric dependencies are mandatory: no FFT/sliced-W2 fallback is used when Welch or GeomLoss fails. Errors are recorded in `eval_errors.json` and cause a nonzero exit status.

## Data generation

The exact archived arrays are bundled for metric reproduction. `prepare_data.py` provides the simulator and sensor sampling described by notebook cells 1 and 4: domain length100, 200 grid points, pseudo-spectral exponential stepping at0.1 with 2/3 de-aliasing, Gaussian initial perturbation using JAX key0, sensor indices0/50/80/100/150, steps10000:40000 sampled every10. The first500 observations train the model, and the remaining2500 are the clean evaluation sequence. Training noise has default standard deviation0.25.

```bash
python experiments/ks/prepare_data.py --output-dir runs/ks/generated_data --noise-seed 0 --device cpu
```

JAX is needed only for this optional command. The historical training-noise RNG state was not recorded; this command makes a new deterministic noise draw controlled by `--noise-seed`. JAX/device differences can also change a chaotic trajectory. It does not claim to regenerate the bundled arrays byte-for-byte. Use bundled data to evaluate or retrain against the archived experiment. The optional generator was syntax/help checked, not run to completion during curation.

## Source and checks

The selected artifacts come from `Entropy/Entropy-1/KS/partial_seed_*`; reference outputs come from `ks_full_metrics_original_tauW1_retrained_results`. Raw CSVs contain640 prediction records and60 DRTD records. Hashes of the source notebook, evaluator, data and checkpoints are in `reference/ks/provenance.json`.

Tested runtime: Python3.11.15, PyTorch2.1.2+cu121, NumPy1.26.0, scikit-learn1.1.3. All four two-step CPU training smoke runs passed, including both PRIME stages. On this environment, these short runs took approximately3–5 seconds each including Python imports; they do not measure full-training cost. See the repository validation results for numerical source-parity checks. Full retraining of the archived experiments was not performed during curation.
