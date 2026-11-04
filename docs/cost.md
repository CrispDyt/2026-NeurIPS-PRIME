# Computational cost (Tables 19–21)

`reference/cost` preserves the recorded measurements used by the paper. Timings depend on hardware, software versions and system load; new runs produce new measurements rather than the same wall-clock numbers.

## Table 19: training cost

```bash
python experiments/cost/profile_training.py --benchmark all --seeds 1000 --device cuda
```

This uses the portable copy of `NIPS/Lorenz/Computational_Cost.py`, which is byte-identical to `Entropy/Entropy-1/KS/Computational_Cost.py`. The only port changes are repository-relative default paths, output-directory CLI aliases, explicit CUDA default, seed1000 matching the archived runs, and nonzero exit status when a run fails. Training objectives and profiling calculations are unchanged. Outputs default to `runs/cost/training`.

The profiler trains models from scratch; it does not load the accuracy checkpoints or overwrite them. It reports setup time separately, training time, average iteration time, CUDA peak allocated/reserved memory, CPU RSS and parameter count. Table19 uses `train_sec`, `avg_iter_ms` and `peak_cuda_alloc_mb`. Each reference row is a single seed1000 run; its standard deviation is therefore zero. The archived GPU was an RTX4060 Laptop GPU with8GB VRAM.

The profiler's historical training configuration is embedded in each raw CSV record. It is a separate profiling script: its DIM/DIM+MSE objectives and some schedules differ from the accuracy-training notebooks. For example, the KS profiler uses the common Sinkhorn objective with an additional MSE term of0.1 for DIM, while the main KS DIM cell uses energy distances without that term. Lorenz PRIME profiling uses4000 steps rather than the latest general training recipe. These differences are preserved for Table19 provenance and should not be described as a full runtime benchmark of every archived accuracy checkpoint.

For a short installation check, without a full benchmark:

```bash
python experiments/cost/profile_training.py --benchmark all --methods mse --quick --device cpu --output-dir runs/cost/smoke_training
```

`--quick` reduces baseline runs to50 steps, PRIME to80 steps with40 warmup steps, and also reduces batch/block/landmark sizes. It is not comparable to the reference timing. `--lorenz-train`, `--ks-train`, `--methods` (comma separated), and `--output-dir` allow explicit inputs.

Reference files are under `reference/cost/lorenz/` and `reference/cost/ks/`.

## Table 20: inference cost

```bash
python experiments/cost/profile_inference.py --device cuda
```

This ports code cell5 of `NIPS/Lorenz/1-step MSE/pP3L_Rebuttal.ipynb`. It loads the repository's20 Lorenz63 accuracy checkpoints: four methods across seeds33,123,1000,13532,123456. `NODE` is the source label for the MSE model. All four architectures have20702 trainable parameters.

Only core autoregressive inference is timed: batch size1,1000 Euler steps, five full warmup rollouts and20 timed repetitions per checkpoint. CUDA is synchronized around timing boundaries. Checkpoint/data loading, scaling, inverse scaling and metric calculations are excluded. The per-checkpoint result is the median of20 repetitions; the displayed mean and population standard deviation are then computed across five seeds. At1000 steps, a rollout time in milliseconds is numerically equal to microseconds per step.

The source loader is replaced by the verified Lorenz loader; the core Euler update and timing/statistical definitions are preserved. New summary column names use `time_rollout_ms_*` so shorter user-requested horizons are not mislabeled as1000-step results.

```bash
python experiments/cost/profile_inference.py --quick --seeds 33 --device cpu --output-dir runs/cost/smoke_inference
```

`--quick` uses10 steps, one warmup and two repetitions. Normal defaults write to `runs/cost/inference`. `--methods`, `--seeds`, `--rollout-steps`, `--warmup-runs` and `--timed-repeats` are explicit controls. No training is performed by this command.

Reference results and hardware metadata are `reference/cost/inference_time_per_checkpoint.csv`, `inference_time_summary_by_method.csv` and `inference_time_metadata.json`.

## Table 21: average update cost

The first four values are the Lorenz rows of Table19. The remaining LSTM, Koopman-AE and DySLIM-style values are in `reference/cost/heterogeneous/training_time_summary.csv`, copied from the independent five-seed heterogeneous training benchmark:

| Method | Recorded mean milliseconds per update |
|---|---:|
| LSTM | 1.932845175 |
| Koopman-AE | 1.727480744 |
| DySLIM-style | 103.267651157 |

Raw per-seed records and hardware/configuration metadata accompany that summary. The exact training-time harness was recovered from the fully commented code cell at **index 6 (zero-based)** of `NIPS/Lorenz/1-step MSE/pP3L_Rebuttal.ipynb`.

```bash
python experiments/cost/profile_heterogeneous.py --device cuda
```

Defaults reproduce the source benchmark settings: seeds `33 123 123456 1000 13532`; methods `ngrc esn lstm koopman dyslim`; **4000 neural optimization steps**, batch 256, sequence length 10, and the original network/regularization parameters. This timing profile differs from the 3000-step Table 4 baseline training run. Each model/seed run resets its RNG seed independently. NVAR and ESN use closed-form NumPy/CPU fits and have no per-update value.

The timer includes model construction, tensor preparation inside the training routine, and training. It excludes file loading, scaling, checkpoint serialization, evaluation and plotting. A CUDA matrix multiplication warms the device before benchmarking, and CUDA synchronization brackets the timer. `milliseconds_per_update` is **1000 × total timed seconds / number of optimization updates**, so it includes amortized construction and startup overhead; it is not a separately sampled steady-state update latency. The seed summary preserves pandas' sample standard deviation (`ddof=1`), which is undefined for a one-seed smoke run.

The port reuses 24 baseline helper/model/training definitions verified AST-identical to the uncommented source instead of duplicating the full baseline module. IPython Styler and Markdown-table rendering are removed. Optional `--save-checkpoints` saves fitted models after the timer; it is off by default. `--steps`, `--seeds`, `--models`, `--data-dir`, `--output-dir` and `--device` expose the benchmark controls. An optional `--threads` value is recorded; omitting it preserves the process's CPU thread configuration. Outputs default to `runs/cost/heterogeneous`, and completed runs are not overwritten.

```bash
python experiments/cost/profile_heterogeneous.py --steps 2 --seeds 33 \
  --models lstm --device cpu --threads 1 \
  --output-dir runs/cost/smoke_heterogeneous
```

This two-update run checks execution and timer output only; its amortized startup cost is not comparable to the paper's 4000-update timing.

## Validation and provenance

All three CLIs passed syntax and `--help` checks. The CPU quick training check executes MSE on both Lorenz63 and KS; the CPU quick inference check loads and times all four Lorenz63 methods at seed33 and verifies20702 parameters each. The heterogeneous CLI passed a two-update LSTM CPU smoke run at seed33 (216450 parameters), producing one successful raw timing row and the matching one-seed summary. No full training benchmark or CUDA timing benchmark was run during repository curation. Validation details are in `reference/cost/port_validation.json`; original file hashes and source paths are in `reference/cost/provenance.json`.
