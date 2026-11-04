# Paper-to-code coverage

Audited against the supplied `2026_NIPS_PRIME.pdf` on2026-11-04. This inventory distinguishes recovered source code/assets from an independently completed numerical reproduction. Original local folders were read as sources; their contents were not modified.

## Tables

| Paper item | Repository entry point / reference | Recovery status |
|---|---|---|
| Table1 | Theoretical comparison | No numerical experiment required |
| Tables2,3 | `experiments/{lorenz63,ks,cylinder}/{train,evaluate}.py`; `scripts/build_tables.py` | All60 selected main checkpoints, exact small observation arrays,1920 raw prediction rows and180 DRTD scalar slots retained |
| Table4 | `experiments/baselines/{train,evaluate}.py`; `reference/baselines/combined_results_with_PRIME.csv` | Five baseline trainers, selected models and comparison PRIME recovered; fresh short runs tested |
| Table5(a/b) | `experiments/ablations/run.py schedule_eval` / `section_eval` | Schedule/section train/evaluation sources, selected models and reference CSVs recovered |
| Table6 | `fixed_lag_train`, `fixed_lag_eval`, `sequential_lag_train`, `variable_lag_eval` through ablation runner | Fixed-lag and sequential training sources recovered; empirical-variable-lag checkpoint/evaluation recovered but its exact training cell not located. Reference rows mix evaluation cohorts; see ablation guide |
| Table7 | `prime_w2_eval`, `prime_only_eval`, `dim_mse_w2_eval`; `prime_only_train` | Separate full PRIME / PRIME-only / DIM+MSE evaluation protocols recovered; not the160-rollout main-table SD protocol |
| Tables8–13 | `scripts/build_tables.py` | All seed-level rows and cross-seed quartiles computed from the same source data as Tables2/3 |
| Table14 | `loss_terms_train`, `loss_terms_eval` | Shared-warmstart loss-term ablation source/checkpoints/reference recovered |
| Table15 | `graph_specificity_train`, `graph_specificity_eval` | Five-seed graph/marginal/pairwise-loss ablation source and references recovered |
| Table16 | `speed_train`, `speed_eval`; `reference/ablations/speed_threshold/` | Multiseed training/evaluation source and four q-speed raw/config/aggregate records recovered; all20 selected sweep checkpoints are bundled |
| Table17 | `clock_train`, `clock_eval` | Clock-jitter code, selected checkpoints and references recovered |
| Table18 | `section_diagnostics` | Pre-training extractor diagnostic code and per-seed/aggregate records recovered |
| Table19 | `experiments/cost/profile_training.py` | Historical profiler and timing records recovered; some profiling objectives differ from accuracy trainers |
| Table20 | `experiments/cost/profile_inference.py` | Parameter/inference timing code and raw records recovered |
| Table21 | `experiments/cost/profile_heterogeneous.py`, `reference/cost/heterogeneous/`, Table19 Lorenz rows | Independent heterogeneous pure-training timing code and records recovered |

Ablation commands above are subcommands of `python experiments/ablations/run.py`; use `TASK --help` and [the ablation guide](ablations.md) for precise training/evaluation defaults. Several original evaluators intentionally use different horizons, rollout counts or selected checkpoints; these have not been unified into the main-table protocol.

## Figures

| Figure | Code | Recovery limit |
|---|---|---|
| 1 | `experiments/figures/lorenz_phase_return.py` | Plotting recovered; original four-checkpoint cohort not fully recovered; inputs required explicitly |
| 2 | `experiments/figures/lorenz_collapse.py` | Original covariance/selected-trajectory plot logic recovered; exact old single-step checkpoint identity unresolved |
| 3 | `experiments/figures/cylinder_forecast.py` | Probe forecasts use small bundled inputs; flow snapshot needs unbundled raw simulation; exact published panel/model selection unverified |
| 4 | `experiments/rossler/train_dim.py`, `compare_delay.py` | Candidate DIM source recovered; exact original y0/y2 models and original CI calculation missing. Comparison utility implements a new explicit protocol |
| 5 | `experiments/figures/lorenz_30_rollouts.py` | Plotting recovered; explicit checkpoint selection required |
| 6 | `experiments/figures/lorenz_multistep_rollouts.py` | Original Euler-NODE checkpoint and plot recovered;20 microsteps is not20 observation targets |
| 7 | `experiments/figures/ks_one_step_rollouts.py` | Original plotted seed13531 MSE checkpoint and plotting cell recovered |
| 8 | `experiments/figures/ks_multistep_rollouts.py` | Original multi-step checkpoint and plotting cell recovered; its dt/stride differ from main tables |
| 9 | `experiments/baselines/plot.py` | Actual plotting cell and its separate PRIME checkpoint recovered |

See [figures](figures.md) for commands and implementation differences. Rendering a source plot is not proof that it matches every published panel pixel or failure-mode count.

## Differences that require a manuscript or experimental decision

The repository preserves evidence for these differences; no historical values were silently relabeled or recomputed under another objective:

1. **Model/seed identities:** Lorenz PRIME folder13532 contains seed13531; KS DIM+MSE folder13531 contains metadata seed200. Baseline files without saved seed metadata can only be traced by their folder/source hash.
2. **Training profiles:** Lorenz and KS PRIME checkpoints do not all use one uniform profile. Some earlier KS PRIME loss routines cannot be recovered exactly from current notebook state. Cylinder PRIME seed123123123 used an MSE coefficient2, while four other PRIME checkpoints used0; seed13532 has a shorter run.
3. **Architecture:** cylinder MSE has two hidden layers; the selected cylinder DIM/DIM+MSE/PRIME models have three. The manuscript's shared-backbone statement does not exactly describe these selected checkpoints.
4. **Metric definitions:** historical DRTD sorts and truncates return-time arrays before comparing them, which is not the general unequal-sample1D Wasserstein formula. Some original SD helpers can fall back to a sliced metric; main-table CLIs require the intended dependencies and preserve the original calculation. Main DIM baselines use energy losses in several source cells despite the manuscript's general Sinkhorn statement.
5. **Cylinder sections:** code and CSV columns use zero-based channels6,4,5 at value4; the manuscript says5,4,6. The repository uses actual executed code order.
6. **Different comparison cohorts:** Table4 PRIME and Figure9 PRIME use different checkpoints. Table6 fixed-lag and variable/sequential rows were evaluated with different PRIME references/configurations.
7. **Timing:** the historical Table19 profiler has objectives/settings that differ from the accuracy trainers. Table21 heterogeneous timings are a separate benchmark. New hardware timings are new measurements.
8. **Table14 DRTD spread:** its reported standard deviation is across three sections, whereas the caption describes30 rollout initializations. Preserve that source aggregation when auditing the table.
9. **Data/figures:** exact archived small arrays are bundled. Original observation-noise RNG states and some figure/checkpoint identities are absent; new simulation or explicit replacement weights do not recreate those missing artifacts exactly.

These findings are supported by checkpoint metadata, source code, archived CSVs, and source-parity probes in the experiment-specific guides. They are not claims that the scientific conclusions are false; they delimit what this recovered code can establish.

## Validation scope

Completed checks include main-table reconstruction against the earlier independent audit, two-update training checks for the12 main system/objective combinations, direct notebook-function/parameter parity checks, selected full-horizon CUDA checkpoint evaluations, baseline smoke runs, selected plot rendering, profiler quick runs, and asset integrity tests. Per-experiment manifests record the exact checks performed. The full five-seed retraining of every benchmark, the full parameter sweep, and the200000-step raw cylinder simulation were not run during curation.
