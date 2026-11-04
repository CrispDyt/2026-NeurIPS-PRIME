# Ablations and recurrence diagnostics

This subset recovers Tables 5–7 and 14–18 from the original Lorenz63 scripts,
notebook code cells, selected checkpoints and numeric result files. It reuses
`data/lorenz63/{train_data,test_data}.npy`; notebook outputs, old logs, duplicate
FINAL/BEST_L2 checkpoints and image archives are excluded. About 20.5 MB of
selected ablation checkpoints are included.

Run all commands below from the repository root. `run.py` supplies portable
data/checkpoint/output paths while preserving each original experiment's
numerical code. Use `python experiments/ablations/run.py TASK --help` to inspect
a task. New results go under `results/ablations`; archived results remain under
`reference/ablations`. The notebook task adapters refuse a nonempty output
folder. The original CLI scripts retain their own original output behavior;
choose a fresh `--run_root`, `--out_dir` or `--output_dir` when rerunning them.

## Table 5: recurrence schedules and recurrence sections

Evaluate the selected historical models:

```bash
python experiments/ablations/run.py schedule_eval
python experiments/ablations/run.py section_eval
```

The schedule experiment comes from `Ab_Lorenz_1.py` and the evaluation code in
`Ab_Lorenz_compare.ipynb`, cell 2 (zero based). Its four modes are `dim_only`,
`fixed`, `shuffled`, and `correct`. The section experiment uses
`Ab_Lorenz_2.py`, `Explicit40.py`, and notebook cell 3. Only the four paper rows
are selected: DIM-only, implicit data-driven events, matched random events, and
the explicit Z-band. The additional exact-Z40-crossing experiment is not a
paper row, and its old checkpoint was absent; it is omitted from the portable
evaluator rather than silently replaced by a different model.

Train new runs using the original CLIs:

```bash
python experiments/ablations/run.py schedule_train --all
python experiments/ablations/run.py section_train --all
python experiments/ablations/run.py explicit_band_train --mode exact_z40_band \
  --init_checkpoint PATH_TO_THE_NEW_SHARED_DIM_SELECTED_CHECKPOINT
```

Use the shared DIM checkpoint produced by `section_train` for the explicit
Z-band refinement. The original `Explicit40.py` option `--auto_dim_init` searches
its configured run root; specifying `--init_checkpoint` makes the intended
initialization explicit. The preserved defaults and source checkpoint metadata
must be checked together before interpreting a fresh run as the old experiment.
For evaluation of fresh runs, notebook tasks accept `--checkpoint-dir` pointing
to the parent directory containing `lorenz_ablation_K` or
`lorenz_ablation_section`.

Both panels use 30 starts for short-horizon NMSE. Panel (a) computes structural
and return metrics on one full test rollout; panel (b) computes them from
20 starts of 800 steps. These panel-specific protocols are intentionally not
replaced by the main-table evaluator's 32-start protocol.

## Table 6: fixed and empirical recurrence lags

```bash
python experiments/ablations/run.py fixed_lag_eval
python experiments/ablations/run.py variable_lag_eval
python experiments/ablations/run.py fixed_lag_train --seed 123456 --fixed-lag 3
python experiments/ablations/run.py sequential_lag_train --seed 123
```

Repeat fixed-lag training with `--fixed-lag 6`, `9`, and `12`. These are extracted
from `Lorenz.ipynb` cells 12–15, including the metric helper functions supplied
by cell 11. Fixed and sequential training use 5,000 total updates, a 3,000-step
DIM warm-up and a 1:3 stage-2/stage-1 refinement schedule, as recorded in those
source cells. New outputs are separate from bundled BEST checkpoints.

**The original Table 6 combines different cohorts.** The fixed-lag block uses
seed 123456 and CUDA evaluation; its PRIME reference is W2 `1.84190893` and
Log-PSD `0.41358417`. The sequential empirical-K checkpoint uses seed 123; the
independent empirical-variable-K checkpoint uses seed 123456. Their later
comparison cell explicitly uses CPU and a different no-warm-up PRIME checkpoint
whose W2 is `1.64578640`, not `1.84190893`. Both original reference CSVs are
preserved. The evaluator does not merge their PRIME rows into a falsely shared
reference. A new controlled common-seed/common-reference comparison would be a
new experiment.

The exact saved independent empirical-variable-K checkpoint and its evaluator
were recovered. Its producing training cell was not present in the available
notebooks/scripts, so no supposedly identical retraining entry point is
invented. The July rebuttal uniform-lag harness is a different experiment and
is not substituted for this row.

## Table 7: retaining DIM after warm-up

```bash
python experiments/ablations/run.py prime_w2_eval
python experiments/ablations/run.py prime_only_eval
python experiments/ablations/run.py dim_mse_w2_eval
python experiments/ablations/run.py prime_only_train
```

These tasks come from `pP3L_Rebuttal.ipynb` cells 0, 1, 9 and 10. Each evaluator
uses one 999-step autonomous rollout from `test_data[0]`, producing 1,000 states,
and its original CPU metric implementation. These W2 values therefore differ
from the 32-start pooled SD values in main Table 3.

The PRIME-only ablation performs 3,000 DIM-only warm-up updates followed by
2,000 PRIME-only updates, selecting `BEST_PRIME` by the PRIME-only loss. The
source comment mentions 4,000 second-stage steps, but the executable constant
and selected checkpoint metadata both say 2,000; the port uses 2,000.
`--steps` overrides only the PRIME-only stage; `--warm-steps` controls warm-up.
`--seed` restricts the five-seed tasks to a single actual seed. Main-table PRIME
and DIM+MSE checkpoints are reused directly without duplicating them.

## Table 14: graph and marginal loss terms

```bash
python experiments/ablations/run.py loss_terms_eval
python experiments/ablations/run.py loss_terms_train --all
```

Training is the original `Ab_Lorenz_3.py` with its two small local helper modules.
Evaluation is from `plot_recurrence_section.ipynb`, cell 6. All six archived modes
are included, allowing the four paper rows plus source-only/target-only
marginal diagnostics to be inspected. The source uses a common DIM warm start.

TCM, SD and Log-PSD aggregate 30 starts of 500 steps. **DRTD mean and standard
deviation in this table aggregate the three section values of a single long
rollout**, not the 30 initial conditions. This is established by the preserved
columns `DRTD_mean_over_sections`, `DRTD_std_over_sections`, and
`DRTD_n_valid_sections`. The paper caption should distinguish these two units
of aggregation.

## Table 15: graph-measure specificity

```bash
python experiments/ablations/run.py graph_specificity_eval
python experiments/ablations/run.py graph_specificity_train
```

These are the two code cells of `graph_abalation.ipynb`. The three trained
controls are DIM only, DIM plus recurrence marginals, and DIM plus paired MSE
and recurrence marginals. CPU evaluation uses one full test rollout for each
of five seeds and compares them with the archived PRIME checkpoints.

Preserve the original distinction: this control trainer uses 4,000 updates
and `warm_steps=0`; it is not the shared-warm-start trainer used for Table 14.
Its shared initial network uses `seed + 1000000`, and each method follows the
original method-specific RNG logic. These source conventions are kept in the
extracted code. The checkpoint selection is BEST_TOTAL. A new run with
`--warm-steps` changed is not the original Table 15 training.

## Table 16: speed-threshold sensitivity

```bash
python experiments/ablations/run.py speed_eval
python experiments/ablations/run.py speed_train --device cuda
```

The four thresholds are 0.10, 0.20, 0.30 and 0.40, and the actual archived
training seeds are **123, 456, 789, 2024, 3407**. The wrapper overrides the
later trainer's different default seed list with these recorded seeds.
`implicit_poincare_multiseed_train_eval_v2.py` contains training and its original
metric summaries; `parameter_abalation.py` is the later full-NMSE evaluator that
produced Table 16. The latter evaluates 30 starts, averages within each seed,
then reports population standard deviations across the five seed means.
Its exact archived rollout/seed/aggregate CSVs are in
`reference/ablations/speed_threshold/reevaluation`.

The older per-q training summaries also remain available for provenance; they
use a different aggregation implementation and do not contain all the NMSE
windows in the paper. Use the `reevaluation` files for Table 16.

## Table 17: recurrence-clock jitter

```bash
python experiments/ablations/run.py clock_eval
python experiments/ablations/run.py clock_train --all
```

These use the original `recurrence_clock_ablation.py` and
`evaluate_clock_robustness_results.py`, seed 1000 and the selected DIM-only,
exact-K, jitter-10% and jitter-20% models. The nominal perturbation settings
produce measured mean relative horizon deviations of 7.2647% and 15.0723%,
which are the labels used in the paper. Evaluation uses 30 starts and a
500-step horizon. The wrapper resolves source summary checkpoint basenames
inside the bundled checkpoint directory.

## Table 18: recurrence-section diagnostics

```bash
python experiments/ablations/run.py section_diagnostics
```

This runs `recurrence_section_diagnostics.py` with extractor seeds
33, 123, 1000, 13532, 123456. It varies speed quantile, delay dimension, landmark
count and entry threshold. It does not train a network. The graph split-half
quantity in this script is a sliced-Wasserstein diagnostic, preserved as
`split_graph_sw2`; it should not be confused with the main-table GeomLoss SD.
The archived per-seed and aggregate files are included.

## Verification and limitations

All task entry points compile. Bounded checks executed the archived evaluators
for the schedule and section panels, fixed and variable lags, seed 33 of each
Table 7 group and graph-specificity group, all loss-term models, all four
clock models, and all 20 speed-threshold models with one test start. The four
notebook training tasks passed one/two-update CPU smoke runs, including both
warm-up and second-stage code paths where applicable. The diagnostics task
passed a one-seed minimal run. No full training sweep was launched during
curation. `reference/ablations/port_validation.json` records the checks and
numeric comparisons.

Archived CPU tasks default to CPU; archived GPU tasks default to CUDA.
Notebook adapters fail clearly if the chosen device is unavailable. Original
standalone scripts retain their documented `--cpu`/`--device` behavior. Long
chaotic trajectories can differ across devices and numerical runtimes; a
changed backend is not guaranteed to match the archived numeric table.
The scripts require NumPy, pandas, PyTorch, scikit-learn, SciPy, GeomLoss and
Matplotlib. Notebook display calls have been replaced by terminal printing,
so IPython/Jupyter is not required. Optional `--plots` enables the source
notebook plots; exact published figure layout is covered separately in
`docs/figures.md`.

`reference/ablations/provenance.json` records every copied source/checkpoint
hash, every extracted notebook cell and the adapted destination hash. Absolute
machine-specific checkpoint paths in reference text were normalized to
source-relative paths; metric values were not changed. All source NIPS and
Entropy files remain unchanged. The identified cohort, caption and
source-history differences require scientific judgment; preserving them here
is preferable to relabeling historical results as a uniform experiment.
