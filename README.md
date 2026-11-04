# PRIME

Code and archived experiment inputs for **PRIME: Poincaré return induced measure for learning partially observed dynamical systems**.

This repository collects the paper-related implementations from the original NIPS and Entropy research folders. It includes training and evaluation for Lorenz63, Kuramoto–Sivashinsky (KS), and flow past a cylinder, together with selected baseline, ablation, timing, and plotting code. The source notebooks' output images, redundant checkpoints, logs, and large raw flow arrays are omitted.

**Reproduction status:** the main tables can be rebuilt from the bundled raw results. The 60 main-experiment checkpoints and their exact input arrays are included. Historical seed/configuration inconsistencies and incomplete figure provenance are documented in [the coverage audit](docs/paper_coverage.md); the repository does not claim that every paper figure or historical training run has been reproduced exactly. 

## Environment

The migration and validation checks used the existing Conda environment **Poincare**, with Python **3.11.15**, PyTorch runtime **2.1.2+cu121**, and CUDA **12.1**. If this environment is already available, use it directly:

```bash
conda activate Poincare
python -m pip check
```

All eight versions in `requirements.txt` were checked against this environment, and its package dependency check passes. The file pins the core direct dependencies; it is not a complete Conda environment lock and does not establish the environment used by every historical checkpoint. Some saved scikit-learn scalers record version1.6.1, while the current environment has1.1.3; see the system-specific validation notes.

For a separate Python3.11 environment with a CUDA12.1 PyTorch build, the installation recipe is below. A fresh installation of this recipe has not been independently validated:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The archived runs used an NVIDIA RTX 4060 Laptop GPU. CPU execution is useful for installation checks, but long chaotic trajectories can differ substantially between CPU, CUDA, and library versions. Optional raw KS/cylinder simulation uses JAX, which is not required for training or evaluating the bundled observations and is not included in the tested core environment.

## Rebuild the main tables without training

```bash
python scripts/build_tables.py
```

This reads `reference/{lorenz63,ks,cylinder}/` and writes:

- `results/tables/overall.csv`: all 108 system/model/metric summaries for Tables 2 and 3.
- `results/tables/per_seed.csv`: 540 seed-level summaries.
- `results/tables/appendix.csv`: five seed values, overall mean/std, and cross-seed median/Q25/Q75.
- `results/tables/tables.tex`: Overleaf-compatible main and appendix table code.
- `results/tables/provenance.json`: input hashes, aggregation definitions, and section ordering.

TCM and prediction metrics pool **160 rollouts** per model (32 starts × five seed folders). DRTD uses one shared-start result per seed/section and excludes missing values. Standard deviations use `ddof=0`; appendix quartiles interpolate linearly across the five seed means or valid DRTD scalars. The single-valid-seed DRTD standard deviation is zero and its valid count is displayed. `SD` retains the original `Attractor_W2` numeric value; no extra square root is applied. The repaired, independently trained KS–MSE seed **123123** is included.

## Re-evaluate the supplied models

From the repository root:

```bash
python experiments/lorenz63/evaluate.py --device cuda --output-dir results/evaluations/lorenz63
python experiments/ks/evaluate.py --device cuda --output-dir results/evaluations/ks
python experiments/cylinder/evaluate.py --device cuda --output-dir results/evaluations/cylinder
python scripts/build_tables.py --input-root results/evaluations --output-dir results/recomputed_tables
```

Choose a fresh output directory for each evaluation; existing results are not overwritten. Each evaluator defaults to all four methods and all five historical seed folders, and writes raw CSVs plus runtime/provenance records. `--model`, `--seeds`, `--data-dir`, `--checkpoint-dir`, and `--output-dir` select a subset or new inputs. See the individual experiment guides for seed argument syntax and horizons. Shortened smoke evaluations intentionally do not satisfy the table builder's full 640-row check.

## Train models from scratch

```bash
python experiments/lorenz63/train.py --model PRIME --seed 33 --device cuda
python experiments/ks/train.py --model PRIME --seed 4242 --device cuda
python experiments/cylinder/train.py --model PRIME --seed 100 --device cuda
```

Use `--model MSE`, `--model DIM`, or `--model 'DIM+MSE'` for the other objectives; repeat for the required actual training seeds. Training writes new run directories. `--seed` always means the actual random seed. **PRIME in the main tables is the full training scheme with DIM retained after warm-up.** PRIME-only is a separate ablation.

The historical checkpoints have heterogeneous settings: the seed label and saved seed are not always the same, and some archived training profiles differ from the latest notebooks. Follow the per-system guides before interpreting a fresh run as an exact repeat:

| Experiment | Guide |
|---|---|
| Lorenz63 | [Training, metrics, historical profiles](docs/lorenz63.md) |
| KS | [Training, repaired seed, historical profiles](docs/ks.md) |
| Cylinder | [Training, sensor data, section definitions](docs/cylinder.md) |
| Heterogeneous baselines, Table 4/Figure 9 | [Baselines](docs/baselines.md) |
| Tables 5–7 and 14–18 | [Ablations](docs/ablations.md) |
| Tables 19–21 | [Computational cost](docs/cost.md) |
| Figures and the Rössler experiment | [Plotting and recovery limits](docs/figures.md) |

Useful additional entry points:

```bash
python experiments/baselines/evaluate.py --device cuda
python experiments/baselines/train.py --all --device cuda
python experiments/ablations/run.py --help
python experiments/cost/profile_training.py --help
python experiments/cost/profile_inference.py --help
python experiments/cost/profile_heterogeneous.py --help
```

## Verify the curated assets

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests validate input/checkpoint hashes, the corrected KS checkpoint, missing DRTD handling, aggregation grain, and audited numeric anchors. Small training runs, source-function comparisons, and selected full-horizon CUDA evaluations are recorded in `reference/*/port_validation.json`. A complete fresh retraining of every experiment was not performed during repository assembly.

## Layout and provenance

```text
experiments/   training, evaluation, simulation, ablation and plotting entry points
data/          exact small observation arrays used by the retained experiments
checkpoints/   selected historical weights, including fitted scalers
reference/     original raw/summary results, source hashes and validation evidence
scripts/       table reconstruction
docs/          protocols, coverage and unresolved provenance differences
tests/         integrity and statistical regression checks
```

`results/` and `runs/` are generated and ignored by Git. New evaluation outputs and archived reference results have separate locations. The repository does not require the old NIPS/Entropy directories at runtime. Historical pickle checkpoints contain Python/scikit-learn objects and should only be loaded from trusted sources.

Third-party simulation notices are retained in [CREDITS.md](CREDITS.md). A project-wide release license has not been inferred from the upstream simulator's license.
