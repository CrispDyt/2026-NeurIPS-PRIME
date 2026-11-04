# PRIME

Code and archived experiment inputs for **PRIME: Poincaré return induced measure for learning partially observed dynamical systems**.

This repository collects the paper-related implementations. It includes training and evaluation for Lorenz63, Kuramoto–Sivashinsky (KS), and flow past a cylinder, together with selected baseline, ablation, and code for plotting. 

## Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

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

TCM and prediction metrics pool **160 rollouts** per model (32 starts × five seed folders). DRTD uses one shared-start result per seed/section and excludes missing values. 

## Re-evaluate the supplied models

From the repository root:

```bash
python experiments/lorenz63/evaluate.py --device cuda --output-dir results/evaluations/lorenz63
python experiments/ks/evaluate.py --device cuda --output-dir results/evaluations/ks
python experiments/cylinder/evaluate.py --device cuda --output-dir results/evaluations/cylinder
python scripts/build_tables.py --input-root results/evaluations --output-dir results/recomputed_tables
```

`--model`, `--seeds`, `--data-dir`, `--checkpoint-dir`, and `--output-dir` select a subset or new inputs. 

## Train models from scratch

```bash
python experiments/lorenz63/train.py --model PRIME --seed 33 --device cuda
python experiments/ks/train.py --model PRIME --seed 4242 --device cuda
python experiments/cylinder/train.py --model PRIME --seed 100 --device cuda
```

Use `--model MSE`, `--model DIM`, or `--model 'DIM+MSE'` for the other objectives; repeat for the required actual training seeds. Training writes new run directories. `--seed` always means the actual random seed. **PRIME in the main tables is the full training scheme with DIM retained after warm-up.** PRIME-only is a separate ablation.

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

## Layout

```text
experiments/   training, evaluation, simulation, ablation and plotting entry points
data/          exact small observation arrays used by the retained experiments
checkpoints/   selected historical weights, including fitted scalers
reference/     original raw/summary results, source hashes and validation evidence
scripts/       table reconstruction
docs/          protocols, coverage and unresolved provenance differences
tests/         integrity and statistical regression checks
```
