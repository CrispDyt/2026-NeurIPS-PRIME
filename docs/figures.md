# Plotting and Rössler source recovery

The scripts in `experiments/figures/` are output-stripped extractions of selected original notebook cells. They load repository data and write to an explicit output directory; no notebook session or old project path is required. Original plotting and rollout logic is retained. A stray non-code `Machine Learning` line in the Lorenz30-rollout cell was removed, notebook-global assertions were adapted to function-local variables, and display calls now save artifacts. Checkpoint uncertainty is exposed rather than silently selecting a different model. Source notebook paths/cell indices/hashes are in `reference/figures/provenance.json`.

## Lorenz63 phase/return maps (Figure 1)

The original plotting cell names `Implicit_BEST_11_0.5.p`, `DIM_BEST.pt`, `DIM_MSE_BEST_0.111.pt` and `NODE_BEST.p` in its working directory. Several paths are no longer present; in particular, the specially named DIM+MSE checkpoint has not been located. The plotting code is recovered, but the exact four-checkpoint Figure 1 cohort is not verified. Select all four inputs explicitly, for example to visualize main-table seed-folder13532:

```bash
python experiments/figures/lorenz_phase_return.py --device cuda \
  --prime checkpoints/lorenz63/seed_13532/Implicit_BEST.p \
  --dim checkpoints/lorenz63/seed_13532/DIM_BEST.pt \
  --dim-mse checkpoints/lorenz63/seed_13532/DIM_MSE_BEST.pt \
  --mse checkpoints/lorenz63/seed_13532/NODE_BEST.p
```

This example is an explicit new selection, not a claim to recover the published figure's exact curves.

## Lorenz63 collapse (Figures 2, 5, 6)

```bash
python experiments/figures/lorenz_collapse.py --device cuda \
  --checkpoint checkpoints/lorenz63/seed_13532/NODE_BEST.p
python experiments/figures/lorenz_30_rollouts.py --device cuda \
  --checkpoint checkpoints/lorenz63/seed_13532/NODE_BEST.p
python experiments/figures/lorenz_multistep_rollouts.py --device cuda
```

The first two commands require an explicit checkpoint because their source notebooks refer to an absent working-directory `NODE_BEST.p`/`NODE.p`. The chosen example above is not asserted to reproduce the paper's 26/4 failure-mode count. Figure 2's code draws the originally selected trajectories and ensemble covariance determinant; it does not recalculate a robust limit-cycle classifier. Figure 5's 30-start plot uses a different RNG API than Figure 2; the original distinction is retained.

Figure 6 uses the recovered `Euler-NODE/NODE_BEST.p`, stored as `checkpoints/figures/lorenz_mse20.p`. Its metadata specifies 20 **microsteps** per two-observation training pair (`dt_base=.01`, `dt_obs=.1`, `tau_pair=2`); this is not 20 observation-time targets. The original drawing uses 30 starts and 1000 observation steps. That terminology differs from a literal reading of "20-step objective" in the manuscript and should be clarified before release.

## Cylinder (Figure 3)

See [cylinder figure instructions](figures_cylinder.md). Probe forecasts use bundled sensor data. The raw field snapshot additionally needs the original unbundled `flow_full70.p` or a new simulation:

```bash
# Requires a separately installed JAX runtime.
python experiments/data_generation/simulate_cylinder.py --output-dir results/cylinder_raw
python experiments/cylinder/prepare_data.py --raw results/cylinder_raw/flow_full70.p \
  --output-dir results/cylinder_new_sensors --seed 0
```

The default raw simulator saves approximately1.2GB. Its full simulation was not rerun in the core environment. The original spatial-noise seed was not saved; newly extracted sensors are a different noise realization. Training/evaluation of the paper's models should use the exact bundled sensor array.

## KS collapse (Figures 7, 8)

```bash
python experiments/figures/ks_one_step_rollouts.py --device cuda
python experiments/figures/ks_multistep_rollouts.py --device cuda
```

These preserve `KS/plot.ipynb` cells0 and2. The first uses archived seed13531 MSE; the second uses `KS/multi/node_multi_BEST.pt`, stored separately under `checkpoints/figures`. Both draw 30 starts and 2000-step rollouts. The multi-step checkpoint records `dt_base=1`, `stride=10`, `dt_obs=10`, `tau_pair=2` (20 microsteps). This differs from the main-table KS protocol's dt_obs=1; the plotting code deliberately retains its source convention. Exact paper failure counts have not been revalidated.

## Rössler delay-coordinate comparison (Figure 4)

The exact pair of original Figure 4 checkpoints and the original confidence-interval calculation were not found in NIPS/Entropy. The recovered candidate code is `Fullstate/Rossler_fullstate_test.ipynb` cell7: a three-dimensional state predictor with a scalar delay loss, energy-distance measure losses, and an additional delay-window pointwise term. It is not identical to the paper's generally stated Sinkhorn objective. It has been preserved as a clearly identified candidate rather than presented as an exact Figure 4 reproduction.

To run a fresh comparison using that recovered implementation:

```bash
python experiments/rossler/train_dim.py --channel 0 --seed 12345 --device cuda --output-dir results/rossler/y0
python experiments/rossler/train_dim.py --channel 2 --seed 12345 --device cuda --output-dir results/rossler/y2
python experiments/rossler/compare_delay.py --device cuda \
  --checkpoint-y0 results/rossler/y0/DIM_channel0_seed12345.pt \
  --checkpoint-y2 results/rossler/y2/DIM_channel2_seed12345.pt
```

The new comparison utility reports cumulative mean squared error over all three state coordinates and a normal-approximation95% interval across30 evenly spaced starts; these starts come from one trajectory and are correlated. It records this explicit protocol in JSON and does not claim that this was the original paper calculation. The exact archived 500×3/1500×3 train/test arrays are included. Full fresh training is not needed to use the three main benchmarks.

## Heterogeneous baselines (Figure 9)

Use `python experiments/baselines/plot.py --device cuda`; see [baselines](baselines.md) for the different PRIME checkpoint used by Figure9 versus Table4.

## Checks performed

The extracted Lorenz covariance, Lorenz multi-step, KS multi-step, and Lorenz phase/return plotting scripts completed CPU execution at their original horizons; rendered examples were inspected. A two-update CPU Rössler DIM training run completed and saved/reloaded its checkpoint. These are execution checks, not verification of the exact published visual cohorts or full fresh optimization.
