# Cylinder forecast and flow panels (Figure 3)

The portable plotting command follows the latest complete forecast workflow in `Entropy/Entropy-1/Cylinder Case/Comparison of Cylinder CFD.ipynb`, cells17 and18 (one-based): all four checkpoints from `Seed100`, common initial state at test index0,1000 observation steps at0.1, sensor channel0, display interval80–100 and inset86–90.

```bash
python experiments/figures/cylinder_forecast.py --device cuda
```

The bundled sensor data and checkpoints are sufficient for this command. It writes `cylinder_forecast.pdf`, `cylinder_forecast.png`, the underlying arrays and a manifest under `runs/figures/cylinder`. Options include `--seed`, `--channel`, `--start-index`, `--horizon`, `--plot-window` and `--zoom-window`. The time axis is local to the requested rollout initialization.

To include the left flow snapshot/probe panel, supply the original large raw data file or a newly generated compatible raw file:

```bash
python experiments/figures/cylinder_forecast.py --device cuda --raw-file /path/to/flow_full70.p
```

This additionally saves `cylinder_flow_and_forecast.pdf` and `.png`. The raw snapshot/probe recipe comes from `Flow_data_preparison.py`: frame4000 (index1000 after discarding3000 transient frames), lattice speed multiplied by100, physical grid spacing0.1, and nine probe indices:

```text
(62,12), (52,44), (77,15), (43,35), (74,35),
(109,40), (125,15), (80,35), (100,20)
```

The cylinder location/radius are read from raw metadata. Probe disks have radius0.15, corresponding to three times the original spatial-noise standard deviation0.05. `--raw-frame` changes the selected snapshot. The raw file is about1.2GB and is deliberately not duplicated in the repository. The command loads it only when requested; without it, no flow field is fabricated.

## Correspondence and limits

The default selections above are explicit in notebook cells17/18. An earlier plotting cell9 instead used older unseeded checkpoint filenames and start index846; notebook cell10 attempted to label an undefined `sensor_xy`. The port uses the later self-contained Seed100 selection and the probe positions from the data-preparation source.

The exact saved model selection and panel assembly used for the published Figure3 cannot be proven from the surviving notebook alone. This command reproduces the documented forecast/snapshot procedure and records its selected inputs; it does not claim a pixel-identical copy of the published figure. The evaluator's checked model loader and Euler/interpolation routine are reused so each checkpoint retains its actual integration step. CPU can be explicitly selected for checking, but chaotic rollout values may differ from the archived CUDA backend.

Validation: syntax and CLI help passed. The full1000-step CPU forecast from bundled Seed100 models completed and produced four finite1001×9 trajectories and a readable PDF/PNG. The PNG was visually inspected. The optional large-raw-file snapshot path was not executed during curation. Source hashes and verification status are stored in `reference/figures/cylinder_plot_provenance.json`.

## Optional Lorenz data simulator

`experiments/data_generation/lorenz63.py` also exposes the Euler generator in code cell1 of `NIPS/Lorenz/1-step MSE/Lorenz.ipynb`:

```bash
python experiments/data_generation/lorenz63.py --output-dir runs/generated/lorenz63 --noise-seed 0
```

That source cell integrates100000 states at0.01 from `(0.1,0,0)`, adds Gaussian noise of standard deviation1 and downsamples by10. Its stored long-data split is5000 training and5000 testing observations in all three coordinates. This is distinct from the bundled paper inputs of500×2/1000×2, which the notebook loads from separate files. `--channels 0 2`, `--train-count` and `--test-count` can explicitly select another split, but neither its original slicing history nor its unrecorded noise RNG state is inferred. Generated files are labeled as new realizations, include the selected settings and hashes, and cannot overwrite existing train/test arrays. Use bundled data for archived table reproduction.
