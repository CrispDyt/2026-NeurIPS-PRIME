# Computational cost summary

Wall-clock time includes objective optimization only; setup time is reported separately. CUDA memory is peak allocated memory.

| Benchmark | Method | Steps | Train sec | Avg iter ms | Peak CUDA MB | Params |
|---|---|---:|---:|---:|---:|---:|
| lorenz | mse | 10000 | 12.85 ± 0.00 | 1.28 ± 0.00 | 18.8 ± 0.0 | 20702 |
| lorenz | dim | 10000 | 107.99 ± 0.00 | 10.80 ± 0.00 | 30.3 ± 0.0 | 20702 |
| lorenz | dim_mse | 10000 | 105.46 ± 0.00 | 10.55 ± 0.00 | 30.3 ± 0.0 | 20702 |
| lorenz | prime | 4000 | 701.20 ± 0.00 | 175.30 ± 0.00 | 32.7 ± 0.0 | 20702 |
