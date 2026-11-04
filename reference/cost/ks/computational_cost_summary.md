# Computational cost summary

Wall-clock time includes objective optimization only; setup time is reported separately. CUDA memory is peak allocated memory.

| Benchmark | Method | Steps | Train sec | Avg iter ms | Peak CUDA MB | Params |
|---|---|---:|---:|---:|---:|---:|
| ks | mse | 10000 | 12.11 ± 0.00 | 1.21 ± 0.00 | 18.8 ± 0.0 | 21305 |
| ks | dim | 10000 | 172.34 ± 0.00 | 17.23 ± 0.00 | 46.3 ± 0.0 | 21305 |
| ks | dim_mse | 10000 | 167.40 ± 0.00 | 16.74 ± 0.00 | 46.3 ± 0.0 | 21305 |
| ks | prime | 12000 | 2179.05 ± 0.00 | 181.59 ± 0.00 | 46.4 ± 0.0 | 21305 |
