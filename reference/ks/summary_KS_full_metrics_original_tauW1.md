# KS evaluation: prediction metrics + original tau_dist_W1 logic

Prediction metrics use 5 seed folders × 32 starts by default. Poincare `tau_dist_W1` uses the original crossing-time logic: linearly interpolated hit times, `tau=np.diff(hit_times)`, and `wasserstein_1d_empirical(ref_tau, pred_tau)`.

## Prediction / structure / temporal metrics

| Model | n | NMSE_20 | NMSE_100 | NMSE_400 | Attractor W2 | PSD logL1 | TCM |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRIME | 160 | 1.3047 ± 0.51712 | 1.4596 ± 0.23497 | 1.4689 ± 0.12148 | 1.9948 ± 0.77831 | 1.6554 ± 1.2383 | 10.176 ± 18.776 |
| MSE | 160 | 1.9163 ± 0.8668 | 7.2065 ± 30.14 | 257.03 ± 988.26 | 11883 ± 39154 | 1.5893 ± 0.24529 | 88.028 ± 114.19 |
| DIM | 160 | 1.7247 ± 0.60882 | 2.2179 ± 0.93412 | 16.028 ± 46.36 | 1197.6 ± 2725.4 | 0.96174 ± 0.62579 | 91.67 ± 120.64 |
| DIM+MSE | 160 | 1.6769 ± 0.78425 | 2.52 ± 1.7655 | 11.7 ± 31.208 | 336.12 ± 945.88 | 1.4798 ± 0.84984 | 84.646 ± 103.3 |

## Poincare `tau_dist_W1` — original logic

### Sigma1: u0=0

| Model | records | GT hits | Pred hits | tau_dist_W1 | valid |
|---|---:|---:|---:|---:|---:|
| PRIME | 5 | 61 ± 0 | 73.2 ± 11.461 | 7.3039 ± 1.5774 | 5 |
| MSE | 5 | 61 ± 0 | 93.8 ± 46.533 | 11.401 ± 4.7391 | 5 |
| DIM | 5 | 61 ± 0 | 54.6 ± 28.415 | 10.392 ± 3.6571 | 5 |
| DIM+MSE | 5 | 61 ± 0 | 49 ± 12.41 | 10.704 ± 7.6402 | 5 |

### Sigma2: u2=0

| Model | records | GT hits | Pred hits | tau_dist_W1 | valid |
|---|---:|---:|---:|---:|---:|
| PRIME | 5 | 63 ± 0 | 47.6 ± 11.056 | 12.333 ± 7.6951 | 5 |
| MSE | 5 | 63 ± 0 | 70 ± 31.209 | 9.799 ± 3.573 | 5 |
| DIM | 5 | 63 ± 0 | 40 ± 22.19 | 11.332 ± 7.2354 | 4 |
| DIM+MSE | 5 | 63 ± 0 | 31.6 ± 17.351 | 16.572 ± 5.1827 | 4 |

### Sigma3: u1=0

| Model | records | GT hits | Pred hits | tau_dist_W1 | valid |
|---|---:|---:|---:|---:|---:|
| PRIME | 5 | 61 ± 0 | 58.4 ± 10.763 | 6.4944 ± 3.1157 | 5 |
| MSE | 5 | 61 ± 0 | 84.2 ± 40.814 | 11.157 ± 3.0867 | 5 |
| DIM | 5 | 61 ± 0 | 75 ± 43.29 | 11.178 ± 4.3395 | 5 |
| DIM+MSE | 5 | 61 ± 0 | 53.6 ± 5.3141 | 7.5803 ± 2.1231 | 5 |

