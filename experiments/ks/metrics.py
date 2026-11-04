"""KS metric kernels adapted from Entropy-1/KS/evaluateKS.py.

Historical numerical formulas are retained, including sorted-and-truncated
return-interval distance. SciPy Welch and GeomLoss are required, so metric
failures are reported rather than replaced by another algorithm.
"""
import csv
import io
import os
import pickle
import sys
import numpy as np
import torch
from torch import nn
from scipy.signal import welch
from geomloss import SamplesLoss
from sklearn.preprocessing import MaxAbsScaler

# NumPy 2 pickle module names in some archived checkpoints.
if np.lib.NumpyVersion(np.__version__) < "2.0.0":
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
HAS_WELCH = True
MODEL_ORDER = ["PRIME", "MSE", "DIM", "DIM+MSE"]
MODEL_FILE_CANDIDATES = {
    "PRIME": ["KS_Implicit_seed_BEST.p"], "MSE": ["node_BEST.pt"],
    "DIM": ["DIM_BEST.pt"], "DIM+MSE": ["DIM_MSE_BEST.pt"],
}

def as_float_or_nan(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan

def finite_array(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return vals[np.isfinite(vals)]

def finite_mean_std(vals):
    vals = finite_array(vals)
    if len(vals) == 0:
        return (np.nan, np.nan, 0)
    return (float(np.mean(vals)), float(np.std(vals, ddof=0)), int(len(vals)))

def fmt_mean_std(mu, sd, nd=5):
    if not np.isfinite(mu):
        return 'nan'
    if not np.isfinite(sd):
        return f'{mu:.{nd}g} ± nan'
    return f'{mu:.{nd}g} ± {sd:.{nd}g}'

def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

class CPU_Unpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)

def load_p_cpu(path):
    with open(path, 'rb') as f:
        return CPU_Unpickler(f).load()

def smart_load(path):
    last_err = None
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        last_err = e
    try:
        return load_p_cpu(path)
    except Exception as e:
        raise RuntimeError(f'Failed to load checkpoint: {path}\ntorch.load error: {last_err}\npickle error: {e}')

def strip_module_prefix(sd):
    if isinstance(sd, dict) and len(sd) > 0 and all((k.startswith('module.') for k in sd.keys())):
        return {k.replace('module.', '', 1): v for (k, v) in sd.items()}
    return sd

def infer_mlp_dims_from_state_dict(sd):
    sd = strip_module_prefix(sd)
    wkeys = []
    for k in sd.keys():
        if k.endswith('.weight'):
            p = k.split('.')[0]
            if p.isdigit():
                wkeys.append((int(p), k))
    wkeys = sorted(wkeys, key=lambda x: x[0])
    if not wkeys:
        raise ValueError("Could not infer MLP dims from state_dict: no numeric '*.weight' keys.")
    W0 = sd[wkeys[0][1]]
    Wlast = sd[wkeys[-1][1]]
    return (int(W0.shape[1]), int(W0.shape[0]), int(Wlast.shape[0]))

def build_net(d, width=100):
    return nn.Sequential(nn.Linear(d, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, d))

def load_model_ckpt(path, train_arr):
    obj = smart_load(path)
    sd = None
    scaler = None
    if isinstance(obj, dict):
        if 'model' in obj and isinstance(obj['model'], dict):
            sd = obj['model'].get('state_dict', None)
            scaler = obj['model'].get('transformer', obj['model'].get('scaler', None))
        if sd is None and 'state_dict' in obj:
            sd = obj['state_dict']
        if sd is None and 'model_state_dict' in obj:
            sd = obj['model_state_dict']
        if sd is None and 'net_state_dict' in obj:
            sd = obj['net_state_dict']
        if scaler is None and 'transformer' in obj:
            scaler = obj['transformer']
        if scaler is None and 'scaler' in obj:
            scaler = obj['scaler']
    else:
        sd = obj
    if sd is None:
        raise RuntimeError(f'Could not find usable state_dict in checkpoint: {path}')
    sd = strip_module_prefix(sd)
    if scaler is None:
        raise ValueError(f"Checkpoint has no training scaler: {path}")
    return (obj, sd, scaler)

@torch.no_grad()
def rollout_from_x0(net, scaler, x0_raw, K, dt_obs, device):
    x0_s = scaler.transform(np.asarray(x0_raw).reshape(1, -1)).astype(np.float32)
    x = torch.tensor(x0_s, dtype=torch.float32, device=device)
    pred_s = np.zeros((K + 1, x.shape[1]), dtype=np.float32)
    for k in range(K + 1):
        pred_s[k] = x.detach().cpu().numpy()[0]
        if k == K:
            break
        x = x + float(dt_obs) * net(x)
    return scaler.inverse_transform(pred_s).astype(np.float64)

def extract_seed_id(seed_folder):
    base = os.path.basename(seed_folder.rstrip('/'))
    parts = base.split('_')
    for p in reversed(parts):
        if p.isdigit():
            return p
    return base

def resolve_model_path(seed_folder, model_name):
    seed_id = extract_seed_id(seed_folder)
    for cand in MODEL_FILE_CANDIDATES[model_name]:
        p = os.path.join(seed_folder, cand.format(seed=seed_id))
        if os.path.exists(p):
            return p
    return None

def nmse_window_steps(gt, pred, k, eps=1e-12):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    L = min(len(gt), len(pred))
    if L <= 0:
        return np.nan
    k = int(max(0, min(k, L - 1)))
    err = gt[:k + 1] - pred[:k + 1]
    return float(np.mean(err ** 2) / (np.mean(gt[:k + 1] ** 2) + eps))

def sinkhorn_w2_cloud(A, B, blur=0.05, device='cpu', max_points=1500):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    A = A[np.isfinite(A).all(axis=1)]
    B = B[np.isfinite(B).all(axis=1)]
    if len(A) == 0 or len(B) == 0:
        return np.nan
    if np.nanmax(np.abs(A)) > 1000000.0 or np.nanmax(np.abs(B)) > 1000000.0:
        return np.nan
    if len(A) > max_points:
        idx = np.linspace(0, len(A) - 1, max_points).astype(int)
        A = A[idx]
    if len(B) > max_points:
        idx = np.linspace(0, len(B) - 1, max_points).astype(int)
        B = B[idx]
    XA = torch.tensor(A, dtype=torch.float32, device=device)
    XB = torch.tensor(B, dtype=torch.float32, device=device)
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=float(blur), backend="tensorized")
    return float(loss(XA, XB).detach().cpu())


def psd_logl1_distance(ref, pred, dt, nfft=2048, channels=None, eps=1e-12):
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    L = min(len(ref), len(pred))
    ref = ref[:L]
    pred = pred[:L]
    if channels is None:
        channels = list(range(ref.shape[1]))
    fs = 1.0 / float(dt)
    dists = []
    for ch in channels:
        x = ref[:, ch] - np.mean(ref[:, ch])
        y = pred[:, ch] - np.mean(pred[:, ch])
        n = min(int(nfft), L)
        x = x[:n]
        y = y[:n]
        if len(x) < 8 or len(y) < 8:
            continue
        if HAS_WELCH:
            (_, Pxx) = welch(x, fs=fs, nperseg=min(512, n), noverlap=min(256, n // 2))
            (_, Pyy) = welch(y, fs=fs, nperseg=min(512, n), noverlap=min(256, n // 2))
        else:
            Pxx = np.abs(np.fft.rfft(x)) ** 2 / max(1, len(x))
            Pyy = np.abs(np.fft.rfft(y)) ** 2 / max(1, len(y))
        m = min(len(Pxx), len(Pyy))
        if m > 0:
            dists.append(float(np.mean(np.abs(np.log10(Pxx[:m] + eps) - np.log10(Pyy[:m] + eps)))))
    dists = finite_array(dists)
    return float(np.mean(dists)) if len(dists) else np.nan

def autocorr_1d(x, max_lag, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return np.zeros((0,), dtype=np.float64)
    x = x - np.mean(x)
    n = len(x)
    c0 = float(np.dot(x, x) / n)
    if c0 < eps:
        return np.zeros((int(min(max_lag, n - 2)) + 1,), dtype=np.float64)
    max_lag = int(min(max_lag, n - 2))
    rho = np.empty(max_lag + 1, dtype=np.float64)
    rho[0] = 1.0
    for lag in range(1, max_lag + 1):
        rho[lag] = float(np.dot(x[lag:], x[:-lag]) / (n - lag) / c0)
    return rho

def integrated_autocorr_time_1d(x, dt, max_lag=200, stop_first_nonpositive=False):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return np.nan
    max_lag = min(int(max_lag), len(x) // 2)
    rho = autocorr_1d(x, max_lag=max_lag)
    if len(rho) == 0:
        return np.nan
    if np.allclose(rho, 0.0):
        return 0.0
    tail = rho[1:]
    if stop_first_nonpositive:
        bad = np.where(tail <= 0.0)[0]
        if len(bad):
            tail = tail[:bad[0]]
    tau = float(dt) * (1.0 + 2.0 * float(np.sum(tail)))
    return float(max(tau, 0.0)) if np.isfinite(tau) else np.nan

def tcm_distance(ref, pred, dt, max_lag=200, channels=None, stop_first_nonpositive=False):
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    L = min(len(ref), len(pred))
    ref = ref[:L]
    pred = pred[:L]
    if ref.ndim == 1:
        ref = ref[:, None]
    if pred.ndim == 1:
        pred = pred[:, None]
    if channels is None:
        channels = list(range(ref.shape[1]))
    (tr_list, tp_list) = ([], [])
    for ch in channels:
        tr = integrated_autocorr_time_1d(ref[:, ch], dt=dt, max_lag=max_lag, stop_first_nonpositive=stop_first_nonpositive)
        tp = integrated_autocorr_time_1d(pred[:, ch], dt=dt, max_lag=max_lag, stop_first_nonpositive=stop_first_nonpositive)
        if np.isfinite(tr) and np.isfinite(tp):
            tr_list.append(tr)
            tp_list.append(tp)
    if len(tr_list) == 0:
        return np.nan
    return float(abs(np.mean(tr_list) - np.mean(tp_list)))

def crossing_happened(a, b, direction):
    if direction > 0:
        return a < 0.0 and b >= 0.0
    elif direction < 0:
        return a > 0.0 and b <= 0.0
    else:
        return a == 0.0 or b == 0.0 or a * b < 0.0

def extract_poincare_hits(traj, dt, section_dim=0, section_value=0.0, direction=+1):
    X = np.asarray(traj, dtype=np.float64)
    X = X[np.isfinite(X).all(axis=1)]
    if len(X) < 2:
        return (np.array([], dtype=np.float64), np.zeros((0, X.shape[1] if X.ndim == 2 else 0), dtype=np.float64), np.array([], dtype=np.float64))
    (T, d) = X.shape
    hit_times = []
    hit_points = []
    for k in range(T - 1):
        a = X[k, section_dim] - section_value
        b = X[k + 1, section_dim] - section_value
        if not crossing_happened(a, b, direction):
            continue
        denom = b - a
        alpha = 0.0 if abs(denom) < 1e-14 else -a / denom
        if alpha < 0.0 or alpha > 1.0:
            continue
        t_hit = (k + alpha) * float(dt)
        x_hit = X[k] + alpha * (X[k + 1] - X[k])
        hit_times.append(t_hit)
        hit_points.append(x_hit)
    hit_times = np.asarray(hit_times, dtype=np.float64)
    hit_points = np.asarray(hit_points, dtype=np.float64) if hit_points else np.zeros((0, d), dtype=np.float64)
    return_times = np.diff(hit_times) if len(hit_times) >= 2 else np.zeros((0,), dtype=np.float64)
    return (hit_times, hit_points, return_times)

def wasserstein_1d_empirical(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    n = min(len(x), len(y))
    xs = np.sort(x)[:n]
    ys = np.sort(y)[:n]
    return float(np.mean(np.abs(xs - ys)))

def summarize_poincare_tau(ref_tau, pred_tau, pred_hit_times):
    row = {}
    row['pred_hits'] = int(len(pred_hit_times))
    row['pred_mean_tau'] = float(np.mean(pred_tau)) if len(pred_tau) else np.nan
    row['pred_std_tau'] = float(np.std(pred_tau)) if len(pred_tau) else np.nan
    row['pred_median_tau'] = float(np.median(pred_tau)) if len(pred_tau) else np.nan
    if len(ref_tau) and len(pred_tau):
        n = min(len(ref_tau), len(pred_tau))
        row['tau_seq_MAE'] = float(np.mean(np.abs(ref_tau[:n] - pred_tau[:n])))
        row['tau_dist_W1'] = wasserstein_1d_empirical(ref_tau, pred_tau)
    else:
        row['tau_seq_MAE'] = np.nan
        row['tau_dist_W1'] = np.nan
    return row

def build_fixed_sections(test_data, direction=0):
    """Paper order is channels 0, 1, 2; retain source IDs for CSV joins."""
    return [
        dict(name="Sigma1_u0_0", label="Sigma1: u0=0", section_dim=0, section_value=0.0, direction=direction),
        dict(name="Sigma3_u5_0", label="Sigma2 (paper): u1=0", section_dim=1, section_value=0.0, direction=direction),
        dict(name="Sigma2_u3_0", label="Sigma3 (paper): u2=0", section_dim=2, section_value=0.0, direction=direction),
    ]


def compute_start_indices(T_eval, horizon, n_starts):
    valid_max_idx = T_eval - horizon - 1
    if valid_max_idx <= 0:
        raise ValueError(f'test length too short: T_eval={T_eval}, horizon={horizon}')
    n = min(int(n_starts), valid_max_idx + 1)
    starts = np.linspace(0, valid_max_idx, n, dtype=int)
    return np.unique(starts)

def evaluate_prediction_metrics(loaded_models, test, dt, args, device):
    starts = compute_start_indices(len(test), args.pred_horizon, args.n_starts)
    print('\n============================================================')
    print('Prediction metrics: NMSE / W2 / PSD / TCM')
    print('============================================================')
    print('starts:', starts)
    print('pred_horizon:', args.pred_horizon)
    print('nmse_windows:', args.nmse_windows)
    (rows, errors) = ([], [])
    for item in loaded_models:
        (model, sf, net, scaler, ckpt) = (item['model'], item['seed_folder'], item['net'], item['scaler'], item['checkpoint'])
        print(f'\nEvaluating prediction metrics: {model}, {sf}')
        for start_idx in starts:
            try:
                gt = test[int(start_idx):int(start_idx) + args.pred_horizon + 1]
                pred = rollout_from_x0(net, scaler, test[int(start_idx)], args.pred_horizon, dt, device)
                row = {'model': model, 'seed_folder': sf, 'checkpoint': ckpt, 'start_idx': int(start_idx), 'pred_horizon': int(args.pred_horizon)}
                for k in args.nmse_windows:
                    row[f'NMSE_{k}'] = nmse_window_steps(gt, pred, k)
                row['Attractor_W2'] = sinkhorn_w2_cloud(gt, pred, blur=args.w2_blur, device=device, max_points=args.w2_max_points)
                row['PSD_logL1'] = psd_logl1_distance(gt, pred, dt=dt, nfft=args.psd_nfft)
                row['TCM'] = tcm_distance(gt, pred, dt=dt, max_lag=args.tcm_max_lag, stop_first_nonpositive=args.tcm_truncate_first_nonpositive)
                rows.append(row)
            except Exception as e:
                errors.append({'stage': 'prediction_metrics', 'model': model, 'seed_folder': sf, 'checkpoint': ckpt, 'start_idx': int(start_idx), 'error': repr(e)})
                print(f'[warn] prediction eval failed: {model}, {sf}, start={start_idx}: {e}')
    return (rows, errors)

def evaluate_tauW1_original_logic(loaded_models, test, dt, args, device):
    sections = build_fixed_sections(test, direction=args.section_direction)
    starts = compute_start_indices(len(test), args.poincare_rollout_len, args.poincare_num_starts)
    print('\n============================================================')
    print('Poincare tau_dist_W1: ORIGINAL LOGIC')
    print('============================================================')
    print('poincare_starts:', starts)
    print('poincare_rollout_len:', args.poincare_rollout_len)
    print('section_direction:', args.section_direction)
    for sec in sections:
        print(f"{sec['name']}: {sec['label']}, direction={sec['direction']:+d}")
    (rows, errors) = ([], [])
    for item in loaded_models:
        (model, sf, net, scaler, ckpt) = (item['model'], item['seed_folder'], item['net'], item['scaler'], item['checkpoint'])
        print(f'\nEvaluating tau_dist_W1: {model}, {sf}')
        for start_idx in starts:
            try:
                ref = np.asarray(test[int(start_idx):int(start_idx) + args.poincare_rollout_len + 1], dtype=np.float64)
                pred = rollout_from_x0(net, scaler, test[int(start_idx)], args.poincare_rollout_len, dt, device)
                for sec in sections:
                    (ref_hit_t, ref_hit_x, ref_tau) = extract_poincare_hits(ref, dt, section_dim=sec['section_dim'], section_value=sec['section_value'], direction=sec['direction'])
                    (pred_hit_t, pred_hit_x, pred_tau) = extract_poincare_hits(pred, dt, section_dim=sec['section_dim'], section_value=sec['section_value'], direction=sec['direction'])
                    out = summarize_poincare_tau(ref_tau, pred_tau, pred_hit_t)
                    row = {'model': model, 'seed_folder': sf, 'checkpoint': ckpt, 'start_idx': int(start_idx), 'poincare_rollout_len': int(args.poincare_rollout_len), 'section_name': sec['name'], 'section_label': sec['label'], 'section_dim': sec['section_dim'], 'section_value': sec['section_value'], 'section_direction': sec['direction'], 'gt_hits': int(len(ref_hit_t)), 'gt_mean_tau': float(np.mean(ref_tau)) if len(ref_tau) else np.nan, 'gt_std_tau': float(np.std(ref_tau)) if len(ref_tau) else np.nan, 'gt_median_tau': float(np.median(ref_tau)) if len(ref_tau) else np.nan}
                    row.update(out)
                    rows.append(row)
            except Exception as e:
                errors.append({'stage': 'tauW1_original_logic', 'model': model, 'seed_folder': sf, 'checkpoint': ckpt, 'start_idx': int(start_idx), 'error': repr(e)})
                print(f'[warn] tauW1 eval failed: {model}, {sf}, start={start_idx}: {e}')
    return (rows, errors)

def summarize_prediction(rows, nmse_windows):
    out = []
    for model in MODEL_ORDER:
        group = [r for r in rows if r['model'] == model]
        if not group:
            continue
        row = {'model': model, 'n_prediction_rollouts': int(len(group))}
        for metric in [f'NMSE_{k}' for k in nmse_windows] + ['Attractor_W2', 'PSD_logL1', 'TCM']:
            vals = [as_float_or_nan(g.get(metric, np.nan)) for g in group]
            (mu, sd, n) = finite_mean_std(vals)
            row[f'{metric}_mean'] = mu
            row[f'{metric}_std'] = sd
            row[f'{metric}_n_valid'] = n
            row[metric] = fmt_mean_std(mu, sd)
        out.append(row)
    return out

def summarize_tauW1(rows):
    out = []
    section_order = []
    for r in rows:
        name = r['section_name']
        if name not in section_order:
            section_order.append(name)
    for sec_name in section_order:
        sec_rows = [r for r in rows if r['section_name'] == sec_name]
        label = sec_rows[0]['section_label'] if sec_rows else sec_name
        for model in MODEL_ORDER:
            group = [r for r in sec_rows if r['model'] == model]
            if not group:
                continue
            row = {'section_name': sec_name, 'section_label': label, 'model': model, 'records': int(len(group))}
            for metric in ['gt_hits', 'pred_hits', 'gt_mean_tau', 'pred_mean_tau', 'gt_std_tau', 'pred_std_tau', 'gt_median_tau', 'pred_median_tau', 'tau_seq_MAE', 'tau_dist_W1']:
                vals = [as_float_or_nan(g.get(metric, np.nan)) for g in group]
                (mu, sd, n) = finite_mean_std(vals)
                row[f'{metric}_mean'] = mu
                row[f'{metric}_std'] = sd
                row[f'{metric}_n_valid'] = n
                row[metric] = fmt_mean_std(mu, sd)
            row['tau_dist_W1_valid_count'] = row['tau_dist_W1_n_valid']
            out.append(row)
    return out

def write_markdown_summary(path, pred_summary, tau_summary, nmse_windows):
    with open(path, 'w') as f:
        f.write('# KS evaluation: prediction metrics + original tau_dist_W1 logic\n\n')
        f.write('Prediction metrics use 5 seed folders × 32 starts by default. Poincare `tau_dist_W1` uses the original crossing-time logic: linearly interpolated hit times, `tau=np.diff(hit_times)`, and `wasserstein_1d_empirical(ref_tau, pred_tau)`.\n\n')
        f.write('## Prediction / structure / temporal metrics\n\n')
        f.write('| Model | n | ' + ' | '.join([f'NMSE_{k}' for k in nmse_windows]) + ' | Attractor W2 | PSD logL1 | TCM |\n')
        f.write('|---|---:|' + '---:|' * (len(nmse_windows) + 3) + '\n')
        for r in pred_summary:
            f.write(f"| {r['model']} | {r['n_prediction_rollouts']} | " + ' | '.join([r.get(f'NMSE_{k}', 'nan') for k in nmse_windows]) + f" | {r.get('Attractor_W2', 'nan')} | {r.get('PSD_logL1', 'nan')} | {r.get('TCM', 'nan')} |\n")
        f.write('\n## Poincare `tau_dist_W1` — original logic\n\n')
        for sec_name in [] if not tau_summary else []:
            pass
        sections = []
        for r in tau_summary:
            if r['section_name'] not in sections:
                sections.append(r['section_name'])
        for sec_name in sections:
            label = next((r['section_label'] for r in tau_summary if r['section_name'] == sec_name))
            f.write(f'### {label}\n\n')
            f.write('| Model | records | GT hits | Pred hits | tau_dist_W1 | valid |\n')
            f.write('|---|---:|---:|---:|---:|---:|\n')
            for model in MODEL_ORDER:
                r = next((x for x in tau_summary if x['section_name'] == sec_name and x['model'] == model), None)
                if r is None:
                    continue
                f.write(f"| {model} | {r['records']} | {r['gt_hits']} | {r['pred_hits']} | {r['tau_dist_W1']} | {r['tau_dist_W1_valid_count']} |\n")
            f.write('\n')
