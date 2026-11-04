"""Historical CFD training recipes, parameterized from new_try.py and NN_Cylinder_CFD.py.
See docs/cylinder.md for per-seed historical differences and checkpoint selection.
"""
import argparse
import csv
import hashlib
import json
import pickle
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from geomloss import SamplesLoss
from sklearn.preprocessing import StandardScaler
from evaluate import load_p

ROOT = Path(__file__).resolve().parents[2]


def train_measure(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = args.device
    print("device:", device)

    # ============================================================
    # 1. Data Loading
    # ============================================================
    data = load_p(args.data_dir / "flow_sensors.p")

    training = data[0]
    testing_true = data[1]
    testing_noised = data[2]
    data_sample_rate = int(data[3])
    dt_data = float(data[4])
    obs_error = float(data[5])
    dt_data_sub = float(data[6])
    dx_phys = float(data[7])

    m = training.shape[1]
    print("m probes =", m)

    # ============================================================
    # 2. Hyperparameters & Config
    # ============================================================
    ode_sampling_freq = 10
    dt = dt_data_sub / ode_sampling_freq
    tau = 2
    dim_delay = 4
    tau_delay = tau
    tau_pair = tau * ode_sampling_freq

    print("dt_data_sub:", dt_data_sub, "dt(fine):", dt, "tau_pair:", tau_pair)

    Nsteps = args.steps
    warm_steps = 3000
    stage2_steps_per_cycle = 1
    stage1_steps_per_cycle = 3

    stage1_batch = 512
    L_hits_fullbatch = 16
    BATCH_SIZE_STAGE2 = 8

    blur_start = 0.20
    blur_end   = 0.07

    w_defect = 0
    alpha_pair_state = 1.0
    alpha_pair_delay = 0.7

    alpha_mse = args.alpha_mse

    w_marg_u0 = 0.20
    w_marg_u1 = 0.20

    target2 = 0.05
    w2_min = 1e-5
    w2_max = 1.0
    w2_ramp_steps = 200
    ema_beta = 0.98
    ema_L2 = None

    print_every = 50

    USE_RP = True
    k_u = 16
    k_pair_u = 32
    k_single_delay = 32  # RP dimension for stage1 delay marginal-measure

    def make_ortho_proj(in_dim, out_dim, seed_offset=0):
        g = torch.Generator(device=device)
        g.manual_seed(seed + seed_offset)
        A = torch.randn(in_dim, out_dim, generator=g, device=device)
        Q, _ = torch.linalg.qr(A)
        P = Q.T.contiguous()
        return P

    def proj(X, P):
        return X @ P.T

    # ============================================================
    # 3. Preprocessing
    # ============================================================
    scaler = StandardScaler()
    scaler.fit(training)

    Y_train = scaler.transform(training)
    Y_ref_sub = Y_train.copy()

    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    Y_ref_sub_t = torch.tensor(Y_ref_sub, dtype=torch.float32, device=device)

    N_train = Y_train.shape[0]
    N_ref = Y_ref_sub.shape[0]
    print("N_train:", N_train, "N_ref_sub (train-only):", N_ref)

    x_mean_all = Y_ref_sub.mean(axis=0)
    x_std_all  = Y_ref_sub.std(axis=0) + 1e-6
    x_mean_all_t = torch.tensor(x_mean_all, dtype=torch.float32, device=device)
    x_std_all_t  = torch.tensor(x_std_all,  dtype=torch.float32, device=device)

    def whiten_x(x: torch.Tensor):
        return (x - x_mean_all_t[None, :]) / x_std_all_t[None, :]

    USE_JOINT_DELAY_ALL_CHANNELS = True
    DELAY_CHANNEL = 8

    def delay_vectors_at_indices_1d(x_series_1d: np.ndarray, idx: np.ndarray, tau: int, dim: int):
        out = np.zeros((len(idx), dim), dtype=np.float64)
        for a, t in enumerate(idx):
            for i in range(dim):
                out[a, i] = x_series_1d[t - i * tau]
        return out

    def delay_vectors_at_indices_all(Y_series: np.ndarray, idx: np.ndarray, tau: int, dim: int):
        out = np.zeros((len(idx), dim * Y_series.shape[1]), dtype=np.float64)
        for a, t in enumerate(idx):
            pieces = []
            for i in range(dim):
                pieces.append(Y_series[t - i * tau])
            out[a] = np.concatenate(pieces, axis=0)
        return out

    SECTION_CHANNEL = 8
    z0_s = float(np.mean(Y_ref_sub[:, SECTION_CHANNEL]))
    delta_z_s = float(1 * np.std(Y_ref_sub[:, SECTION_CHANNEL]))
    L_sec = 3

    def compute_section_hits_scaled_numpy(Y_scaled: np.ndarray, ch: int, z0: float, delta: float, dt_coarse: float, L_sec: int):
        z = Y_scaled[:, ch]
        dz = (z[1:] - z[:-1]) / dt_coarse
        hits = []
        last = -10**12
        for t in range(len(z) - 1):
            if abs(z[t] - z0) <= delta and dz[t] > 0:
                if t - last >= L_sec:
                    hits.append(t)
                    last = t
        return np.array(hits, dtype=np.int64)

    hist_len_delay = (dim_delay - 1) * tau_delay
    min_t = hist_len_delay

    hits_data = compute_section_hits_scaled_numpy(
        Y_ref_sub, ch=SECTION_CHANNEL, z0=z0_s, delta=delta_z_s, dt_coarse=dt_data_sub, L_sec=L_sec
    )
    hits_data = hits_data[hits_data >= min_t]
    hits_data = np.sort(hits_data)

    print(f"TRAIN-REF: usable section hits N_U={len(hits_data)} | z0={z0_s:.3f}, delta={delta_z_s:.3f}")
    if len(hits_data) < 50:
        print("Warning: Too few section hits on TRAIN. Stage2 may be weak/unstable.")
    if len(hits_data) < 4:
        print("Warning: Hits < 4. Stage2 will effectively fall back to Stage1-only in loop.")

    XU_data = Y_ref_sub[hits_data]

    if USE_JOINT_DELAY_ALL_CHANNELS:
        DU_data = delay_vectors_at_indices_all(Y_ref_sub, hits_data, tau=tau_delay, dim=dim_delay)
    else:
        x_series = Y_ref_sub[:, DELAY_CHANNEL]
        DU_data = delay_vectors_at_indices_1d(x_series, hits_data, tau=tau_delay, dim=dim_delay)

    # delay stats for stage2 U-space
    U_hits_data = np.concatenate([XU_data, DU_data], axis=1)
    d_u = U_hits_data.shape[1]

    u_mean = U_hits_data.mean(axis=0)
    u_std  = U_hits_data.std(axis=0) + 1e-6
    u_mean_t = torch.tensor(u_mean, dtype=torch.float32, device=device)
    u_std_t  = torch.tensor(u_std,  dtype=torch.float32, device=device)

    def whiten_u(u: torch.Tensor):
        return (u - u_mean_t[None, :]) / u_std_t[None, :]

    U_hits_data_t = torch.tensor(U_hits_data, dtype=torch.float32, device=device)
    U_hits_data_dw = whiten_u(U_hits_data_t)

    # delay-only stats for Stage1 delay measure
    valid_delay_idx_all = np.arange(min_t, len(Y_ref_sub), dtype=np.int64)
    if USE_JOINT_DELAY_ALL_CHANNELS:
        D_ref_all = delay_vectors_at_indices_all(Y_ref_sub, valid_delay_idx_all, tau=tau_delay, dim=dim_delay)
    else:
        x_series = Y_ref_sub[:, DELAY_CHANNEL]
        D_ref_all = delay_vectors_at_indices_1d(x_series, valid_delay_idx_all, tau=tau_delay, dim=dim_delay)

    d_delay_dim = D_ref_all.shape[1]
    d_mean = D_ref_all.mean(axis=0)
    d_std  = D_ref_all.std(axis=0) + 1e-6
    d_mean_t = torch.tensor(d_mean, dtype=torch.float32, device=device)
    d_std_t  = torch.tensor(d_std,  dtype=torch.float32, device=device)

    def whiten_d(d: torch.Tensor):
        return (d - d_mean_t[None, :]) / d_std_t[None, :]

    # ============================================================
    # 4. Model Setup
    # ============================================================
    net = nn.Sequential(
        nn.Linear(m, 100), nn.Tanh(),
        nn.Linear(100, 100), nn.Tanh(),
        nn.Linear(100, 100), nn.Tanh(),
        nn.Linear(100, m),
    ).to(device)

    optimizer1 = optim.Adam(net.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(net.parameters(), lr=1e-5)

    def rollout_steps(x0: torch.Tensor, steps: int):
        x = x0
        for _ in range(int(steps)):
            x = x + dt * net(x)
        return x

    def make_sinkhorn(blur):
        return SamplesLoss(loss="sinkhorn", p=2, blur=float(blur))

    if USE_RP:
        P_u = make_ortho_proj(in_dim=d_u, out_dim=k_u, seed_offset=2)
        P_pair_u = make_ortho_proj(in_dim=2 * k_u, out_dim=k_pair_u, seed_offset=3)
        # 修改为了投影单维度 Delay
        P_single_delay = make_ortho_proj(in_dim=d_delay_dim, out_dim=k_single_delay, seed_offset=4)

    # ============================================================
    # 5. Helpers for model-generated blocks
    # ============================================================
    def rollout_record_coarse_batched(x_start: torch.Tensor, n_coarse_steps: int):
        xs = [x_start]
        x = x_start
        for _ in range(int(n_coarse_steps)):
            x = rollout_steps(x, steps=ode_sampling_freq)
            xs.append(x)
        return torch.stack(xs, dim=1)

    def rollout_record_coarse_single(x_start: torch.Tensor, n_coarse_steps: int, detach_offsets=None, defect_detach=True):
        if detach_offsets is None:
            detach_offsets = []
        detach_offsets = set(int(v) for v in detach_offsets)

        xs = [x_start]
        x = x_start
        for k in range(1, int(n_coarse_steps) + 1):
            x = rollout_steps(x, steps=ode_sampling_freq)
            xs.append(x)
            if defect_detach and (k in detach_offsets):
                x = x.detach()
        return torch.stack(xs, dim=0)

    def extract_delay_from_batched_seq(seq_coarse: torch.Tensor, center_offsets, tau: int, dim: int):
        outs = []
        for c in center_offsets:
            pieces = []
            for i in range(dim):
                pos = int(c - i * tau)
                if USE_JOINT_DELAY_ALL_CHANNELS:
                    pieces.append(seq_coarse[:, pos, :])
                else:
                    pieces.append(seq_coarse[:, pos, DELAY_CHANNEL:DELAY_CHANNEL+1])
            outs.append(torch.cat(pieces, dim=1))
        return outs

    def extract_delay_from_single_seq(seq_coarse: torch.Tensor, center_offsets, tau: int, dim: int):
        all_delays = []
        for c in center_offsets:
            pieces = []
            for i in range(dim):
                pos = int(c - i * tau)
                if USE_JOINT_DELAY_ALL_CHANNELS:
                    pieces.append(seq_coarse[pos])
                else:
                    pieces.append(seq_coarse[pos, DELAY_CHANNEL:DELAY_CHANNEL+1])
            all_delays.append(torch.cat(pieces, dim=0))
        return torch.stack(all_delays, dim=0)

    # ============================================================
    # 6. Stage1: Pushforward + Delay Marginal-Measure
    # ============================================================
    def stage1_sample_indices(batch_size: int, tau_pair: int):
        coarse_shift = int(tau_pair // ode_sampling_freq)
        low = min_t
        high = len(Y_ref_sub) - coarse_shift 
        if high <= low:
            raise ValueError("Not enough data to sample stage1 indices with valid delay history.")
        idx = np.random.randint(low, high, size=batch_size)
        return idx, coarse_shift

    def stage1_delay_pairs(idx: np.ndarray, coarse_shift: int):
        # ====== Data Delay ======
        if USE_JOINT_DELAY_ALL_CHANNELS:
            D0_data_np = delay_vectors_at_indices_all(Y_ref_sub, idx, tau=tau_delay, dim=dim_delay)
            D1_data_np = delay_vectors_at_indices_all(Y_ref_sub, idx + coarse_shift, tau=tau_delay, dim=dim_delay)
        else:
            x_series = Y_ref_sub[:, DELAY_CHANNEL]
            D0_data_np = delay_vectors_at_indices_1d(x_series, idx, tau=tau_delay, dim=dim_delay)
            D1_data_np = delay_vectors_at_indices_1d(x_series, idx + coarse_shift, tau=tau_delay, dim=dim_delay)

        D0_data = torch.tensor(D0_data_np, dtype=torch.float32, device=device)
        D1_data = torch.tensor(D1_data_np, dtype=torch.float32, device=device)

        # ====== Data Pushforward ======
        X0_data = torch.tensor(Y_ref_sub[idx], dtype=torch.float32, device=device)
        X1_data = torch.tensor(Y_ref_sub[idx + coarse_shift], dtype=torch.float32, device=device)

        # ====== Model Generated ======
        start_idx = idx - hist_len_delay
        x_start = torch.tensor(Y_ref_sub[start_idx], dtype=torch.float32, device=device)

        seq_model = rollout_record_coarse_batched(
            x_start=x_start,
            n_coarse_steps=hist_len_delay + coarse_shift
        )

        center0 = hist_len_delay
        center1 = hist_len_delay + coarse_shift
    
        # Model Delay
        D0_model, D1_model = extract_delay_from_batched_seq(
            seq_model, [center0, center1], tau=tau_delay, dim=dim_delay
        )

        # Model Pushforward
        X0_model = seq_model[:, center0, :]
        X1_model = seq_model[:, center1, :]

        return D0_data, D1_data, D0_model, D1_model, X0_data, X1_data, X0_model, X1_model


    def stage1_loss_components(batch_size: int, tau_pair: int, blur: float):
        loss_sink = make_sinkhorn(blur)

        idx, coarse_shift = stage1_sample_indices(
            batch_size=batch_size, tau_pair=tau_pair
        )

        D0_data, D1_data, D0_model, D1_model, X0_data, X1_data, X0_model, X1_model = stage1_delay_pairs(
            idx=idx, coarse_shift=coarse_shift
        )

        # ==========================================
        # 1. Delay Measure (Marginal Measure Only)
        # ==========================================
        D0_dw = whiten_d(D0_data)
        D1_dw = whiten_d(D1_data)
        D0_mw = whiten_d(D0_model)
        D1_mw = whiten_d(D1_model)

        # 在 batch 维度拼接，把 D0 和 D1 当作独立的样本点来匹配分布 (不再配对)
        marg_d_delay = torch.cat([D0_dw, D1_dw], dim=0)
        marg_m_delay = torch.cat([D0_mw, D1_mw], dim=0)

        if USE_RP:
            marg_d_delay = proj(marg_d_delay, P_single_delay)
            marg_m_delay = proj(marg_m_delay, P_single_delay)

        L_delay = loss_sink(marg_m_delay, marg_d_delay)

        # ==========================================
        # 2. Pushforward Measure (保持 Graph Measure)
        # ==========================================
        X0_dw = whiten_x(X0_data)
        X1_dw = whiten_x(X1_data)
        X0_mw = whiten_x(X0_model)
        X1_mw = whiten_x(X1_model)

        pair_d_state = torch.cat([X0_dw, X1_dw], dim=1)
        pair_m_state = torch.cat([X0_mw, X1_mw], dim=1)

        L_push = loss_sink(pair_m_state, pair_d_state)

        # ==========================================
        L_mse_X = torch.mean((X0_mw - X0_dw)**2) + torch.mean((X1_mw - X1_dw)**2)
        # 计算预测延迟 D 与真实延迟 D 的均方误差
        L_mse_D = torch.mean((D0_mw - D0_dw)**2) + torch.mean((D1_mw - D1_dw)**2)
    
        # 汇总 Pointwise MSE
        L_mse = L_mse_X + L_mse_D 

        # ==========================================
        # 3. Total Loss
        # ==========================================
        L_total = alpha_pair_delay * L_delay + alpha_pair_state * L_push + alpha_mse * L_mse

        return L_total, L_delay, L_mse


    @torch.no_grad()
    def stage1_diag_loss(tau_pair: int, batch_size: int, blur: float):
        L_total, L_delay, L_mse = stage1_loss_components(
            batch_size=batch_size, tau_pair=tau_pair, blur=blur
        )
        # 修改后：括号严格对应每个 float 转换
        return float(L_total.item()), float(L_delay.item()), float(L_mse.item())

    # ============================================================
    # 7. Stage2 with model-generated delay block
    # ============================================================
    def build_multishoot_induced_block_fullbatch(k0: int, L_hits: int, defect_detach=True):
        assert 0 <= k0 and (k0 + L_hits) <= len(hits_data)

        hit_times = hits_data[k0:k0+L_hits].astype(np.int64)

        U_blk_d_dw = U_hits_data_dw[k0:k0+L_hits]
        X_blk_d = torch.tensor(XU_data[k0:k0+L_hits], dtype=torch.float32, device=device)
        D_blk_d = torch.tensor(DU_data[k0:k0+L_hits], dtype=torch.float32, device=device)

        start_t = int(hit_times[0] - hist_len_delay)
        end_t   = int(hit_times[-1])
        n_coarse_steps = end_t - start_t

        x_start = torch.tensor(Y_ref_sub[start_t], dtype=torch.float32, device=device)

        hit_offsets = (hit_times - start_t).astype(np.int64)
        detach_offsets = list(hit_offsets[:-1]) if defect_detach else []

        seq_model = rollout_record_coarse_single(
            x_start=x_start,
            n_coarse_steps=n_coarse_steps,
            detach_offsets=detach_offsets,
            defect_detach=defect_detach
        )

        hit_offsets_t = torch.tensor(hit_offsets, dtype=torch.long, device=device)
        X_blk_m = seq_model[hit_offsets_t]

        D_blk_m = extract_delay_from_single_seq(
            seq_model,
            center_offsets=hit_offsets,
            tau=tau_delay,
            dim=dim_delay
        )

        U_blk_m = torch.cat([X_blk_m, D_blk_m], dim=1)
        U_blk_m_dw = whiten_u(U_blk_m)

        def split_weight(Udw):
            xw = Udw[:, :m]
            dw = Udw[:, m:]
            return torch.cat([alpha_pair_state * xw, alpha_pair_delay * dw], dim=1)

        u_d = split_weight(U_blk_d_dw)
        u_m = split_weight(U_blk_m_dw)

        if USE_RP:
            u_d = proj(u_d, P_u)
            u_m = proj(u_m, P_u)

        u0_d, u1_d = u_d[:-1], u_d[1:]
        u0_m, u1_m = u_m[:-1], u_m[1:]

        pair_d = torch.cat([u0_d, u1_d], dim=1)
        pair_m = torch.cat([u0_m, u1_m], dim=1)

        if USE_RP:
            pair_d = proj(pair_d, P_pair_u)
            pair_m = proj(pair_m, P_pair_u)

        state_m = whiten_x(X_blk_m)
        state_d = whiten_x(X_blk_d)
        defect_loss = torch.mean((state_m - state_d) ** 2)

        return pair_m, pair_d, u0_m, u0_d, u1_m, u1_d, defect_loss

    # ============================================================
    # 8. Training Loop
    # ============================================================
    best_L2_val = float("inf")
    best_model_state_dict = None
    best_iter = -1

    loss_total_hist = []
    loss_stage1_hist = []
    loss_stage2_raw_hist = []
    w2_hist = []
    stage_used_hist = []
    stage2_update_count = 0

    print("============================================================")
    print("Start TwoStage (STRICT NO-LEAKAGE) on CFD")
    print("============================================================")

    net.train()

    for step in range(Nsteps):
        t = step / max(1, Nsteps - 1)
        blur = blur_start + (blur_end - blur_start) * t
        loss_sink = make_sinkhorn(blur)

        if step < warm_steps:
            stage_used = "stage1"
        else:
            cyc = stage2_steps_per_cycle + stage1_steps_per_cycle
            k = (step - warm_steps) % cyc
            stage_used = "stage2" if k < stage2_steps_per_cycle else "stage1"

        curr_opt = optimizer1 if stage_used == "stage1" else optimizer2
        curr_opt.zero_grad()

        if stage_used == "stage1":
            L_total, L1_delay, L1_mse = stage1_loss_components(
                batch_size=stage1_batch, tau_pair=tau_pair, blur=blur
            )
            L2_raw_avg = torch.tensor(0.0, device=device)
            w2_eff = 0.0

        else:
            if len(hits_data) < 4:
                L_total, L1_delay, L1_mse = stage1_loss_components(
                    batch_size=stage1_batch, tau_pair=tau_pair, blur=blur
                )
                L2_raw_avg = torch.tensor(0.0, device=device)
                w2_eff = 0.0
            else:
                stage2_update_count += 1
                l2_losses = []

                L_hits = int(min(len(hits_data), L_hits_fullbatch))
                k0_max = len(hits_data) - L_hits

                for _ in range(BATCH_SIZE_STAGE2):
                    k0 = int(np.random.randint(0, max(1, k0_max + 1)))
                    pair_m, pair_d, u0_m, u0_d, u1_m, u1_d, L_def = build_multishoot_induced_block_fullbatch(
                        k0=k0, L_hits=L_hits, defect_detach=True
                    )

                    L2_graph = loss_sink(pair_m, pair_d)
                    L2_marg0 = loss_sink(u0_m, u0_d)
                    L2_marg1 = loss_sink(u1_m, u1_d)
                    L2_marg  = w_marg_u0 * L2_marg0 + w_marg_u1 * L2_marg1

                    single_l2 = (L2_graph + L2_marg + w_defect * L_def) / float(L_hits)
                    l2_losses.append(single_l2)

                L2_raw_avg = torch.stack(l2_losses).mean()

                with torch.no_grad():
                    val = float(L2_raw_avg.detach().item())
                    if ema_L2 is None:
                        ema_L2 = val
                    else:
                        ema_L2 = ema_beta * ema_L2 + (1.0 - ema_beta) * val

                    w2_auto = target2 / max(ema_L2, 1e-12)
                    w2_auto = float(np.clip(w2_auto, w2_min, w2_max))
                    ramp = min(1.0, stage2_update_count / max(1, w2_ramp_steps))
                    w2_eff = float(ramp * w2_auto)

                L_total = (w2_eff * L2_raw_avg)

                cur = float(L2_raw_avg.detach().item())
                if cur < best_L2_val:
                    best_L2_val = cur
                    best_iter = step
                    best_model_state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    print(f"  --> [New Best] Iter {step} | L2_raw_avg={best_L2_val:.4f}")

        L_total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        curr_opt.step()

        loss_total_hist.append(float(L_total.detach().item()))
        w2_hist.append(float(w2_eff))
        loss_stage2_raw_hist.append(float(L2_raw_avg.detach().item()))
        stage_used_hist.append(stage_used)

        if step % print_every == 0:
            s1_diag_total, s1_diag_delay,s1_diag_mse = stage1_diag_loss(
                tau_pair=tau_pair, batch_size=stage1_batch, blur=blur
            )
            loss_stage1_hist.append(s1_diag_total)

            if stage_used == "stage1":
                print(
                    f"Iter {step:5d} [stage1] | "
                    f"L_tot={loss_total_hist[-1]:.6f} | "
                    f"Stage1_delay_diag={s1_diag_delay:.6f} | "
                    f"L_mse={s1_diag_mse:.6f} | " 
                    f"blur={blur:.3f}"
                )
            else:
                print(
                    f"Iter {step:5d} [stage2] | "
                    f"L_tot={loss_total_hist[-1]:.6f} | "
                    f"L2_raw={loss_stage2_raw_hist[-1]:.4f} | "
                    f"w2={w2_eff:.6f} | "
                    f"L_delay={s1_diag_delay:.6f} | "  # Stage 2 阶段也监控 Stage 1的指标
                    f"L_mse={s1_diag_mse:.6f} | "
                    f"blur={blur:.3f}"
                )

    print("Training complete.")
    print("Best stage2 iter:", best_iter, "| best_L2:", best_L2_val)

    # ============================================================
    # 9. Save and Evaluate
    # ============================================================


    save_dict_best = {
        "config": {
            "seed": seed,
            "model_name": args.model,
            "training_steps": Nsteps,
            "alpha_mse": alpha_mse,
            "warm_steps": warm_steps,
            "dt_fine": dt,
            "dt_data_sub": dt_data_sub,
            "ode_sampling_freq": ode_sampling_freq,
            "tau": tau,
            "tau_pair": tau_pair,
            "dim_delay": dim_delay,
            "tau_delay": tau_delay,
            "USE_RP": USE_RP,
            "SECTION_CHANNEL": SECTION_CHANNEL,
            "z0_s": z0_s,
            "delta_z_s": delta_z_s,
            "L_sec": L_sec,
            "best_iter": best_iter,
            "best_L2": best_L2_val,
            "N_train": int(N_train),
            "N_ref_sub": int(N_ref),
        },
        "model": {
            "state_dict": best_model_state_dict if best_model_state_dict is not None else {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "scaler": scaler,
        },
        "train_hist": {
            "loss_total": np.array(loss_total_hist),
            "loss_diag": np.array(loss_stage1_hist),
            "loss_stage2_raw": np.array(loss_stage2_raw_hist),
            "w2": np.array(w2_hist),
            "stage": np.array(stage_used_hist),
        }
    }

    filename_best = args.checkpoint_path
    with open(filename_best, "wb") as f:
        pickle.dump(save_dict_best, f)
    print("Saved BEST to:", filename_best)


    return save_dict_best


def train_mse(args):
    """The original pointwise delay-block MSE, including its RNG initialization order."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = load_p(args.data_dir / 'flow_sensors.p')
    coeffs = data[0]
    scaler = StandardScaler().fit(coeffs)
    y = torch.tensor(scaler.transform(coeffs), dtype=torch.float32, device=args.device)
    m, tau, delay_dim, microsteps = coeffs.shape[1], 2, 4, 10
    dt = float(data[6]) / microsteps
    target = torch.zeros((m, len(y) - tau * (delay_dim - 1), delay_dim), device=args.device)
    for i in range(delay_dim):
        target[:, :, i] = y[delay_dim*tau - (i+1)*tau:][:len(y)-tau*(delay_dim-1)].T

    class FlowStep(nn.Module):
        def __init__(self):
            super().__init__()
            self.L1 = nn.Linear(m, 100)
            self.T1 = nn.Tanh()
            self.L2 = nn.Linear(100, 100)
            self.T2 = nn.Tanh()
            # The source overwrote L2; retain the extra RNG draws for exact initialization.
            self.L2 = nn.Linear(100, 100)
            self.T2 = nn.Tanh()
            self.L3 = nn.Linear(100, m)
        def forward(self, x):
            return self.L3(self.T2(self.L2(self.T1(self.L1(x.clone())))))

    net = FlowStep().to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    best_loss, best_iter, best_state = float('inf'), -1, None
    history = []
    for step in range(args.steps):
        optimizer.zero_grad()
        states, yy = [], y.clone()
        for _ in range(tau * delay_dim * microsteps):
            states.append(yy)
            yy = net(yy) * dt + yy
        sol = torch.stack(states, dim=0)[:, :-tau*(delay_dim-1), :]
        predicted = torch.flip(sol.permute(2,1,0)[:, :, ::tau*microsteps], [2])
        loss = torch.mean((predicted - target)**2)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite training loss at step {step}')
        loss.backward()
        optimizer.step()
        current = float(loss.detach())
        history.append(current)
        # Historical convention: pre-update loss selects the post-update weights.
        if current < best_loss:
            best_loss, best_iter = current, step
            best_state = {k: v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if step % 100 == 0:
            print(f'Iter {step}: loss={current:.9g}', flush=True)
    payload = {'state_dict': best_state, 'dt': dt, 'scaler': scaler,
               'ode_sampling_freq': microsteps, 'method': 'P', 'best_loss': best_loss,
               'config': {'seed': args.seed, 'model_name': 'MSE', 'training_steps': args.steps,
                          'best_iter': best_iter, 'dt_fine': dt},
               'train_hist': {'loss_total': np.asarray(history)}}
    with args.checkpoint_path.open('wb') as f:
        pickle.dump(payload, f)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description='Fresh CFD training with historical per-model/per-seed defaults.')
    parser.add_argument('--model', required=True, choices=['PRIME','MSE','DIM','DIM+MSE'])
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--steps', type=int, help='Override total training steps (also changes the blur schedule).')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/cylinder')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/cylinder/trained')
    parser.add_argument('--alpha-mse', type=float, help='Explicit recipe override; recorded in the manifest.')
    return parser.parse_args()


def main():
    args = parse_args()
    historical_steps = 10000 if args.model == 'MSE' else (4000 if args.seed == 13532 else 5000) if args.model == 'PRIME' else 3000
    if args.steps is None:
        args.steps = historical_steps
    if args.steps < 1:
        raise ValueError('--steps must be positive')
    if args.alpha_mse is None:
        args.alpha_mse = 2.0 if args.model == 'DIM+MSE' or (args.model == 'PRIME' and args.seed == 123123123) else 0.0
    if args.device == 'auto':
        args.device = ('cpu' if args.model == 'MSE' else 'cuda' if torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    torch.set_num_threads(1)
    output = args.output_dir / f'Seed{args.seed}'
    output.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path = output / (args.model.replace('+','_') + '.p')
    if args.checkpoint_path.exists():
        raise FileExistsError(f'Refusing to replace {args.checkpoint_path}; choose a new --output-dir.')
    started = time.perf_counter()
    payload = train_mse(args) if args.model == 'MSE' else train_measure(args)
    history = payload['train_hist']['loss_total']
    with (output / (args.model.replace('+','_') + '_history.csv')).open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['step','loss_total']); writer.writerows(enumerate(history))
    manifest = {'model': args.model, 'seed': args.seed, 'steps': args.steps, 'historical_steps': historical_steps,
                'alpha_mse': args.alpha_mse if args.model != 'MSE' else None,
                'device': args.device, 'torch': torch.__version__, 'numpy': np.__version__,
                'fresh_initialization': True, 'elapsed_seconds': time.perf_counter()-started,
                'data_sha256': hashlib.sha256((args.data_dir/'flow_sensors.p').read_bytes()).hexdigest(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'checkpoint_sha256': hashlib.sha256(args.checkpoint_path.read_bytes()).hexdigest(),
                'checkpoint': str(args.checkpoint_path),
                'selection': ('minimum stage2 pre-update loss; final weights if no stage2 occurred' if args.model != 'MSE' else 'minimum pre-update loss selecting post-update weights')}
    (output / (args.model.replace('+','_') + '_manifest.json')).write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Saved {args.checkpoint_path}; elapsed {manifest["elapsed_seconds"]:.2f}s')


if __name__ == '__main__':
    main()
