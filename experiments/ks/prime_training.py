"""Current KS PRIME training from KS_test.ipynb, code cell 19.

Default profile: 10,000 DIM warmup steps followed by 2,000 PRIME steps.
See docs/ks.md for the older archived seeds with different training profiles.
"""
import os
import pickle
import numpy as np
import torch
from torch import nn as tnn, optim
from geomloss import SamplesLoss
from sklearn.preprocessing import MaxAbsScaler

def train_prime(train, seed, steps, warmup_steps, device, output_dir):
    dt_obs = 1.0
    stride = 1
    dt_base = dt_obs / stride
    sub = int(stride)
    cand_frac = 0.2
    q_speed = 0.3
    M_landmarks = 128
    L_sec = 5
    tau_delay = 10
    dim_delay = 3
    eps_in_mult = 0.35
    eps_out_mult = 1.8
    Nsteps = steps
    warm_steps = warmup_steps
    tau_pair = 2
    stage1_batch = 512
    lr1 = 0.001
    L_hits_fullbatch = 16
    BATCH_SIZE_STAGE2 = 16
    lr2 = 0.001
    alpha_pair_state = 1.0
    alpha_pair_delay = 0.7
    w_marg_u0 = 0.2
    w_marg_u1 = 0.2
    w_glue = 1
    blur_start = 0.2
    blur_end = 0.1
    stage2_steps_per_cycle = 1
    stage1_steps_per_cycle = 0
    w2_const = 3
    OUT_DIR = str(output_dir)
    SAVE_PREFIX = 'KS_Implicit_seed'
    os.makedirs(OUT_DIR, exist_ok=True)
    device = str(device)
    print('device:', device)

    def set_seed(s):
        import random
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)

    def make_sinkhorn(blur):
        return SamplesLoss(loss='sinkhorn', p=2, blur=float(blur), backend='tensorized')

    def build_net(d, width=100):
        return tnn.Sequential(tnn.Linear(d, width), tnn.Tanh(), tnn.Linear(width, width), tnn.Tanh(), tnn.Linear(width, width), tnn.Tanh(), tnn.Linear(width, d))

    def cpu_state_dict(net):
        return {k: v.detach().cpu().clone() for (k, v) in net.state_dict().items()}

    def build_delay_full_multich(y_scaled, tau, dim):
        """
        y_scaled: (T,d)
        returns:
          D_full: (T, d*dim), with lag blocks [y(t), y(t-tau), ..., y(t-(dim-1)tau)]
          min_t
        """
        (T, d) = y_scaled.shape
        min_t = (dim - 1) * tau
        D = np.zeros((T, d * dim), dtype=np.float64)
        for t in range(min_t, T):
            blocks = []
            for i in range(dim):
                blocks.append(y_scaled[t - i * tau])
            D[t] = np.concatenate(blocks, axis=0)
        return (D, min_t)

    def farthest_point_sampling(X, M, start=0):
        N = int(X.shape[0])
        M = int(min(int(M), N))
        idx = np.zeros((M,), dtype=np.int64)
        idx[0] = int(start)
        dist2 = np.sum((X - X[idx[0]]) ** 2, axis=1)
        for m in range(1, M):
            j = int(np.argmax(dist2))
            idx[m] = j
            d2_new = np.sum((X - X[j]) ** 2, axis=1)
            dist2 = np.minimum(dist2, d2_new)
        return idx

    def extract_hits_from_landmarks(U_dw, C_dw, eps_in, eps_out, L_sec):
        """
        hysteresis + min-gap (L_sec)
        also returns visit lengths (steps)
        """
        T = int(U_dw.shape[0])
        hits = []
        visit_lens = []
        last = -10 ** 12
        active = False
        active_i = -1
        enter_t = -1
        for t in range(T):
            d2 = np.sum((C_dw - U_dw[t]) ** 2, axis=1)
            i = int(np.argmin(d2))
            d = float(np.sqrt(d2[i]))
            if not active:
                if d <= eps_in and t - last >= int(L_sec):
                    hits.append(t)
                    last = t
                    active = True
                    active_i = i
                    enter_t = t
            else:
                d_active = float(np.sqrt(np.sum((U_dw[t] - C_dw[active_i]) ** 2)))
                if d_active >= eps_out:
                    active = False
                    active_i = -1
                    if enter_t >= 0:
                        visit_lens.append(t - enter_t)
                    enter_t = -1
        return (np.array(hits, dtype=np.int64), np.array(visit_lens, dtype=np.int64))

    def pstats(a):
        a = np.asarray(a)
        if len(a) == 0:
            return dict(median=np.nan, mean=np.nan, p10=np.nan, p90=np.nan)
        return dict(median=float(np.median(a)), mean=float(np.mean(a)), p10=float(np.quantile(a, 0.1)), p90=float(np.quantile(a, 0.9)))

    def build_delay_data_tensor(y_np, dim_delay, tau_delay, device):
        Y = torch.tensor(y_np, dtype=torch.float32, device=device)
        (T, d) = Y.shape
        L = T - tau_delay * (dim_delay - 1)
        delayed = torch.zeros((d, L, dim_delay), dtype=torch.float32, device=device)
        for i in range(dim_delay):
            start = dim_delay * tau_delay - (i + 1) * tau_delay
            seg = Y[start:start + L, :]
            delayed[:, :, i] = seg.T
        return delayed

    def forward_full_traj_for_stage1(net, Y_start, n_fine_steps, dt_base, device):
        """
        Mimic Cylinder DIM code:
          yy = y.clone()
          for j in range(tau*dim*sub):
              sols.append(yy)
              yy = yy + dt_base * net(yy)
        Y_start: torch.Tensor (T, d) on device
        """
        yy = Y_start.clone()
        sols = []
        for _ in range(n_fine_steps):
            sols.append(yy)
            yy = yy + float(dt_base) * net(yy)
        sol = torch.stack(sols, dim=0)
        return sol

    def build_delay_cloud_from_sol(sol, tau_delay, dim_delay, sub):
        """
        Build delay cloud DIM_model from stacked sol:
          sol: (Kfine, T, d)
          return: DIM_model: (d, L, dim_delay)
        """
        (Kfine, T, d) = sol.shape
        K_needed = tau_delay * dim_delay * sub
        if K_needed > Kfine:
            raise ValueError(f'Kfine={Kfine} < K_needed={K_needed}, increase n_fine_steps.')
        sol_use = sol[:K_needed]
        L = T - tau_delay * (dim_delay - 1)
        sol_use = sol_use[:, :L, :]
        sol_perm = sol_use.permute(2, 1, 0)
        step = tau_delay * sub
        DIM_model = torch.flip(sol_perm[:, :, ::step], dims=[2])
        return DIM_model
    set_seed(seed)
    train = np.asarray(train)
    (T, d) = train.shape
    scaler = MaxAbsScaler().fit(train)
    y_scaled = scaler.transform(train).astype(np.float64)
    (D_full, min_t) = build_delay_full_multich(y_scaled, tau=tau_delay, dim=dim_delay)
    U_full = np.concatenate([y_scaled, D_full], axis=1)
    T_eff = T - min_t
    dU = U_full.shape[1]
    speed = np.linalg.norm(y_scaled[1:] - y_scaled[:-1], axis=1)
    speed = np.concatenate([[speed[0]], speed], axis=0)
    idx_eff = np.arange(min_t, T)
    thr = np.quantile(speed[idx_eff], q_speed)
    pool = idx_eff[speed[idx_eff] >= thr]
    cand_target = int(round(cand_frac * T_eff))
    cand_target = max(1, cand_target)
    if len(pool) >= cand_target:
        idx_cand = np.random.choice(pool, size=cand_target, replace=False)
    else:
        idx_cand = pool.copy()
    idx_cand = np.sort(idx_cand)
    Ucand = U_full[idx_cand]
    u_mean_c = Ucand.mean(axis=0)
    u_std_c = Ucand.std(axis=0) + 1e-06
    Ucand_dw = (Ucand - u_mean_c[None, :]) / u_std_c[None, :]
    Ufull_dw = (U_full - u_mean_c[None, :]) / u_std_c[None, :]
    start0 = int(np.random.randint(0, len(idx_cand)))
    fps_idx_local = farthest_point_sampling(Ucand_dw, M=M_landmarks, start=start0)
    C_dw = Ucand_dw[fps_idx_local]
    nn = []
    for i in range(len(C_dw)):
        d2 = np.sum((C_dw - C_dw[i]) ** 2, axis=1)
        d2[i] = np.inf
        nn.append(np.sqrt(np.min(d2)))
    nn_med = float(np.median(nn))
    eps_in = eps_in_mult * nn_med
    eps_out = eps_out_mult * eps_in
    (hits_raw, visit_lens) = extract_hits_from_landmarks(Ufull_dw, C_dw, eps_in, eps_out, L_sec=L_sec)
    hits = hits_raw[hits_raw >= min_t]
    hits = np.sort(hits)
    taus = np.diff(hits).astype(np.float64)
    XU_data = y_scaled[hits]
    DU_data = D_full[hits]
    U_hits_data = np.concatenate([XU_data, DU_data], axis=1)
    L_delay = T - min_t
    D_slice = D_full[min_t:]
    D_t = torch.tensor(D_slice, dtype=torch.float32, device=device)
    y_delayed_data = D_t.view(L_delay, dim_delay, d).permute(2, 0, 1).contiguous()
    Y_all_t = torch.tensor(y_scaled, dtype=torch.float32, device=device)
    u_mean = U_hits_data.mean(axis=0)
    u_std = U_hits_data.std(axis=0) + 1e-06
    u_mean_t = torch.tensor(u_mean, dtype=torch.float32, device=device)
    u_std_t = torch.tensor(u_std, dtype=torch.float32, device=device)

    def whiten_u(U):
        return (U - u_mean_t[None, :]) / u_std_t[None, :]
    x_mean = y_scaled.mean(axis=0)
    x_std = y_scaled.std(axis=0) + 1e-06
    x_mean_t = torch.tensor(x_mean, dtype=torch.float32, device=device)
    x_std_t = torch.tensor(x_std, dtype=torch.float32, device=device)

    def whiten_x(X):
        return (X - x_mean_t[None, :]) / x_std_t[None, :]
    y_scaled_t = torch.tensor(y_scaled, dtype=torch.float32, device=device)
    y_delayed_data = build_delay_data_tensor(y_scaled, dim_delay, tau_delay, device)
    net = build_net(d, width=100).to(device)
    opt1 = optim.Adam(net.parameters(), lr=lr1)
    opt2 = optim.Adam(net.parameters(), lr=lr2)
    lambda_state = 1
    lambda_delay = 1
    lambda_def = 1

    def time_delay_multich(Y, tau_delay=1, dim_delay=5):
        Y = np.asarray(Y, dtype=np.float32)
        (T, d) = Y.shape
        H = (dim_delay - 1) * tau_delay
        D = np.zeros((T - H, d * dim_delay), dtype=np.float32)
        for t in range(T - H):
            blocks = []
            for i in range(dim_delay):
                blocks.append(Y[t + H - i * tau_delay])
            D[t] = np.concatenate(blocks, axis=0)
        return (D, H)
    (delay_targets, min_t) = time_delay_multich(y_scaled, tau_delay=tau_delay, dim_delay=dim_delay)
    delay_mean = delay_targets.mean(axis=0)
    delay_std = delay_targets.std(axis=0) + 1e-06
    delay_mean_t = torch.tensor(delay_mean, dtype=torch.float32, device=device)
    delay_std_t = torch.tensor(delay_std, dtype=torch.float32, device=device)

    def whiten_delay(D):
        return (D - delay_mean_t[None, :]) / delay_std_t[None, :]
    w_push = 1.0
    w_dim = 1.0
    w_mse = 0.0

    def compute_L1(loss_sink):
        Tmax_state = T - max(tau_pair, (dim_delay - 1) * tau_delay)
        idx_s = np.random.randint(0, Tmax_state, size=stage1_batch)
        x0_s = torch.tensor(y_scaled[idx_s], dtype=torch.float32, device=device)
        yT_s = torch.tensor(y_scaled[idx_s + tau_pair], dtype=torch.float32, device=device)
        x = x0_s.clone()
        n_micro = int(tau_pair * sub)
        for _ in range(n_micro):
            x = x + float(dt_base) * net(x)
        xT_m = x
        x0_s_w = whiten_x(x0_s)
        xT_m_w = whiten_x(xT_m)
        yT_s_w = whiten_x(yT_s)
        pair_m_state = torch.cat([x0_s_w, xT_m_w], dim=1)
        pair_d_state = torch.cat([x0_s_w, yT_s_w], dim=1)
        L_state = loss_sink(pair_m_state, pair_d_state)
        L_mse = torch.mean((xT_m - yT_s) ** 2)
        y_delay_batch = torch.tensor(delay_targets[idx_s], dtype=torch.float32, device=device)
        x0_d = torch.tensor(y_scaled[idx_s], dtype=torch.float32, device=device)
        x_d = x0_d.clone()
        sols = []
        n_dim_steps = tau_delay * dim_delay * sub
        for _ in range(n_dim_steps):
            sols.append(x_d)
            x_d = x_d + float(dt_base) * net(x_d)
        sol_d = torch.stack(sols, dim=0)
        step_fine = tau_delay * sub
        delay_blocks = torch.flip(sol_d[::step_fine], dims=[0])
        DIM_model_batch = delay_blocks.permute(1, 0, 2).contiguous().view(stage1_batch, dim_delay * d)
        L_delay_loss = loss_sink(whiten_delay(DIM_model_batch), whiten_delay(y_delay_batch))
        L1 = w_push * L_state + w_dim * L_delay_loss + w_mse * L_mse
        return L1

    def build_induced_block(k0, L_hits, loss_sink):
        hit_times = hits[k0:k0 + L_hits].astype(np.int64)
        X_blk_d = torch.tensor(XU_data[k0:k0 + L_hits], dtype=torch.float32, device=device)
        D_blk_d = torch.tensor(DU_data[k0:k0 + L_hits], dtype=torch.float32, device=device)
        t0 = int(hit_times[0])
        t_start = t0 - int((dim_delay - 1) * tau_delay)
        t_end = int(hit_times[-1])
        total_obs_steps = int(t_end - t_start)
        x = torch.tensor(y_scaled[t_start], dtype=torch.float32, device=device)
        hit_map = {int(t): j for (j, t) in enumerate(hit_times.tolist())}
        x_hits_m = [None] * L_hits
        x_hist = [x.clone()]
        for n in range(total_obs_steps + 1):
            t_abs = t_start + n
            if t_abs in hit_map:
                x_hits_m[hit_map[t_abs]] = x
                x = x.detach()
            if n == total_obs_steps:
                break
            for _ in range(sub):
                x = x + float(dt_base) * net(x)
            x_hist.append(x.clone())
        X_blk_m = torch.stack(x_hits_m, dim=0)
        Dm_list = []
        for t_abs in hit_times:
            idx0 = int(t_abs - t_start)
            lag_states = []
            for i in range(dim_delay):
                lag_states.append(x_hist[idx0 - i * tau_delay])
            lag_states = torch.stack(lag_states, dim=0)
            Dm_list.append(lag_states.reshape(-1))
        D_blk_m = torch.stack(Dm_list, dim=0)
        U_blk_m = torch.cat([X_blk_m, D_blk_m], dim=1)
        U_blk_d = torch.cat([X_blk_d, D_blk_d], dim=1)
        Um = whiten_u(U_blk_m)
        Ud = whiten_u(U_blk_d)
        (xm, dm) = (Um[:, :d], Um[:, d:])
        (xd, dd) = (Ud[:, :d], Ud[:, d:])
        u_m = torch.cat([alpha_pair_state * xm, alpha_pair_delay * dm], dim=1)
        u_d = torch.cat([alpha_pair_state * xd, alpha_pair_delay * dd], dim=1)
        (u0_m, u1_m) = (u_m[:-1], u_m[1:])
        (u0_d, u1_d) = (u_d[:-1], u_d[1:])
        pair_m = torch.cat([u0_m, u1_m], dim=1)
        pair_d = torch.cat([u0_d, u1_d], dim=1)
        L2_graph = loss_sink(pair_m, pair_d)
        L2_marg0 = loss_sink(u0_m, u0_d)
        L2_marg1 = loss_sink(u1_m, u1_d)
        L2_marg = w_marg_u0 * L2_marg0 + w_marg_u1 * L2_marg1
        L2_raw = L2_graph + L2_marg
        L2_norm = L2_raw / float(L_hits)
        return (L2_norm, L2_raw)
    best_total = float('inf')
    best_total_it = -1
    best_total_sd = None
    best_total_L1 = None
    best_total_L2raw = None
    best_total_L2norm = None
    best_L2raw = float('inf')
    best_L2_it = -1
    best_L2_sd = None
    best_L2_at_best = None
    best_L1_at_best = None
    best_L2norm_at_best = None
    (loss_tot_hist, loss_L1_hist, loss_L2norm_hist, loss_L2raw_hist, stage_hist) = ([], [], [], [], [])
    print('============================================================')
    print('Start training (TRAIN split only)')
    print(f'dt_obs={dt_obs} | dt_base={dt_base} | sub={sub}')
    print(f'Nsteps={Nsteps} | warm={warm_steps} | N_hits={len(hits)} | L_hits={L_hits_fullbatch}')
    print(f'Stage2 objective: L_total = L1 + w2_const * L2_norm   |  w2_const={w2_const}')
    print('============================================================')
    net.train()
    for step in range(Nsteps):
        t = step / max(1, Nsteps - 1)
        blur = blur_start + (blur_end - blur_start) * t
        loss_sink = make_sinkhorn(blur)
        if step < warm_steps:
            stage = 'stage1'
        else:
            cyc = stage2_steps_per_cycle + stage1_steps_per_cycle
            k = (step - warm_steps) % cyc
            stage = 'stage2' if k < stage2_steps_per_cycle else 'stage1'
        opt = opt1 if stage == 'stage1' else opt2
        opt.zero_grad(set_to_none=True)
        L1 = compute_L1(loss_sink)
        if stage == 'stage1':
            L2_norm = torch.tensor(0.0, device=device)
            L2_raw = torch.tensor(0.0, device=device)
            Ltot = L1
        else:
            L_hits = min(L_hits_fullbatch, len(hits))
            k0max = len(hits) - L_hits
            l2n_list = []
            l2r_list = []
            for _ in range(BATCH_SIZE_STAGE2):
                k0 = np.random.randint(0, k0max + 1)
                (l2n, l2r) = build_induced_block(k0, L_hits, loss_sink)
                l2n_list.append(l2n)
                l2r_list.append(l2r)
            L2_norm = torch.stack(l2n_list).mean()
            L2_raw = torch.stack(l2r_list).mean()
            Ltot = L1 + float(w2_const) * L2_norm
        Ltot.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 500.0)
        opt.step()
        ltot_v = float(Ltot.detach().cpu())
        l1_v = float(L1.detach().cpu())
        l2n_v = float(L2_norm.detach().cpu())
        l2r_v = float(L2_raw.detach().cpu())
        loss_tot_hist.append(ltot_v)
        loss_L1_hist.append(l1_v)
        loss_L2norm_hist.append(l2n_v)
        loss_L2raw_hist.append(l2r_v)
        stage_hist.append(stage)
        if stage == 'stage2':
            if ltot_v < best_total:
                best_total = ltot_v
                best_total_it = step
                best_total_sd = cpu_state_dict(net)
                best_total_L1 = l1_v
                best_total_L2norm = l2n_v
                best_total_L2raw = l2r_v
                print(f'  --> [New BEST_TOTAL] Iter {step} | L_total={best_total:.6f} | L1={best_total_L1:.6f} | L2_norm={best_total_L2norm:.6f} | L2_raw={best_total_L2raw:.6f}')
            if l2r_v < best_L2raw:
                best_L2raw = l2r_v
                best_L2_it = step
                best_L2_sd = cpu_state_dict(net)
                best_L2_at_best = ltot_v
                best_L1_at_best = l1_v
                best_L2norm_at_best = l2n_v
                print(f'  --> [New BEST_L2]    Iter {step} | L2_raw={best_L2raw:.6f} | L_total={best_L2_at_best:.6f} | L1={best_L1_at_best:.6f} | L2_norm={best_L2norm_at_best:.6f}')
        if step % 100 == 0:
            print(f'Iter {step:5d} [{stage}] | Ltot={ltot_v:.6f} | L1={l1_v:.6f} | L2_norm={l2n_v:.6f} | L2_raw={l2r_v:.6f} | blur={blur:.3f}')
    print('Training done.')
    stamp = ''
    path_final = os.path.join(OUT_DIR, f'{SAVE_PREFIX}_FINAL.p')
    path_best = os.path.join(OUT_DIR, f'{SAVE_PREFIX}_BEST.p')
    path_best_l2 = os.path.join(OUT_DIR, f'{SAVE_PREFIX}_BEST_L2.p')
    config = dict(seed=int(seed), dt_obs=float(dt_obs), dt_base=float(dt_base), sub=int(sub), tau_pair=int(tau_pair), tau_delay=int(tau_delay), dim_delay=int(dim_delay), L_sec=int(L_sec), M_landmarks=int(M_landmarks), cand_frac=float(cand_frac), q_speed=float(q_speed), eps_in_mult=float(eps_in_mult), eps_out_mult=float(eps_out_mult), L_hits_fullbatch=int(L_hits_fullbatch), BATCH_SIZE_STAGE2=int(BATCH_SIZE_STAGE2), lr1=float(lr1), lr2=float(lr2), w2_const=float(w2_const), training_source='KS_test.ipynb cell 19; current notebook profile', alpha_pair_state=float(alpha_pair_state), alpha_pair_delay=float(alpha_pair_delay), w_marg_u0=float(w_marg_u0), w_marg_u1=float(w_marg_u1), w_glue=float(w_glue), blur_start=float(blur_start), blur_end=float(blur_end), warm_steps=int(warm_steps), Nsteps=int(Nsteps), min_t=int(min_t), eps_in=float(eps_in), eps_out=float(eps_out), nn_med=float(nn_med), N_hits=int(len(hits)))
    base_payload = dict(config=config, model=dict(transformer=scaler, state_dict=None), stats=dict(hits=hits, taus_steps=taus, visit_lens=visit_lens, C_dw=C_dw, u_mean_c=u_mean_c, u_std_c=u_std_c, u_mean=u_mean, u_std=u_std), train_hist=dict(stage=np.array(stage_hist, dtype=object), L_total=np.array(loss_tot_hist, dtype=np.float64), L1=np.array(loss_L1_hist, dtype=np.float64), L2_norm=np.array(loss_L2norm_hist, dtype=np.float64), L2_raw=np.array(loss_L2raw_hist, dtype=np.float64)))
    payload_final = dict(base_payload)
    payload_final['model'] = dict(transformer=scaler, state_dict=cpu_state_dict(net))
    with open(path_final, 'wb') as f:
        pickle.dump(payload_final, f)
    print('Saved FINAL:', path_final)
    if best_total_sd is not None:
        payload_best = dict(base_payload)
        payload_best['config'] = dict(config)
        payload_best['config'].update(dict(best_total_iter=int(best_total_it), best_total=float(best_total), best_total_L1=float(best_total_L1), best_total_L2_norm=float(best_total_L2norm), best_total_L2_raw=float(best_total_L2raw)))
        payload_best['model'] = dict(transformer=scaler, state_dict=best_total_sd)
        with open(path_best, 'wb') as f:
            pickle.dump(payload_best, f)
        print(f'Saved BEST_TOTAL: {path_best}  (L_total={best_total:.6f} @ iter {best_total_it})')
    else:
        print('No BEST_TOTAL saved (maybe never entered stage2).')
    if best_L2_sd is not None:
        payload_best_l2 = dict(base_payload)
        payload_best_l2['config'] = dict(config)
        payload_best_l2['config'].update(dict(best_l2_iter=int(best_L2_it), best_l2_L2_raw=float(best_L2raw), best_l2_L_total=float(best_L2_at_best), best_l2_L1=float(best_L1_at_best), best_l2_L2_norm=float(best_L2norm_at_best)))
        payload_best_l2['model'] = dict(transformer=scaler, state_dict=best_L2_sd)
        with open(path_best_l2, 'wb') as f:
            pickle.dump(payload_best_l2, f)
        print(f'Saved BEST_L2: {path_best_l2}  (L2_raw={best_L2raw:.6f} @ iter {best_L2_it})')
    else:
        print('No BEST_L2 saved (maybe never entered stage2).')
    return {'best_loss': best_total if best_total_sd is not None else None, 'best_iteration': best_total_it, 'steps': Nsteps, 'warmup_steps': warm_steps, 'n_hits': len(hits)}
