# Extracted notebook code; CLI/path/device/display adaptations only.
from _notebook_bootstrap import prepare
globals().update(prepare(__file__))
# ============================================================
# ORDINARY-START SEQUENTIAL EMPIRICAL-K CHAIN CONTROL
#
# Paste this entire cell into Jupyter Notebook and run it after
# train_data has been loaded.
#
# This is a full standalone modification of the original PRIME
# training cell. It still runs Algorithm 2 to obtain the empirical
# consecutive inter-event intervals K_i, but Stage 2 no longer uses
# Algorithm-2 hit locations as source/target times.
# ============================================================

import os
import copy
import pickle
import random
import datetime

import numpy as np
import torch
import torch.nn as tnn
from torch import optim
import matplotlib.pyplot as plt

from geomloss import SamplesLoss
from sklearn.preprocessing import MaxAbsScaler

if "train_data" not in globals():
    raise RuntimeError(
        "train_data is not defined. Run the data-loading cell first, "
        "then paste and run this entire cell."
    )

dt_obs  = 0.1 # Observation 
dt_base = 0.1 # Intergration step
sub = int(round(dt_obs/dt_base))

seed = PORT_ARGS.seed if PORT_ARGS.seed is not None else 123
cand_frac   = 0.5
q_speed     = 0.3
M_landmarks = 200
L_sec       = 5
tau_delay   = 1    
dim_delay   = 5       
eps_in_mult = 0.35
eps_out_mult= 1.8

# Training Config
Nsteps = PORT_ARGS.steps if PORT_ARGS.steps is not None else 5000
warm_steps = PORT_ARGS.warm_steps if PORT_ARGS.warm_steps is not None else 3000

tau_pair    = 2       
stage1_batch= 512     
lr1         = 1e-3

L_hits_fullbatch     = 32
BATCH_SIZE_STAGE2    = 16
lr2                  = 1e-3

alpha_pair_state = 1.0
alpha_pair_delay = 0.7
w_marg_u0 = 0.20
w_marg_u1 = 0.20
w_defect  = 0

blur_start = 0.2
blur_end   = 0.1

stage2_steps_per_cycle = 1
stage1_steps_per_cycle = 3

w2_const = 0.5

# Save Config
OUT_DIR = str(PORT_OUTPUT / f"OrdinaryStartSequentialEmpiricalK_seed{seed}")
SAVE_PREFIX = "OrdinaryStartSequentialEmpiricalK"
os.makedirs(OUT_DIR, exist_ok=True)

device = PORT_DEVICE
print("device:", device)

# -------------------------
# Utils
# -------------------------
def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def make_sinkhorn(blur):
    return SamplesLoss(loss="sinkhorn", p=2, blur=float(blur), backend="tensorized")

def build_net(d, width=100):
    return tnn.Sequential(
        tnn.Linear(d, width), tnn.Tanh(),
        tnn.Linear(width, width), tnn.Tanh(),
        tnn.Linear(width, width), tnn.Tanh(),
        tnn.Linear(width, d),
    )

def cpu_state_dict(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

DELAY_CH = 0  

def build_delay_full_onech(y_scaled, tau, dim, ch):
    """
    y_scaled: (T, d)
    returns:
      D_full: (T, dim), with lag blocks [y_ch(t), y_ch(t-tau), ..., y_ch(t-(dim-1)tau)]
      min_t
    """
    T, d = y_scaled.shape
    min_t = (dim - 1) * tau
    D = np.zeros((T, dim), dtype=np.float64)

    s = y_scaled[:, ch]   # 只取一个 channel

    for t in range(min_t, T):
        D[t] = np.array([s[t - i * tau] for i in range(dim)], dtype=np.float64)

    return D, min_t

def farthest_point_sampling(X, M, start=0):
    N = int(X.shape[0])
    M = int(min(int(M), N))
    idx = np.zeros((M,), dtype=np.int64)
    idx[0] = int(start)
    dist2 = np.sum((X - X[idx[0]])**2, axis=1)
    for m in range(1, M):
        j = int(np.argmax(dist2))
        idx[m] = j
        d2_new = np.sum((X - X[j])**2, axis=1)
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
    last = -10**12
    active = False
    active_i = -1
    enter_t = -1

    for t in range(T):
        d2 = np.sum((C_dw - U_dw[t])**2, axis=1)
        i = int(np.argmin(d2))
        d = float(np.sqrt(d2[i]))

        if not active:
            if d <= eps_in and (t - last) >= int(L_sec):
                hits.append(t)
                last = t
                active = True
                active_i = i
                enter_t = t
        else:
            d_active = float(np.sqrt(np.sum((U_dw[t] - C_dw[active_i])**2)))
            if d_active >= eps_out:
                active = False
                active_i = -1
                if enter_t >= 0:
                    visit_lens.append(t - enter_t)
                enter_t = -1

    return np.array(hits, dtype=np.int64), np.array(visit_lens, dtype=np.int64)

def pstats(a):
    a = np.asarray(a)
    if len(a) == 0:
        return dict(median=np.nan, mean=np.nan, p10=np.nan, p90=np.nan)
    return dict(median=float(np.median(a)), mean=float(np.mean(a)),
                p10=float(np.quantile(a,0.1)), p90=float(np.quantile(a,0.9)))

# ========================================================
# NEW: helpers for DIM-style Stage 1 (IM + delay-IM)
# ========================================================
def build_delay_data_tensor_onech(y_np, dim_delay, tau_delay, ch, device):
    """
    只对一个 channel 做 delay embedding
    return: (1, L, dim_delay)
    """
    Y = torch.tensor(y_np, dtype=torch.float32, device=device)   # (T, d)
    s = Y[:, ch]                                                 # (T,)
    T = s.shape[0]

    L = T - tau_delay * (dim_delay - 1)
    delayed = torch.zeros((1, L, dim_delay), dtype=torch.float32, device=device)

    for i in range(dim_delay):
        start = dim_delay * tau_delay - (i + 1) * tau_delay
        delayed[0, :, i] = s[start:start + L]

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
    sol = torch.stack(sols, dim=0)  # (n_fine_steps, T, d)
    return sol

def build_delay_cloud_from_sol(sol, tau_delay, dim_delay, sub):
    """
    Build delay cloud DIM_model from stacked sol:
      sol: (Kfine, T, d)
      return: DIM_model: (d, L, dim_delay)
    """
    Kfine, T, d = sol.shape
    K_needed = tau_delay * dim_delay * sub
    if K_needed > Kfine:
        raise ValueError(f"Kfine={Kfine} < K_needed={K_needed}, increase n_fine_steps.")
    sol_use = sol[:K_needed]                 # (K_needed, T, d)

    L = T - tau_delay * (dim_delay - 1)
    sol_use = sol_use[:, :L, :]             # (K_needed, L, d)

    # (d, L, K_needed)
    sol_perm = sol_use.permute(2, 1, 0)

    step = tau_delay * sub
    DIM_model = torch.flip(sol_perm[:, :, ::step], dims=[2])  # (d, L, dim_delay)
    return DIM_model

# -------------------------
# Build implicit hits on TRAIN
# -------------------------
set_seed(seed)

# 这里假定外部已经有 train: np.ndarray, shape (T, d)
train = np.asarray(train_data)
assert train.ndim == 2
T, d = train.shape

scaler = MaxAbsScaler().fit(train)
y_scaled = scaler.transform(train).astype(np.float64)

# for Stage-2 hits:
D_full, min_t = build_delay_full_onech(
    y_scaled, tau=tau_delay, dim=dim_delay, ch=DELAY_CH
)
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
u_std_c  = Ucand.std(axis=0) + 1e-6
Ucand_dw = (Ucand - u_mean_c[None,:]) / u_std_c[None,:]
Ufull_dw = (U_full - u_mean_c[None,:]) / u_std_c[None,:]

start0 = int(np.random.randint(0, len(idx_cand)))
fps_idx_local = farthest_point_sampling(Ucand_dw, M=M_landmarks, start=start0)
C_dw = Ucand_dw[fps_idx_local]

nn = []
for i in range(len(C_dw)):
    d2 = np.sum((C_dw - C_dw[i])**2, axis=1)
    d2[i] = np.inf
    nn.append(np.sqrt(np.min(d2)))
nn_med = float(np.median(nn))

eps_in  = eps_in_mult * nn_med
eps_out = eps_out_mult * eps_in

hits_raw, visit_lens = extract_hits_from_landmarks(Ufull_dw, C_dw, eps_in, eps_out, L_sec=L_sec)
hits = hits_raw[hits_raw >= min_t]
hits = np.sort(hits)

taus = np.diff(hits).astype(np.int64)  # empirical K_i in observation steps

if len(hits) < 2:
    raise RuntimeError(
        "Algorithm 2 produced fewer than two hits; "
        "the empirical-K chain cannot be constructed."
    )
if np.any(taus <= 0):
    raise RuntimeError("All empirical K_i values must be positive.")

# ---------------------------------------------------------
# Ordinary-start sequential empirical-K control preparation
# ---------------------------------------------------------
#
# For a contiguous K block
#
#     K_k0, ..., K_{k0+L-2},
#
# sample an ordinary legal source s0 independently of k0 and define
#
#     s_j = s0 + sum_{r<j} K_{k0+r}.
#
# This retains:
#   - the same consecutive empirical K sequence,
#   - the same continuous model chain,
#   - the same graph/marginal losses,
# while deleting Algorithm-2 recurrence-conditioned event locations.

history_len = int((dim_delay - 1) * tau_delay)
L_chain_control = min(int(L_hits_fullbatch), int(len(hits)))
K_edges_control = L_chain_control - 1

if K_edges_control < 1:
    raise RuntimeError("At least two chain atoms are required.")

n_K_blocks_control = len(taus) - K_edges_control + 1
if n_K_blocks_control < 1:
    raise RuntimeError(
        f"Not enough K values for L_chain={L_chain_control}: "
        f"len(taus)={len(taus)}"
    )

K_block_spans_control = np.array(
    [
        int(np.sum(taus[k0:k0 + K_edges_control]))
        for k0 in range(n_K_blocks_control)
    ],
    dtype=np.int64,
)
max_K_block_span_control = int(np.max(K_block_spans_control))

# Common legal pool for every K block. Therefore s0 and the selected
# K block are sampled independently.
ordinary_start_pool = np.arange(
    history_len,
    T - max_K_block_span_control,
    dtype=np.int64,
)

if len(ordinary_start_pool) == 0:
    raise RuntimeError(
        "No legal ordinary start exists for the longest empirical-K block. "
        f"T={T}, history_len={history_len}, "
        f"max_span={max_K_block_span_control}"
    )

print("------------------------------------------------------------")
print("Ordinary-start sequential empirical-K control")
print(f"N_hits={len(hits)} | N_K={len(taus)}")
print(
    f"L_chain={L_chain_control} | K_edges={K_edges_control} | "
    f"N_contiguous_K_blocks={n_K_blocks_control}"
)
print(
    f"max_K_block_span={max_K_block_span_control} | "
    f"N_common_ordinary_starts={len(ordinary_start_pool)}"
)
print("------------------------------------------------------------")

# data blocks at hits
XU_data = y_scaled[hits]
DU_data = D_full[hits]
U_hits_data = np.concatenate([XU_data, DU_data], axis=1)

# 全局的 state 变量 Tensor 版（用于 Stage1-DIM 的 teacher forcing）
Y_all_t = torch.tensor(y_scaled, dtype=torch.float32, device=device)  # (T, d)

# whitening for stage-2 (U-space)
u_mean = U_hits_data.mean(axis=0)
u_std  = U_hits_data.std(axis=0) + 1e-6
u_mean_t = torch.tensor(u_mean, dtype=torch.float32, device=device)
u_std_t  = torch.tensor(u_std,  dtype=torch.float32, device=device)

def whiten_u(U):
    return (U - u_mean_t[None,:]) / u_std_t[None,:]

# whitening for Stage-1 (state-only)
x_mean = y_scaled.mean(axis=0)
x_std  = y_scaled.std(axis=0) + 1e-6
x_mean_t = torch.tensor(x_mean, dtype=torch.float32, device=device)
x_std_t  = torch.tensor(x_std,  dtype=torch.float32, device=device)

def whiten_x(X):
    return (X - x_mean_t[None,:]) / x_std_t[None,:]

# tensors for Stage-1 DIM-style loss
y_scaled_t = torch.tensor(y_scaled, dtype=torch.float32, device=device)
y_delayed_data = build_delay_data_tensor_onech(
    y_scaled, dim_delay, tau_delay, DELAY_CH, device
)   # shape = (1, L, dim_delay)

# -------------------------
# Build model + optimizers
# -------------------------
net = build_net(d, width=100).to(device)
opt1 = optim.Adam(net.parameters(), lr=lr1)
opt2 = optim.Adam(net.parameters(), lr=lr2)

# =========================================================
# Stage-1 L1 (DIM-style: IM(state) + IM(delay) + small glue)
# =========================================================
lambda_state = 1
lambda_delay = 1
lambda_def   = 1

def compute_L1(loss_sink):
    # ---------- 1) IM: state space graph measure ----------
    Tmax = len(y_scaled) - tau_pair - 1
    idx = np.random.randint(0, Tmax, size=stage1_batch)

    # teacher forcing pair: x0 -> xT_data / xT_model
    x0 = torch.tensor(y_scaled[idx], dtype=torch.float32, device=device)              # (B, d)
    xT_d = torch.tensor(y_scaled[idx + tau_pair], dtype=torch.float32, device=device) # (B, d)

    # 在 fine grid 上走 tau_pair * sub 步
    x = x0.clone()
    n_micro = int(tau_pair * sub)
    for _ in range(n_micro):
        x = x + float(dt_base) * net(x)
    xT_m = x  # (B, d)

    # whiten + 拼接 graph pairs
    x0w = whiten_x(x0)
    xTd = whiten_x(xT_d)
    xTm = whiten_x(xT_m)

    pair_d_state = torch.cat([x0w, xTd], dim=1)   # (B, 2d)
    pair_m_state = torch.cat([x0w, xTm], dim=1)   # (B, 2d)

    # OT on state graph (IM)
    L_state = loss_sink(pair_m_state, pair_d_state)

    # ---------- 2) DIM: delay space graph measure ----------
    # 完全照 Cylinder-DIM：对整条 train 轨迹做 teacher forcing，
    # 生成 Kfine = tau_delay * dim_delay * sub 个 fine step，
    # 然后在 fine 维度上每隔 tau_delay*sub 取一个点构成 delay 向量。

    Kfine = tau_delay * dim_delay * sub  # = 20 * 5 * 1 = 100
    yy = Y_all_t.clone()                 # (T, d)，整条 train 轨迹
    sols = []

    for j in range(Kfine):
        sols.append(yy)                  # 每个 fine step 存一整条 (T,d) 的状态
        for _ in range(sub):             # 这里 sub=1，但写成循环以兼容一般情况
            yy = yy + float(dt_base) * net(yy)

    sol = torch.stack(sols, dim=0)       # (Kfine, T, d)

    # 只取从 min_t 开始的那段，与 D_full / y_delayed_data 对齐
    sol = sol[:, min_t:, :]              # (Kfine, L_delay, d) 其中 L_delay = T - min_t

    # 现在维度是: (Kfine, L_delay, d)
    # 我们想要 (d, L_delay, dim_delay)
    #   - 先 permute 成 (d, L_delay, Kfine)
    #   - 然后在最后一个维度上，每隔 tau_delay*sub 抽样 dim_delay 个点，并 flip 掉顺序
    step_fine = tau_delay * sub

    # sol: (Kfine, L_delay, d)
    sol_ch = sol[:, :, DELAY_CH]                    # (Kfine, L_delay)
    sol_ch = sol_ch.permute(1, 0).contiguous()     # (L_delay, Kfine)

    DIM_model = torch.flip(sol_ch[:, ::step_fine], dims=[1])   # (L_delay, dim_delay)
    DIM_model = DIM_model.unsqueeze(0).contiguous()            # (1, L_delay, dim_delay)
    y_delay = y_delayed_data                 # (d, L_delay, dim_delay)
    L_delay = loss_sink(DIM_model, y_delay).sum()
    # Align with the initial slice
    glue_defect = torch.mean((DIM_model[:, 0, :-1] - y_delay[:, 0, :-1]) ** 2)
    L1 = L_state + L_delay + glue_defect
    return L1

# =========================================================
# Stage 2: ordinary-start sequential empirical-K chain
# =========================================================
def build_ordinary_sequential_empiricalK_block(k0, s0, L_hits, loss_sink):
    """
    One control block.

    1) Take the contiguous empirical Algorithm-2 interval sequence

           K_seq = taus[k0 : k0 + L_hits - 1].

    2) Start at an ordinary legal time s0, sampled independently of k0.

    3) Define the ordinary chain times

           s_j = s0 + cumulative_sum(K_seq).

    4) Build data atoms at these ordinary times.

    5) Roll the model continuously from s0-history_len through the
       complete chain. At each chain boundary, detach gradients only;
       never reset the numerical model state to the data.

    Thus the only intended removal relative to PRIME Stage 2 is the
    Algorithm-2 recurrence-conditioned event location.
    """
    k0 = int(k0)
    s0 = int(s0)
    L_hits = int(L_hits)

    K_seq = taus[k0:k0 + L_hits - 1].astype(np.int64)
    if len(K_seq) != L_hits - 1:
        raise AssertionError("Incorrect empirical-K block length.")

    offsets = np.concatenate(
        [
            np.zeros(1, dtype=np.int64),
            np.cumsum(K_seq, dtype=np.int64),
        ]
    )
    chain_times = s0 + offsets

    if len(chain_times) != L_hits:
        raise AssertionError("Incorrect number of ordinary chain times.")
    if not np.array_equal(np.diff(chain_times), K_seq):
        raise AssertionError("The ordinary chain does not preserve K_seq.")
    if chain_times[0] < history_len:
        raise AssertionError("Ordinary start lacks delay history.")
    if chain_times[-1] >= T:
        raise AssertionError("Ordinary chain exceeds train trajectory.")

    # Data atoms are taken at ordinary chain times, not Algorithm-2 hits.
    X_blk_d = torch.tensor(
        y_scaled[chain_times],
        dtype=torch.float32,
        device=device,
    )
    D_blk_d = torch.tensor(
        D_full[chain_times],
        dtype=torch.float32,
        device=device,
    )

    t_start = int(chain_times[0] - history_len)
    t_end = int(chain_times[-1])
    total_obs_steps = int(t_end - t_start)

    x = torch.tensor(
        y_scaled[t_start],
        dtype=torch.float32,
        device=device,
    )

    chain_map = {
        int(t_abs): j
        for j, t_abs in enumerate(chain_times.tolist())
    }
    x_chain_m = [None] * L_hits

    # x_hist[n] = model state at observation time t_start+n.
    x_hist = [x.clone()]

    for n in range(total_obs_steps + 1):
        t_abs = t_start + n

        if t_abs in chain_map:
            j = chain_map[t_abs]
            x_chain_m[j] = x

            # PRIME-style multi-shoot cut:
            # numerical state continues; only the gradient is cut.
            x = x.detach()

        if n == total_obs_steps:
            break

        # Advance exactly one observation step.
        for _ in range(sub):
            x = x + float(dt_base) * net(x)

        x_hist.append(x.clone())

    if any(v is None for v in x_chain_m):
        raise AssertionError("A model chain atom was not recorded.")

    X_blk_m = torch.stack(x_chain_m, dim=0)

    # Model-generated delay vectors at ordinary chain times.
    Dm_list = []
    for t_abs in chain_times:
        idx0 = int(t_abs - t_start)
        lag_vals = []
        for i in range(dim_delay):
            hist_idx = idx0 - i * tau_delay
            if hist_idx < 0:
                raise AssertionError("Negative delay-history index.")
            lag_vals.append(x_hist[hist_idx][DELAY_CH])
        Dm_list.append(torch.stack(lag_vals, dim=0))

    D_blk_m = torch.stack(Dm_list, dim=0)

    U_blk_m = torch.cat([X_blk_m, D_blk_m], dim=1)
    U_blk_d = torch.cat([X_blk_d, D_blk_d], dim=1)

    # Keep PRIME's original hit-atom whitening to minimize changes.
    Um = whiten_u(U_blk_m)
    Ud = whiten_u(U_blk_d)

    xm, dm = Um[:, :d], Um[:, d:]
    xd, dd = Ud[:, :d], Ud[:, d:]

    u_m = torch.cat(
        [alpha_pair_state * xm, alpha_pair_delay * dm],
        dim=1,
    )
    u_d = torch.cat(
        [alpha_pair_state * xd, alpha_pair_delay * dd],
        dim=1,
    )

    u0_m, u1_m = u_m[:-1], u_m[1:]
    u0_d, u1_d = u_d[:-1], u_d[1:]

    pair_m = torch.cat([u0_m, u1_m], dim=1)
    pair_d = torch.cat([u0_d, u1_d], dim=1)

    L2_graph = loss_sink(pair_m, pair_d)
    L2_marg0 = loss_sink(u0_m, u0_d)
    L2_marg1 = loss_sink(u1_m, u1_d)
    L2_marg = w_marg_u0 * L2_marg0 + w_marg_u1 * L2_marg1

    defect = torch.mean((Um[:, :d] - Ud[:, :d]) ** 2)

    L2_raw = L2_graph + L2_marg + w_defect * defect
    L2_norm = L2_raw / float(L_hits)

    return (
        L2_norm,
        L2_raw,
        L2_graph,
        L2_marg0,
        L2_marg1,
        int(k0),
        int(s0),
        int(chain_times[-1] - chain_times[0]),
    )


# -------------------------
# BEST trackers
# -------------------------
best_total = float("inf")
best_total_it = -1
best_total_sd = None
best_total_L1 = None
best_total_L2raw = None
best_total_L2norm = None

best_L2raw = float("inf")
best_L2_it = -1
best_L2_sd = None
best_L2_at_best = None
best_L1_at_best = None
best_L2norm_at_best = None

# -------------------------
# Training loop
# -------------------------
loss_tot_hist, loss_L1_hist, loss_L2norm_hist, loss_L2raw_hist, stage_hist = [], [], [], [], []
loss_L2graph_hist, loss_L2marg0_hist, loss_L2marg1_hist = [], [], []
sampled_k0_hist, sampled_s0_hist, sampled_span_hist = [], [], []

print("============================================================")
print("Start training (TRAIN split only)")
print("Method: ordinary-start sequential empirical-K chain")
print("Algorithm-2 hit locations used as Stage-2 times: NO")
print("Contiguous empirical K sequence retained: YES")
print("Continuous model chain retained: YES")
print(f"dt_obs={dt_obs} | dt_base={dt_base} | sub={sub}")
print(
    f"Nsteps={Nsteps} | warm={warm_steps} | "
    f"N_hits={len(hits)} | L_chain={L_chain_control}"
)
print(
    f"Stage2 objective: L_total = L1 + w2_const * L2_norm | "
    f"w2_const={w2_const}"
)
print("============================================================")

net.train()
for step in range(Nsteps):
    t = step / max(1, Nsteps - 1)
    blur = blur_start + (blur_end - blur_start) * t
    loss_sink = make_sinkhorn(blur)

    if step < warm_steps:
        stage = "stage1"
    else:
        cyc = stage2_steps_per_cycle + stage1_steps_per_cycle
        k = (step - warm_steps) % cyc
        stage = "stage2" if k < stage2_steps_per_cycle else "stage1"

    opt = opt1 if stage == "stage1" else opt2
    opt.zero_grad(set_to_none=True)

    L1 = compute_L1(loss_sink)

    if stage == "stage1":
        L2_norm = torch.tensor(0.0, device=device)
        L2_raw = torch.tensor(0.0, device=device)
        L2_graph = torch.tensor(0.0, device=device)
        L2_marg0 = torch.tensor(0.0, device=device)
        L2_marg1 = torch.tensor(0.0, device=device)
        Ltot = L1
    else:
        L_hits = L_chain_control
        k0max = n_K_blocks_control - 1

        l2n_list = []
        l2r_list = []
        l2g_list = []
        l2m0_list = []
        l2m1_list = []

        for _ in range(BATCH_SIZE_STAGE2):
            # k0 and s0 are sampled independently.
            k0 = int(np.random.randint(0, k0max + 1))
            s0 = int(np.random.choice(ordinary_start_pool))

            block = build_ordinary_sequential_empiricalK_block(
                k0=k0,
                s0=s0,
                L_hits=L_hits,
                loss_sink=loss_sink,
            )

            l2n_list.append(block[0])
            l2r_list.append(block[1])
            l2g_list.append(block[2])
            l2m0_list.append(block[3])
            l2m1_list.append(block[4])

            sampled_k0_hist.append(block[5])
            sampled_s0_hist.append(block[6])
            sampled_span_hist.append(block[7])

        L2_norm = torch.stack(l2n_list).mean()
        L2_raw = torch.stack(l2r_list).mean()
        L2_graph = torch.stack(l2g_list).mean()
        L2_marg0 = torch.stack(l2m0_list).mean()
        L2_marg1 = torch.stack(l2m1_list).mean()

        Ltot = L1 + float(w2_const) * L2_norm

    Ltot.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()

    ltot_v = float(Ltot.detach().cpu())
    l1_v   = float(L1.detach().cpu())
    l2n_v  = float(L2_norm.detach().cpu())
    l2r_v  = float(L2_raw.detach().cpu())
    l2g_v  = float(L2_graph.detach().cpu())
    l2m0_v = float(L2_marg0.detach().cpu())
    l2m1_v = float(L2_marg1.detach().cpu())

    loss_tot_hist.append(ltot_v)
    loss_L1_hist.append(l1_v)
    loss_L2norm_hist.append(l2n_v)
    loss_L2raw_hist.append(l2r_v)
    loss_L2graph_hist.append(l2g_v)
    loss_L2marg0_hist.append(l2m0_v)
    loss_L2marg1_hist.append(l2m1_v)
    stage_hist.append(stage)

    if stage == "stage2":
        if ltot_v < best_total:
            best_total = ltot_v
            best_total_it = step
            best_total_sd = cpu_state_dict(net)
            best_total_L1 = l1_v
            best_total_L2norm = l2n_v
            best_total_L2raw  = l2r_v
            print(f"  --> [New BEST_TOTAL] Iter {step} | L_total={best_total:.6f} | L1={best_total_L1:.6f} | L2_norm={best_total_L2norm:.6f} | L2_raw={best_total_L2raw:.6f}")

        if l2r_v < best_L2raw:
            best_L2raw = l2r_v
            best_L2_it = step
            best_L2_sd = cpu_state_dict(net)
            best_L2_at_best = ltot_v
            best_L1_at_best = l1_v
            best_L2norm_at_best = l2n_v
            print(f"  --> [New BEST_L2]    Iter {step} | L2_raw={best_L2raw:.6f} | L_total={best_L2_at_best:.6f} | L1={best_L1_at_best:.6f} | L2_norm={best_L2norm_at_best:.6f}")

    if step % 100 == 0:
        print(f"Iter {step:5d} [{stage}] | Ltot={ltot_v:.6f} | L1={l1_v:.6f} | L2_norm={l2n_v:.6f} | L2_raw={l2r_v:.6f} | blur={blur:.3f}")

print("Training done.")

# -------------------------
# Save FINAL / BEST_TOTAL / BEST_L2
# -------------------------
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
path_final   = os.path.join(OUT_DIR, f"{SAVE_PREFIX}_FINAL_{stamp}.p")
path_best    = os.path.join(OUT_DIR, f"{SAVE_PREFIX}_BEST_{stamp}.p")
path_best_l2 = os.path.join(OUT_DIR, f"{SAVE_PREFIX}_BEST_L2_{stamp}.p")

config = dict(
    seed=int(seed),
    method="ordinary_start_sequential_empirical_K_chain",
    uses_algorithm2_hit_locations_as_stage2_times=False,
    uses_contiguous_empirical_K_sequence=True,
    uses_continuous_model_chain=True,
    source_and_K_block_sampled_independently=True,
    stage2_whitening="same_hit_atom_whitening_as_original_PRIME",
    dt_obs=float(dt_obs),
    dt_base=float(dt_base),
    sub=int(sub),
    tau_pair=int(tau_pair),
    tau_delay=int(tau_delay),
    dim_delay=int(dim_delay),
    L_sec=int(L_sec),
    M_landmarks=int(M_landmarks),
    cand_frac=float(cand_frac),
    q_speed=float(q_speed),
    eps_in_mult=float(eps_in_mult),
    eps_out_mult=float(eps_out_mult),
    L_hits_fullbatch=int(L_hits_fullbatch),
    L_chain_control=int(L_chain_control),
    K_edges_control=int(K_edges_control),
    n_K_blocks_control=int(n_K_blocks_control),
    max_K_block_span_control=int(max_K_block_span_control),
    ordinary_start_pool_size=int(len(ordinary_start_pool)),
    BATCH_SIZE_STAGE2=int(BATCH_SIZE_STAGE2),
    lr1=float(lr1), lr2=float(lr2),
    w2_const=float(w2_const),
    alpha_pair_state=float(alpha_pair_state),
    alpha_pair_delay=float(alpha_pair_delay),
    w_marg_u0=float(w_marg_u0),
    w_marg_u1=float(w_marg_u1),
    w_defect=float(w_defect),
    blur_start=float(blur_start),
    blur_end=float(blur_end),
    warm_steps=int(warm_steps),
    Nsteps=int(Nsteps),
    min_t=int(min_t),
    eps_in=float(eps_in),
    eps_out=float(eps_out),
    nn_med=float(nn_med),
    N_hits=int(len(hits)),
)

base_payload = dict(
    config=config,
    model=dict(
        transformer=scaler,
        state_dict=None,
    ),
    stats=dict(
        # hits are retained only to define/provenance the empirical K sequence;
        # they are not Stage-2 source/target locations in this control.
        hits=hits,
        taus_steps=taus,
        empirical_K_steps=taus,
        K_block_spans=K_block_spans_control,
        ordinary_start_pool=ordinary_start_pool,
        sampled_K_block_starts=np.asarray(sampled_k0_hist, dtype=np.int64),
        sampled_ordinary_starts=np.asarray(sampled_s0_hist, dtype=np.int64),
        sampled_chain_spans=np.asarray(sampled_span_hist, dtype=np.int64),
        visit_lens=visit_lens,
        C_dw=C_dw,
        u_mean_c=u_mean_c,
        u_std_c=u_std_c,
        u_mean=u_mean,
        u_std=u_std,
    ),
    train_hist=dict(
        stage=np.array(stage_hist, dtype=object),
        L_total=np.array(loss_tot_hist, dtype=np.float64),
        L1=np.array(loss_L1_hist, dtype=np.float64),
        L2_norm=np.array(loss_L2norm_hist, dtype=np.float64),
        L2_raw=np.array(loss_L2raw_hist, dtype=np.float64),
        L2_graph=np.array(loss_L2graph_hist, dtype=np.float64),
        L2_source_marginal=np.array(loss_L2marg0_hist, dtype=np.float64),
        L2_target_marginal=np.array(loss_L2marg1_hist, dtype=np.float64),
    )
)

payload_final = dict(base_payload)
payload_final["model"] = dict(transformer=scaler, state_dict=cpu_state_dict(net))
with open(path_final, "wb") as f:
    pickle.dump(payload_final, f)
print("Saved FINAL:", path_final)

if best_total_sd is not None:
    payload_best = dict(base_payload)
    payload_best["config"] = dict(config)
    payload_best["config"].update(dict(
        best_total_iter=int(best_total_it),
        best_total=float(best_total),
        best_total_L1=float(best_total_L1),
        best_total_L2_norm=float(best_total_L2norm),
        best_total_L2_raw=float(best_total_L2raw),
    ))
    payload_best["model"] = dict(transformer=scaler, state_dict=best_total_sd)
    with open(path_best, "wb") as f:
        pickle.dump(payload_best, f)
    print(f"Saved BEST_TOTAL: {path_best}  (L_total={best_total:.6f} @ iter {best_total_it})")
else:
    print("No BEST_TOTAL saved (maybe never entered stage2).")

if best_L2_sd is not None:
    payload_best_l2 = dict(base_payload)
    payload_best_l2["config"] = dict(config)
    payload_best_l2["config"].update(dict(
        best_l2_iter=int(best_L2_it),
        best_l2_L2_raw=float(best_L2raw),
        best_l2_L_total=float(best_L2_at_best),
        best_l2_L1=float(best_L1_at_best),
        best_l2_L2_norm=float(best_L2norm_at_best),
    ))
    payload_best_l2["model"] = dict(transformer=scaler, state_dict=best_L2_sd)
    with open(path_best_l2, "wb") as f:
        pickle.dump(payload_best_l2, f)
    print(f"Saved BEST_L2: {path_best_l2}  (L2_raw={best_L2raw:.6f} @ iter {best_L2_it})")
else:
    print("No BEST_L2 saved (maybe never entered stage2).")

# -------------------------
# Plots
# -------------------------
plt.figure(figsize=(10,3), dpi=140)
plt.plot(loss_tot_hist, label="L_total")
plt.plot(loss_L1_hist, label="L1")
plt.legend(); plt.title("Loss"); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,3), dpi=140)
plt.plot(loss_L2norm_hist, label="L2_norm (0 during stage1)")
plt.legend(); plt.title("Stage2 loss (norm)"); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,3), dpi=140)
plt.plot(loss_L2raw_hist, label="L2_raw (0 during stage1)")
plt.legend(); plt.title("Stage2 loss (raw)"); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,3), dpi=140)
plt.plot(loss_L2graph_hist, label="L2_graph")
plt.plot(loss_L2marg0_hist, label="source marginal")
plt.plot(loss_L2marg1_hist, label="target marginal")
plt.legend()
plt.title("Stage2 loss components")
plt.tight_layout()
plt.show()

ordinary_start_sequential_empiricalK_run = dict(
    net=net,
    scaler=scaler,
    config=config,
    paths=dict(
        FINAL=path_final,
        BEST_TOTAL=path_best if best_total_sd is not None else None,
        BEST_L2=path_best_l2 if best_L2_sd is not None else None,
    ),
)

print("Notebook object available as: ordinary_start_sequential_empiricalK_run")

