# Extracted notebook code; CLI/path/device/display adaptations only.
from _notebook_bootstrap import prepare
globals().update(prepare(__file__))
# Fixed-time transition-measure baseline for PRIME
# Paste this entire file into ONE new Jupyter cell.
#
# REQUIRED VARIABLES ALREADY DEFINED IN THE NOTEBOOK:
#   train_data, test_data, dt_obs, dt_base, sub
#
# This cell does NOT construct Algorithm-2 recurrence events.
# Stage 2 uniformly samples ordinary legal times s and matches
# the transition graph measure of (U_s, U_{s + fixed_lag}).

import os
import copy
import pickle
import random
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as tnn
from torch import optim
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from sklearn.preprocessing import MaxAbsScaler


# ============================================================
# 0. CHECK THE VARIABLES FROM YOUR PREVIOUS NOTEBOOK CELL
# ============================================================
_required = ("train_data", "test_data", "dt_obs", "dt_base", "sub")
_missing = [name for name in _required if name not in globals()]
if _missing:
    raise RuntimeError(
        "Run your data-loading cell first. Missing variables: "
        + ", ".join(_missing)
    )

train = np.asarray(train_data, dtype=np.float64)
test = np.asarray(test_data, dtype=np.float64)  # kept only for later evaluation; never used for training

if train.ndim != 2:
    raise ValueError(f"train_data must have shape (T,d), got {train.shape}")
if test.ndim != 2:
    raise ValueError(f"test_data must have shape (T,d), got {test.shape}")
if train.shape[1] != test.shape[1]:
    raise ValueError(
        f"train/test dimensions differ: {train.shape[1]} vs {test.shape[1]}"
    )
if not np.all(np.isfinite(train)):
    raise ValueError("train_data contains NaN or Inf")
if not np.all(np.isfinite(test)):
    raise ValueError("test_data contains NaN or Inf")

_expected_sub = int(round(float(dt_obs) / float(dt_base)))
if int(sub) != _expected_sub:
    raise ValueError(f"sub={sub}, but round(dt_obs/dt_base)={_expected_sub}")
if int(sub) < 1 or not np.isclose(int(sub) * float(dt_base), float(dt_obs)):
    raise ValueError(
        "dt_obs/dt_base must be a positive integer; "
        f"got dt_obs={dt_obs}, dt_base={dt_base}, sub={sub}"
    )


# ============================================================
# 1. EASY-TO-EDIT CONFIGURATION
# ============================================================
seed = PORT_ARGS.seed if PORT_ARGS.seed is not None else 123456
network_width = 100

# Fixed time lag in OBSERVATION steps.
# It is explicitly specified and is NOT obtained from Algorithm 2.
fixed_lag = PORT_ARGS.fixed_lag

tau_delay = 1
dim_delay = 5
DELAY_CH = 0

# Same overall training schedule as the original PRIME notebook.
Nsteps = PORT_ARGS.steps if PORT_ARGS.steps is not None else 5000
warm_steps = PORT_ARGS.warm_steps if PORT_ARGS.warm_steps is not None else 3000

tau_pair = 2
stage1_batch = 512
lr1 = 1e-3

# PRIME uses L_hits=32, i.e. 31 transition edges in each Stage-2 block.
# Here we use 31 independent ordinary-time pairs and retain denominator 32.
L_hits_fullbatch = 32
stage2_pairs_per_block = L_hits_fullbatch - 1
BATCH_SIZE_STAGE2 = 16
lr2 = 1e-3

alpha_pair_state = 1.0
alpha_pair_delay = 0.7
w_marg_u0 = 0.20
w_marg_u1 = 0.20
w_defect = 0.0

blur_start = 0.2
blur_end = 0.1

# Refinement schedule after warm-up: Stage 2, Stage 1, Stage 2, Stage 1, ...
stage2_steps_per_cycle = 1
stage1_steps_per_cycle = 3

w2_const = 0.5

print_every = 100

# Optional fixed-budget checkpoints for later trajectory/equal-budget evaluation.
# 0 disables them; 100 saves every 100 refinement iterations.
snapshot_every_refinement_steps = 0

OUT_DIR = PORT_OUTPUT / f"FixedTime_seed{seed}_K{fixed_lag}"
SAVE_PREFIX = f"FixedTime_K{fixed_lag}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. UTILITIES
# ============================================================
def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def make_sinkhorn(blur):
    return SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=float(blur),
        backend="tensorized",
        debias=True,
    )


def build_net(d, width=100):
    return tnn.Sequential(
        tnn.Linear(d, width), tnn.Tanh(),
        tnn.Linear(width, width), tnn.Tanh(),
        tnn.Linear(width, width), tnn.Tanh(),
        tnn.Linear(width, d),
    )


def cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def build_delay_full_onech(y_scaled, tau, dim, ch):
    """D_t = [y_c(t), y_c(t-tau), ..., y_c(t-(dim-1)tau)]."""
    T_local, d_local = y_scaled.shape
    if not (0 <= ch < d_local):
        raise ValueError(f"DELAY_CH={ch} is invalid for d={d_local}")
    if tau < 1 or dim < 1:
        raise ValueError("tau_delay and dim_delay must be positive")

    min_t_local = (dim - 1) * tau
    if T_local <= min_t_local:
        raise ValueError(
            f"trajectory length {T_local} is too short for min_t={min_t_local}"
        )

    D = np.zeros((T_local, dim), dtype=np.float64)
    series = y_scaled[:, ch]
    for time_index in range(min_t_local, T_local):
        D[time_index] = np.asarray(
            [series[time_index - lag * tau] for lag in range(dim)],
            dtype=np.float64,
        )
    return D, min_t_local


def build_delay_data_tensor_onech(y_np, dim_delay, tau_delay, ch, device):
    """Delay cloud used by the unchanged DIM-style Stage-1 objective."""
    Y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    series = Y[:, ch]
    T_local = int(series.shape[0])
    length = T_local - tau_delay * (dim_delay - 1)

    delayed = torch.zeros(
        (1, length, dim_delay),
        dtype=torch.float32,
        device=device,
    )
    for lag in range(dim_delay):
        start = dim_delay * tau_delay - (lag + 1) * tau_delay
        delayed[0, :, lag] = series[start:start + length]
    return delayed


# ============================================================
# 3. DATA, NORMALIZATION, MODEL
# ============================================================
set_seed(seed)

device = PORT_DEVICE
print("device:", device, flush=True)

T, d = train.shape
if fixed_lag < 1:
    raise ValueError("fixed_lag must be >= 1")

scaler = MaxAbsScaler().fit(train)
y_scaled = scaler.transform(train).astype(np.float64)

D_full, min_t = build_delay_full_onech(
    y_scaled,
    tau=tau_delay,
    dim=dim_delay,
    ch=DELAY_CH,
)
U_full = np.concatenate([y_scaled, D_full], axis=1)
history_len = (dim_delay - 1) * tau_delay

# Ordinary sources are all legal trajectory indices, not recurrence events.
# Need complete delay history at s and an available target at s + fixed_lag.
legal_sources = np.arange(min_t, T - fixed_lag, dtype=np.int64)
if len(legal_sources) < stage2_pairs_per_block:
    raise RuntimeError(
        f"Only {len(legal_sources)} legal ordinary sources, but "
        f"{stage2_pairs_per_block} pairs are requested per Stage-2 block"
    )

# Event-independent normalization for the fixed-time control.
# No recurrence-hit statistics are used.
legal_atom_indices = np.arange(min_t, T, dtype=np.int64)
U_reference = U_full[legal_atom_indices]
u_mean = U_reference.mean(axis=0)
u_std = U_reference.std(axis=0) + 1e-6
u_mean_t = torch.as_tensor(u_mean, dtype=torch.float32, device=device)
u_std_t = torch.as_tensor(u_std, dtype=torch.float32, device=device)

# Same global state normalization as original Stage 1.
x_mean = y_scaled.mean(axis=0)
x_std = y_scaled.std(axis=0) + 1e-6
x_mean_t = torch.as_tensor(x_mean, dtype=torch.float32, device=device)
x_std_t = torch.as_tensor(x_std, dtype=torch.float32, device=device)


def whiten_u(U):
    return (U - u_mean_t[None, :]) / u_std_t[None, :]


def whiten_x(X):
    return (X - x_mean_t[None, :]) / x_std_t[None, :]


Y_all_t = torch.as_tensor(y_scaled, dtype=torch.float32, device=device)
y_delayed_data = build_delay_data_tensor_onech(
    y_scaled,
    dim_delay,
    tau_delay,
    DELAY_CH,
    device,
)

net = build_net(d, width=network_width).to(device)
opt1 = optim.Adam(net.parameters(), lr=lr1)
opt2 = optim.Adam(net.parameters(), lr=lr2)


# ============================================================
# 4. STAGE 1: UNCHANGED DIM OBJECTIVE
# ============================================================
def compute_L1(loss_sink):
    # 1) State-space graph measure
    Tmax = len(y_scaled) - tau_pair - 1
    if Tmax <= 0:
        raise RuntimeError("trajectory is too short for tau_pair")

    idx = np.random.randint(0, Tmax, size=stage1_batch)
    x0 = torch.as_tensor(
        y_scaled[idx],
        dtype=torch.float32,
        device=device,
    )
    xT_data = torch.as_tensor(
        y_scaled[idx + tau_pair],
        dtype=torch.float32,
        device=device,
    )

    x = x0.clone()
    for _ in range(int(tau_pair * sub)):
        x = x + float(dt_base) * net(x)
    xT_model = x

    pair_data_state = torch.cat(
        [whiten_x(x0), whiten_x(xT_data)],
        dim=1,
    )
    pair_model_state = torch.cat(
        [whiten_x(x0), whiten_x(xT_model)],
        dim=1,
    )
    L_state = loss_sink(pair_model_state, pair_data_state)

    # 2) Delay-space invariant-measure term
    Kfine = tau_delay * dim_delay * sub
    yy = Y_all_t.clone()
    solutions = []

    for _ in range(Kfine):
        solutions.append(yy)
        for _ in range(sub):
            yy = yy + float(dt_base) * net(yy)

    sol = torch.stack(solutions, dim=0)
    sol = sol[:, min_t:, :]

    step_fine = tau_delay * sub
    sol_ch = sol[:, :, DELAY_CH].permute(1, 0).contiguous()
    DIM_model = torch.flip(sol_ch[:, ::step_fine], dims=[1])
    DIM_model = DIM_model.unsqueeze(0).contiguous()

    L_delay = loss_sink(DIM_model, y_delayed_data).sum()
    glue_defect = torch.mean(
        (DIM_model[:, 0, :-1] - y_delayed_data[:, 0, :-1]) ** 2
    )

    return L_state + L_delay + glue_defect


# ============================================================
# 5. STAGE 2: ORDINARY-SOURCE FIXED-TIME TRANSITIONS
# ============================================================
def sample_ordinary_sources(pair_count):
    """Uniform legal times s; no Algorithm-2 event filtering."""
    return np.random.choice(
        legal_sources,
        size=pair_count,
        replace=False,
    ).astype(np.int64)


def build_fixed_time_block(pair_count, loss_sink):
    """
    Match the empirical graph measures of:
        data:  (U_s,     U_{s+fixed_lag})
        model: (Uhat_s,  Uhat_{s+fixed_lag})

    s is uniformly sampled from ordinary legal trajectory times.
    """
    sources = sample_ordinary_sources(pair_count)
    targets = sources + fixed_lag
    starts = sources - history_len

    if np.any(starts < 0) or np.any(targets >= T):
        raise AssertionError("illegal fixed-time source/target indices")

    # Data atoms
    U0_data = torch.as_tensor(
        U_full[sources],
        dtype=torch.float32,
        device=device,
    )
    U1_data = torch.as_tensor(
        U_full[targets],
        dtype=torch.float32,
        device=device,
    )

    # Model atoms; all independent pairs are integrated in parallel.
    x = torch.as_tensor(
        y_scaled[starts],
        dtype=torch.float32,
        device=device,
    )
    x_history = [x.clone()]

    # Roll from s-history_len to s, so the model generates its own source delay.
    for _ in range(history_len):
        for _ in range(sub):
            x = x + float(dt_base) * net(x)
        x_history.append(x.clone())

    x_source = x

    # Multi-shoot cut at the source time, analogous to PRIME's event boundary.
    x = x.detach()

    # Fixed-time transition from s to s + fixed_lag.
    for _ in range(fixed_lag):
        for _ in range(sub):
            x = x + float(dt_base) * net(x)
        x_history.append(x.clone())

    source_delay = torch.stack(
        [
            torch.stack(
                [
                    x_history[
                        history_len - lag * tau_delay
                    ][pair_index, DELAY_CH]
                    for lag in range(dim_delay)
                ],
                dim=0,
            )
            for pair_index in range(pair_count)
        ],
        dim=0,
    )

    target_index = history_len + fixed_lag
    x_target = x_history[target_index]
    target_delay = torch.stack(
        [
            torch.stack(
                [
                    x_history[
                        target_index - lag * tau_delay
                    ][pair_index, DELAY_CH]
                    for lag in range(dim_delay)
                ],
                dim=0,
            )
            for pair_index in range(pair_count)
        ],
        dim=0,
    )

    U0_model = torch.cat([x_source, source_delay], dim=1)
    U1_model = torch.cat([x_target, target_delay], dim=1)

    Um0 = whiten_u(U0_model)
    Um1 = whiten_u(U1_model)
    Ud0 = whiten_u(U0_data)
    Ud1 = whiten_u(U1_data)

    um0 = torch.cat(
        [
            alpha_pair_state * Um0[:, :d],
            alpha_pair_delay * Um0[:, d:],
        ],
        dim=1,
    )
    um1 = torch.cat(
        [
            alpha_pair_state * Um1[:, :d],
            alpha_pair_delay * Um1[:, d:],
        ],
        dim=1,
    )
    ud0 = torch.cat(
        [
            alpha_pair_state * Ud0[:, :d],
            alpha_pair_delay * Ud0[:, d:],
        ],
        dim=1,
    )
    ud1 = torch.cat(
        [
            alpha_pair_state * Ud1[:, :d],
            alpha_pair_delay * Ud1[:, d:],
        ],
        dim=1,
    )

    pair_model = torch.cat([um0, um1], dim=1)
    pair_data = torch.cat([ud0, ud1], dim=1)

    L2_graph = loss_sink(pair_model, pair_data)
    L2_marg0 = loss_sink(um0, ud0)
    L2_marg1 = loss_sink(um1, ud1)

    defect = 0.5 * (
        torch.mean((Um0[:, :d] - Ud0[:, :d]) ** 2)
        + torch.mean((Um1[:, :d] - Ud1[:, :d]) ** 2)
    )

    L2_raw = (
        L2_graph
        + w_marg_u0 * L2_marg0
        + w_marg_u1 * L2_marg1
        + w_defect * defect
    )

    # Same scale as PRIME: 31 transition pairs divided by L_hits=32.
    L2_norm = L2_raw / float(L_hits_fullbatch)

    return (
        L2_norm,
        L2_raw,
        L2_graph,
        L2_marg0,
        L2_marg1,
        defect,
    )


# ============================================================
# 6. TRAINING
# ============================================================
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

loss_tot_hist = []
loss_L1_hist = []
loss_L2norm_hist = []
loss_L2raw_hist = []
loss_L2graph_hist = []
loss_L2marg0_hist = []
loss_L2marg1_hist = []
gradient_norm_hist = []
stage_hist = []

print("=" * 60, flush=True)
print("Start training (TRAIN split only)", flush=True)
print(
    f"Method=fixed-time ordinary-source transition measure | "
    f"Algorithm-2 events=NO",
    flush=True,
)
print(
    f"dt_obs={dt_obs} | dt_base={dt_base} | sub={sub}",
    flush=True,
)
print(
    f"Nsteps={Nsteps} | warm={warm_steps} | "
    f"fixed_lag={fixed_lag} | legal_sources={len(legal_sources)} | "
    f"L_pairs={stage2_pairs_per_block}",
    flush=True,
)
print(
    "Stage2 objective: L_total = L1 + w2_const * L2_norm"
    f"   |  w2_const={w2_const}",
    flush=True,
)
print(
    f"Refinement schedule: Stage2:Stage1 = "
    f"{stage2_steps_per_cycle}:{stage1_steps_per_cycle}",
    flush=True,
)
print("=" * 60, flush=True)

net.train()

for step in range(Nsteps):
    fraction = step / max(1, Nsteps - 1)
    blur = blur_start + (blur_end - blur_start) * fraction
    loss_sink = make_sinkhorn(blur)

    if step < warm_steps:
        stage = "stage1"
    else:
        cycle = stage2_steps_per_cycle + stage1_steps_per_cycle
        cycle_position = (step - warm_steps) % cycle
        stage = (
            "stage2"
            if cycle_position < stage2_steps_per_cycle
            else "stage1"
        )

    opt = opt1 if stage == "stage1" else opt2
    opt.zero_grad(set_to_none=True)

    L1 = compute_L1(loss_sink)

    if stage == "stage1":
        L2_norm = torch.zeros((), dtype=torch.float32, device=device)
        L2_raw = torch.zeros((), dtype=torch.float32, device=device)
        L2_graph = torch.zeros((), dtype=torch.float32, device=device)
        L2_marg0 = torch.zeros((), dtype=torch.float32, device=device)
        L2_marg1 = torch.zeros((), dtype=torch.float32, device=device)
        Ltot = L1
    else:
        blocks = [
            build_fixed_time_block(stage2_pairs_per_block, loss_sink)
            for _ in range(BATCH_SIZE_STAGE2)
        ]

        L2_norm = torch.stack([block[0] for block in blocks]).mean()
        L2_raw = torch.stack([block[1] for block in blocks]).mean()
        L2_graph = torch.stack([block[2] for block in blocks]).mean()
        L2_marg0 = torch.stack([block[3] for block in blocks]).mean()
        L2_marg1 = torch.stack([block[4] for block in blocks]).mean()

        Ltot = L1 + float(w2_const) * L2_norm

    # Record the loss and the EXACT corresponding PRE-UPDATE parameters.
    # This avoids the old one-step BEST-checkpoint mismatch.
    ltot_v = float(Ltot.detach().cpu())
    l1_v = float(L1.detach().cpu())
    l2n_v = float(L2_norm.detach().cpu())
    l2r_v = float(L2_raw.detach().cpu())
    l2g_v = float(L2_graph.detach().cpu())
    l2m0_v = float(L2_marg0.detach().cpu())
    l2m1_v = float(L2_marg1.detach().cpu())

    if stage == "stage2":
        if ltot_v < best_total:
            best_total = ltot_v
            best_total_it = step
            best_total_sd = cpu_state_dict(net)
            best_total_L1 = l1_v
            best_total_L2norm = l2n_v
            best_total_L2raw = l2r_v

            print(
                f"  --> [New BEST_TOTAL] Iter {step} | "
                f"L_total={best_total:.6f} | "
                f"L1={best_total_L1:.6f} | "
                f"L2_norm={best_total_L2norm:.6f} | "
                f"L2_raw={best_total_L2raw:.6f}",
                flush=True,
            )

        if l2r_v < best_L2raw:
            best_L2raw = l2r_v
            best_L2_it = step
            best_L2_sd = cpu_state_dict(net)
            best_L2_at_best = ltot_v
            best_L1_at_best = l1_v
            best_L2norm_at_best = l2n_v

            print(
                f"  --> [New BEST_L2]    Iter {step} | "
                f"L2_raw={best_L2raw:.6f} | "
                f"L_total={best_L2_at_best:.6f} | "
                f"L1={best_L1_at_best:.6f} | "
                f"L2_norm={best_L2norm_at_best:.6f}",
                flush=True,
            )

    Ltot.backward()
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(
            net.parameters(),
            max_norm=1.0,
        ).detach().cpu()
    )
    opt.step()

    loss_tot_hist.append(ltot_v)
    loss_L1_hist.append(l1_v)
    loss_L2norm_hist.append(l2n_v)
    loss_L2raw_hist.append(l2r_v)
    loss_L2graph_hist.append(l2g_v)
    loss_L2marg0_hist.append(l2m0_v)
    loss_L2marg1_hist.append(l2m1_v)
    gradient_norm_hist.append(gradient_norm)
    stage_hist.append(stage)

    if step % print_every == 0:
        print(
            f"Iter {step:5d} [{stage}] | "
            f"Ltot={ltot_v:.6f} | "
            f"L1={l1_v:.6f} | "
            f"L2_norm={l2n_v:.6f} | "
            f"L2_raw={l2r_v:.6f} | "
            f"blur={blur:.3f}",
            flush=True,
        )

    if (
        snapshot_every_refinement_steps > 0
        and step >= warm_steps
        and (step - warm_steps + 1)
        % snapshot_every_refinement_steps == 0
    ):
        snapshot_path = OUT_DIR / (
            f"{SAVE_PREFIX}_SNAPSHOT_iter{step:05d}.p"
        )
        with snapshot_path.open("wb") as handle:
            pickle.dump(
                {
                    "config": {
                        "seed": int(seed),
                        "method": "ordinary_fixed_lag_transition_measure",
                        "iteration_post_update": int(step),
                        "fixed_lag": int(fixed_lag),
                    },
                    "model": {
                        "transformer": scaler,
                        "state_dict": cpu_state_dict(net),
                    },
                },
                handle,
            )

print("Training done.", flush=True)


# ============================================================
# 7. SAVE FINAL / BEST_TOTAL / BEST_L2
# ============================================================
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
path_final = OUT_DIR / f"{SAVE_PREFIX}_FINAL_{stamp}.p"
path_best = OUT_DIR / f"{SAVE_PREFIX}_BEST_{stamp}.p"
path_best_l2 = OUT_DIR / f"{SAVE_PREFIX}_BEST_L2_{stamp}.p"

config = {
    "seed": int(seed),
    "method": "ordinary_fixed_lag_transition_measure",
    "uses_recurrence_events": False,
    "stage2_source_semantics": "uniform_ordinary_time_indices",
    "fixed_lag_observation_steps": int(fixed_lag),
    "fixed_lag_physical_time": float(fixed_lag * dt_obs),
    "dt_obs": float(dt_obs),
    "dt_base": float(dt_base),
    "sub": int(sub),
    "tau_pair": int(tau_pair),
    "tau_delay": int(tau_delay),
    "dim_delay": int(dim_delay),
    "delay_channel": int(DELAY_CH),
    "Nsteps": int(Nsteps),
    "warm_steps": int(warm_steps),
    "network_width": int(network_width),
    "stage2_pairs_per_block": int(stage2_pairs_per_block),
    "L_hits_normalization_denominator": int(L_hits_fullbatch),
    "BATCH_SIZE_STAGE2": int(BATCH_SIZE_STAGE2),
    "stage2_steps_per_cycle": int(stage2_steps_per_cycle),
    "stage1_steps_per_cycle": int(stage1_steps_per_cycle),
    "lr1": float(lr1),
    "lr2": float(lr2),
    "w2_const": float(w2_const),
    "alpha_pair_state": float(alpha_pair_state),
    "alpha_pair_delay": float(alpha_pair_delay),
    "w_marg_u0": float(w_marg_u0),
    "w_marg_u1": float(w_marg_u1),
    "w_defect": float(w_defect),
    "blur_start": float(blur_start),
    "blur_end": float(blur_end),
    "normalization": "global_legal_delay_atoms",
    "best_total_iter_pre_update": int(best_total_it),
    "best_l2_iter_pre_update": int(best_L2_it),
}

base_payload = {
    "config": config,
    "model": {
        "transformer": scaler,
        "state_dict": None,
    },
    "stats": {
        "fixed_lag": int(fixed_lag),
        "legal_source_count": int(len(legal_sources)),
        "u_mean": u_mean,
        "u_std": u_std,
        "x_mean": x_mean,
        "x_std": x_std,
    },
    "train_hist": {
        "stage": np.asarray(stage_hist, dtype=object),
        "L_total": np.asarray(loss_tot_hist, dtype=np.float64),
        "L1": np.asarray(loss_L1_hist, dtype=np.float64),
        "L2_norm": np.asarray(loss_L2norm_hist, dtype=np.float64),
        "L2_raw": np.asarray(loss_L2raw_hist, dtype=np.float64),
        "L2_graph": np.asarray(loss_L2graph_hist, dtype=np.float64),
        "L2_source_marginal": np.asarray(
            loss_L2marg0_hist,
            dtype=np.float64,
        ),
        "L2_target_marginal": np.asarray(
            loss_L2marg1_hist,
            dtype=np.float64,
        ),
        "gradient_norm": np.asarray(
            gradient_norm_hist,
            dtype=np.float64,
        ),
    },
}

payload_final = copy.deepcopy(base_payload)
payload_final["model"]["state_dict"] = cpu_state_dict(net)
with path_final.open("wb") as handle:
    pickle.dump(payload_final, handle)
print("Saved FINAL:", path_final, flush=True)

if best_total_sd is not None:
    payload_best = copy.deepcopy(base_payload)
    payload_best["config"].update(
        {
            "selection": "minimum_full_objective_pre_update",
            "best_total_iter": int(best_total_it),
            "best_total": float(best_total),
            "best_total_L1": float(best_total_L1),
            "best_total_L2_norm": float(best_total_L2norm),
            "best_total_L2_raw": float(best_total_L2raw),
        }
    )
    payload_best["model"]["state_dict"] = best_total_sd
    with path_best.open("wb") as handle:
        pickle.dump(payload_best, handle)
    print(
        f"Saved BEST_TOTAL: {path_best}  "
        f"(L_total={best_total:.6f} @ iter {best_total_it})",
        flush=True,
    )
else:
    print("No BEST_TOTAL saved (never entered Stage 2).", flush=True)

if best_L2_sd is not None:
    payload_best_l2 = copy.deepcopy(base_payload)
    payload_best_l2["config"].update(
        {
            "selection": "minimum_stage2_transition_objective_pre_update",
            "best_l2_iter": int(best_L2_it),
            "best_l2_L2_raw": float(best_L2raw),
            "best_l2_L_total": float(best_L2_at_best),
            "best_l2_L1": float(best_L1_at_best),
            "best_l2_L2_norm": float(best_L2norm_at_best),
        }
    )
    payload_best_l2["model"]["state_dict"] = best_L2_sd
    with path_best_l2.open("wb") as handle:
        pickle.dump(payload_best_l2, handle)
    print(
        f"Saved BEST_L2: {path_best_l2}  "
        f"(L2_raw={best_L2raw:.6f} @ iter {best_L2_it})",
        flush=True,
    )
else:
    print("No BEST_L2 saved (never entered Stage 2).", flush=True)


# ============================================================
# 8. INLINE NOTEBOOK DIAGNOSTIC PLOTS
# ============================================================
fig, ax = plt.subplots(figsize=(10, 3), dpi=140)
ax.plot(loss_tot_hist, label="L_total")
ax.plot(loss_L1_hist, label="L1")
ax.axvline(warm_steps, linestyle="--", linewidth=1, label="refinement starts")
ax.set_title("Fixed-time training loss")
ax.set_xlabel("Iteration")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SAVE_PREFIX}_loss_{stamp}.png", dpi=160)
plt.show()

fig, ax = plt.subplots(figsize=(10, 3), dpi=140)
ax.plot(loss_L2raw_hist, label="L2_raw")
ax.axvline(warm_steps, linestyle="--", linewidth=1, label="refinement starts")
ax.set_title("Fixed-time Stage-2 transition loss")
ax.set_xlabel("Iteration")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SAVE_PREFIX}_L2raw_{stamp}.png", dpi=160)
plt.show()

fig, ax = plt.subplots(figsize=(10, 3), dpi=140)
ax.plot(loss_L2graph_hist, label="L2_graph")
ax.plot(loss_L2marg0_hist, label="source marginal")
ax.plot(loss_L2marg1_hist, label="target marginal")
ax.axvline(warm_steps, linestyle="--", linewidth=1, label="refinement starts")
ax.set_title("Fixed-time Stage-2 loss components")
ax.set_xlabel("Iteration")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SAVE_PREFIX}_L2components_{stamp}.png", dpi=160)
plt.show()

fig, ax = plt.subplots(figsize=(10, 3), dpi=140)
ax.plot(gradient_norm_hist, label="gradient norm before clipping")
ax.axvline(warm_steps, linestyle="--", linewidth=1, label="refinement starts")
ax.set_title("Gradient norm")
ax.set_xlabel("Iteration")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / f"{SAVE_PREFIX}_gradnorm_{stamp}.png", dpi=160)
plt.show()

# Keep useful objects accessible in the notebook after this cell finishes.
fixed_time_run = {
    "net": net,
    "scaler": scaler,
    "config": config,
    "history": base_payload["train_hist"],
    "paths": {
        "FINAL": str(path_final),
        "BEST_TOTAL": str(path_best) if best_total_sd is not None else None,
        "BEST_L2": str(path_best_l2) if best_L2_sd is not None else None,
    },
}

print("Notebook object available as: fixed_time_run", flush=True)

