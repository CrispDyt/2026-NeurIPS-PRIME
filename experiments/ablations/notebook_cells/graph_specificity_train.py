# Extracted notebook code; CLI/path/device/display adaptations only.
from _notebook_bootstrap import prepare
globals().update(prepare(__file__))
# ============================================================
# ONE COMPLETE JUPYTER CELL
#
# Graph-measure specificity ablation on Lorenz63 partial observations.
#
# Methods trained for each seed:
#   1) DIM only
#   2) DIM + recurrence source/target marginals
#   3) DIM + pointwise recurrence-pair MSE + recurrence marginals
#
# Seeds:
#   33, 123, 1000, 13532, 123456
#
# This cell reads train_data.npy and test_data.npy, trains all 15 runs,
# creates a new output directory, and saves FINAL / BEST_TOTAL checkpoints.
#
# IMPORTANT FAIRNESS DESIGN
# -------------------------
# - All methods use exactly the same network, Algorithm-2 parameters,
#   learning rates, Sinkhorn settings, marginal weights, and 4000 updates.
# - The two recurrence-based ablations use the same 1:1 schedule as PRIME:
#       one enhanced update, then one DIM-only update.
# - On an enhanced update:
#       DIM + marginals:
#           L_total = L_DIM + 0.5 * L_marginal_norm
#       DIM + pair-MSE + marginals:
#           L_total = L_DIM + 0.5 * L_pairMSE_marginal_norm
# - The pair-MSE term replaces PRIME's graph OT term with coefficient 1.0.
# - w_marg_u0 = w_marg_u1 = 0.20, exactly as in PRIME.
# ============================================================

import copy
import datetime
import gc
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn as tnn
from torch import optim
from sklearn.preprocessing import MaxAbsScaler
from geomloss import SamplesLoss
def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)


# ============================================================
# 0. DATA PATHS
# ============================================================

# Change only these two paths if your notebook is stored elsewhere.
TRAIN_DATA_PATH = PORT_DATA / "train_data.npy"
TEST_DATA_PATH = PORT_DATA / "test_data.npy"

if not TRAIN_DATA_PATH.is_file():
    raise FileNotFoundError(
        f"train_data.npy not found:\n{TRAIN_DATA_PATH.resolve()}"
    )

if not TEST_DATA_PATH.is_file():
    raise FileNotFoundError(
        f"test_data.npy not found:\n{TEST_DATA_PATH.resolve()}"
    )

train_data = np.load(TRAIN_DATA_PATH).astype(np.float64)
test_data = np.load(TEST_DATA_PATH).astype(np.float64)

if train_data.ndim != 2 or test_data.ndim != 2:
    raise ValueError(
        f"Expected arrays of shape (T,d), got {train_data.shape} and {test_data.shape}"
    )

if train_data.shape[1] != test_data.shape[1]:
    raise ValueError(
        f"Train/test dimensions differ: {train_data.shape[1]} vs {test_data.shape[1]}"
    )

if not np.isfinite(train_data).all() or not np.isfinite(test_data).all():
    raise ValueError("train_data or test_data contains NaN/Inf")


# ============================================================
# 1. EXACT PRIME CONFIGURATION
# ============================================================

dt_obs = 0.1
dt_base = 0.1
sub = int(round(dt_obs / dt_base))

SEEDS = [PORT_ARGS.seed] if PORT_ARGS.seed is not None else [33,123,1000,13532,123456]

cand_frac = 0.5
q_speed = 0.3
M_landmarks = 200
L_sec = 5
tau_delay = 1
dim_delay = 5
eps_in_mult = 0.35
eps_out_mult = 1.8

# Same total number of updates as the supplied PRIME code.
Nsteps = PORT_ARGS.steps if PORT_ARGS.steps is not None else 4000
warm_steps = PORT_ARGS.warm_steps if PORT_ARGS.warm_steps is not None else 0

tau_pair = 2
stage1_batch = 512
lr1 = 1e-3

L_hits_fullbatch = 32
BATCH_SIZE_STAGE2 = 16
lr2 = 1e-3

alpha_pair_state = 1.0
alpha_pair_delay = 0.7

# EXACT PRIME recurrence-marginal coefficients.
w_marg_u0 = 0.20
w_marg_u1 = 0.20
w_defect = 0.0

blur_start = 0.2
blur_end = 0.1

stage2_steps_per_cycle = 1
stage1_steps_per_cycle = 1

w2_const = 0.5

DELAY_CH = 0
NETWORK_WIDTH = 100
GRAD_CLIP = 1.0
PRINT_EVERY = 100

# The pointwise pair-MSE replaces L2_graph with coefficient 1.0.
pair_mse_weight = 1.0

METHODS = [
    "DIM_only",
    "DIM_plus_recurrence_marginals",
    "DIM_plus_pairMSE_recurrence_marginals",
]

OUTPUT_ROOT = PORT_OUTPUT / "Ablation_GraphMeasureSpecificity"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Reproducible CUDA behavior.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(PORT_DEVICE)

print("device:", device)
print("output root:", OUTPUT_ROOT.resolve())
print("seeds:", SEEDS)
print("methods:", METHODS)
print(
    "marginal weights:",
    f"w_marg_u0={w_marg_u0}, w_marg_u1={w_marg_u1}"
)
print(
    "schedule for recurrence-based methods:",
    f"enhanced:DIM-only = {stage2_steps_per_cycle}:{stage1_steps_per_cycle}"
)


# ============================================================
# 2. UTILITIES
# ============================================================

def set_seed(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def make_sinkhorn(blur: float):
    return SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=float(blur),
        backend="tensorized",
    )


def build_net(d: int, width: int = 100):
    return tnn.Sequential(
        tnn.Linear(d, width), tnn.Tanh(),
        tnn.Linear(width, width), tnn.Tanh(),
        tnn.Linear(width, width), tnn.Tanh(),
        tnn.Linear(width, d),
    )


def cpu_state_dict(net):
    return {
        key: value.detach().cpu().clone()
        for key, value in net.state_dict().items()
    }


def build_delay_full_onech(y_scaled, tau, dim, ch):
    T, _ = y_scaled.shape
    min_t = (dim - 1) * tau
    D = np.zeros((T, dim), dtype=np.float64)
    series = y_scaled[:, ch]

    for t in range(min_t, T):
        D[t] = np.asarray(
            [series[t - i * tau] for i in range(dim)],
            dtype=np.float64,
        )

    return D, min_t


def farthest_point_sampling(X, M, start=0):
    N = int(X.shape[0])
    M = int(min(int(M), N))

    if M < 1:
        raise ValueError("Farthest-point sampling received an empty pool")

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
    T = int(U_dw.shape[0])
    hits = []
    visit_lens = []
    last = -10**12
    active = False
    active_i = -1
    enter_t = -1

    for t in range(T):
        d2 = np.sum((C_dw - U_dw[t]) ** 2, axis=1)
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
            d_active = float(
                np.sqrt(np.sum((U_dw[t] - C_dw[active_i]) ** 2))
            )

            if d_active >= eps_out:
                active = False
                active_i = -1

                if enter_t >= 0:
                    visit_lens.append(t - enter_t)

                enter_t = -1

    return (
        np.asarray(hits, dtype=np.int64),
        np.asarray(visit_lens, dtype=np.int64),
    )


def build_delay_data_tensor_onech(
    y_np,
    dim_delay,
    tau_delay,
    ch,
    current_device,
):
    Y = torch.tensor(
        y_np,
        dtype=torch.float32,
        device=current_device,
    )

    series = Y[:, ch]
    T = series.shape[0]
    L = T - tau_delay * (dim_delay - 1)

    delayed = torch.zeros(
        (1, L, dim_delay),
        dtype=torch.float32,
        device=current_device,
    )

    for i in range(dim_delay):
        start = dim_delay * tau_delay - (i + 1) * tau_delay
        delayed[0, :, i] = series[start : start + L]

    return delayed


def method_folder_name(method: str) -> str:
    return method.replace("+", "plus").replace(" ", "_")


# ============================================================
# 3. PREPARE ALL DATA OBJECTS FOR ONE SEED
# ============================================================

def prepare_seed_context(seed_value: int):
    set_seed(seed_value)

    train = np.asarray(train_data, dtype=np.float64)
    T, d = train.shape

    scaler = MaxAbsScaler().fit(train)
    y_scaled = scaler.transform(train).astype(np.float64)

    D_full, min_t = build_delay_full_onech(
        y_scaled,
        tau=tau_delay,
        dim=dim_delay,
        ch=DELAY_CH,
    )

    U_full = np.concatenate([y_scaled, D_full], axis=1)

    T_eff = T - min_t

    speed = np.linalg.norm(
        y_scaled[1:] - y_scaled[:-1],
        axis=1,
    )
    speed = np.concatenate([[speed[0]], speed], axis=0)

    idx_eff = np.arange(min_t, T)
    speed_threshold = np.quantile(speed[idx_eff], q_speed)
    pool = idx_eff[speed[idx_eff] >= speed_threshold]

    candidate_target = max(1, int(round(cand_frac * T_eff)))

    if len(pool) >= candidate_target:
        idx_cand = np.random.choice(
            pool,
            size=candidate_target,
            replace=False,
        )
    else:
        idx_cand = pool.copy()

    idx_cand = np.sort(idx_cand)
    Ucand = U_full[idx_cand]

    u_mean_c = Ucand.mean(axis=0)
    u_std_c = Ucand.std(axis=0) + 1e-6

    Ucand_dw = (Ucand - u_mean_c[None, :]) / u_std_c[None, :]
    Ufull_dw = (U_full - u_mean_c[None, :]) / u_std_c[None, :]

    fps_start = int(np.random.randint(0, len(idx_cand)))
    fps_idx_local = farthest_point_sampling(
        Ucand_dw,
        M=M_landmarks,
        start=fps_start,
    )

    C_dw = Ucand_dw[fps_idx_local]

    nearest_landmark_distances = []

    for i in range(len(C_dw)):
        d2 = np.sum((C_dw - C_dw[i]) ** 2, axis=1)
        d2[i] = np.inf
        nearest_landmark_distances.append(np.sqrt(np.min(d2)))

    nn_med = float(np.median(nearest_landmark_distances))
    eps_in = eps_in_mult * nn_med
    eps_out = eps_out_mult * eps_in

    hits_raw, visit_lens = extract_hits_from_landmarks(
        Ufull_dw,
        C_dw,
        eps_in,
        eps_out,
        L_sec=L_sec,
    )

    hits = np.sort(hits_raw[hits_raw >= min_t])

    if len(hits) < 2:
        raise RuntimeError(
            f"Seed {seed_value}: Algorithm 2 produced fewer than two hits"
        )

    if len(hits) < min(L_hits_fullbatch, 2):
        raise RuntimeError(
            f"Seed {seed_value}: not enough hits for Stage 2"
        )

    taus = np.diff(hits).astype(np.float64)

    XU_data = y_scaled[hits]
    DU_data = D_full[hits]
    U_hits_data = np.concatenate([XU_data, DU_data], axis=1)

    # Stage-2 recurrence whitening is exactly the PRIME whitening.
    u_mean = U_hits_data.mean(axis=0)
    u_std = U_hits_data.std(axis=0) + 1e-6

    # Stage-1 state whitening.
    x_mean = y_scaled.mean(axis=0)
    x_std = y_scaled.std(axis=0) + 1e-6

    context = {
        "seed": int(seed_value),
        "train": train,
        "T": int(T),
        "d": int(d),
        "scaler": scaler,
        "y_scaled": y_scaled,
        "D_full": D_full,
        "U_full": U_full,
        "min_t": int(min_t),
        "hits": hits,
        "taus": taus,
        "visit_lens": visit_lens,
        "XU_data": XU_data,
        "DU_data": DU_data,
        "U_hits_data": U_hits_data,
        "C_dw": C_dw,
        "u_mean_c": u_mean_c,
        "u_std_c": u_std_c,
        "u_mean": u_mean,
        "u_std": u_std,
        "x_mean": x_mean,
        "x_std": x_std,
        "eps_in": float(eps_in),
        "eps_out": float(eps_out),
        "nn_med": float(nn_med),
        "fps_start": int(fps_start),
    }

    return context


# ============================================================
# 4. BUILD THE LOSS FUNCTIONS FOR ONE MODEL
# ============================================================

def build_loss_functions(net, context):
    y_scaled = context["y_scaled"]
    min_t = context["min_t"]
    hits = context["hits"]
    XU_data = context["XU_data"]
    DU_data = context["DU_data"]
    d = context["d"]

    Y_all_t = torch.tensor(
        y_scaled,
        dtype=torch.float32,
        device=device,
    )

    y_delayed_data = build_delay_data_tensor_onech(
        y_scaled,
        dim_delay,
        tau_delay,
        DELAY_CH,
        device,
    )

    u_mean_t = torch.tensor(
        context["u_mean"],
        dtype=torch.float32,
        device=device,
    )

    u_std_t = torch.tensor(
        context["u_std"],
        dtype=torch.float32,
        device=device,
    )

    x_mean_t = torch.tensor(
        context["x_mean"],
        dtype=torch.float32,
        device=device,
    )

    x_std_t = torch.tensor(
        context["x_std"],
        dtype=torch.float32,
        device=device,
    )

    def whiten_u(U):
        return (U - u_mean_t[None, :]) / u_std_t[None, :]

    def whiten_x(X):
        return (X - x_mean_t[None, :]) / x_std_t[None, :]

    def compute_L_DIM(loss_sink):
        # State-space short-horizon graph term from the supplied PRIME code.
        Tmax = len(y_scaled) - tau_pair - 1
        idx = np.random.randint(0, Tmax, size=stage1_batch)

        x0 = torch.tensor(
            y_scaled[idx],
            dtype=torch.float32,
            device=device,
        )

        xT_data = torch.tensor(
            y_scaled[idx + tau_pair],
            dtype=torch.float32,
            device=device,
        )

        x = x0.clone()
        n_micro = int(tau_pair * sub)

        for _ in range(n_micro):
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

        # Delay invariant-measure term from the supplied PRIME code.
        Kfine = tau_delay * dim_delay * sub
        yy = Y_all_t.clone()
        sols = []

        for _ in range(Kfine):
            sols.append(yy)

            for _ in range(sub):
                yy = yy + float(dt_base) * net(yy)

        sol = torch.stack(sols, dim=0)
        sol = sol[:, min_t:, :]

        step_fine = tau_delay * sub
        sol_ch = sol[:, :, DELAY_CH]
        sol_ch = sol_ch.permute(1, 0).contiguous()

        DIM_model = torch.flip(
            sol_ch[:, ::step_fine],
            dims=[1],
        ).unsqueeze(0).contiguous()

        y_delay = y_delayed_data
        L_delay = loss_sink(DIM_model, y_delay).sum()

        glue_defect = torch.mean(
            (DIM_model[:, 0, :-1] - y_delay[:, 0, :-1]) ** 2
        )

        return L_state + L_delay + glue_defect

    def build_recurrence_components(k0, L_hits, loss_sink):
        hit_times = hits[k0 : k0 + L_hits].astype(np.int64)

        X_block_data = torch.tensor(
            XU_data[k0 : k0 + L_hits],
            dtype=torch.float32,
            device=device,
        )

        D_block_data = torch.tensor(
            DU_data[k0 : k0 + L_hits],
            dtype=torch.float32,
            device=device,
        )

        t0 = int(hit_times[0])
        t_start = t0 - int((dim_delay - 1) * tau_delay)
        t_end = int(hit_times[-1])
        total_obs_steps = int(t_end - t_start)

        x = torch.tensor(
            y_scaled[t_start],
            dtype=torch.float32,
            device=device,
        )

        hit_map = {
            int(t): j
            for j, t in enumerate(hit_times.tolist())
        }

        model_states_at_hits = [None] * L_hits
        x_history = [x.clone()]

        for n in range(total_obs_steps + 1):
            t_abs = t_start + n

            if t_abs in hit_map:
                model_states_at_hits[hit_map[t_abs]] = x
                # Same multi-shoot gradient cut as PRIME; no state reset.
                x = x.detach()

            if n == total_obs_steps:
                break

            for _ in range(sub):
                x = x + float(dt_base) * net(x)

            x_history.append(x.clone())

        if any(value is None for value in model_states_at_hits):
            raise RuntimeError("A recurrence state was not recorded")

        X_block_model = torch.stack(model_states_at_hits, dim=0)

        model_delay_vectors = []

        for t_abs in hit_times:
            idx0 = int(t_abs - t_start)
            lag_values = []

            for i in range(dim_delay):
                lag_values.append(
                    x_history[idx0 - i * tau_delay][DELAY_CH]
                )

            model_delay_vectors.append(torch.stack(lag_values, dim=0))

        D_block_model = torch.stack(model_delay_vectors, dim=0)

        U_block_model = torch.cat(
            [X_block_model, D_block_model],
            dim=1,
        )

        U_block_data = torch.cat(
            [X_block_data, D_block_data],
            dim=1,
        )

        Um = whiten_u(U_block_model)
        Ud = whiten_u(U_block_data)

        xm, dm = Um[:, :d], Um[:, d:]
        xd, dd = Ud[:, :d], Ud[:, d:]

        u_model = torch.cat(
            [alpha_pair_state * xm, alpha_pair_delay * dm],
            dim=1,
        )

        u_data = torch.cat(
            [alpha_pair_state * xd, alpha_pair_delay * dd],
            dim=1,
        )

        source_model = u_model[:-1]
        target_model = u_model[1:]
        source_data = u_data[:-1]
        target_data = u_data[1:]

        pair_model = torch.cat(
            [source_model, target_model],
            dim=1,
        )

        pair_data = torch.cat(
            [source_data, target_data],
            dim=1,
        )

        # Computed for diagnostics only in these ablations.
        L_graph = loss_sink(pair_model, pair_data)

        # EXACT recurrence marginal losses and coefficients used by PRIME.
        L_marg0 = loss_sink(source_model, source_data)
        L_marg1 = loss_sink(target_model, target_data)
        L_marg = w_marg_u0 * L_marg0 + w_marg_u1 * L_marg1

        # Pointwise MSE between the temporally aligned recurrence pairs.
        L_pair_mse = torch.mean((pair_model - pair_data) ** 2)

        defect = torch.mean((Um[:, :d] - Ud[:, :d]) ** 2)

        return {
            "L_graph": L_graph,
            "L_marg0": L_marg0,
            "L_marg1": L_marg1,
            "L_marg": L_marg,
            "L_pair_mse": L_pair_mse,
            "defect": defect,
        }

    return compute_L_DIM, build_recurrence_components


# ============================================================
# 5. TRAIN ONE METHOD FOR ONE SEED
# ============================================================

def train_one_method(
    method,
    seed_value,
    context,
    initial_state_dict,
):
    run_folder = (
        OUTPUT_ROOT
        / method_folder_name(method)
        / f"seed_{seed_value}"
    )
    run_folder.mkdir(parents=True, exist_ok=True)

    # Identical initialization and training random stream within a seed.
    set_seed(seed_value + 10_000_000)

    net = build_net(context["d"], width=NETWORK_WIDTH).to(device)
    net.load_state_dict(initial_state_dict)

    opt1 = optim.Adam(net.parameters(), lr=lr1)
    opt2 = optim.Adam(net.parameters(), lr=lr2)

    compute_L_DIM, build_recurrence_components = build_loss_functions(
        net,
        context,
    )

    best_total = float("inf")
    best_total_step = -1
    best_total_sd = None
    best_total_components = None

    best_selected_raw = float("inf")
    best_selected_step = -1
    best_selected_sd = None
    best_selected_components = None

    history = {
        "stage": [],
        "L_total": [],
        "L_DIM": [],
        "L_selected_norm": [],
        "L_selected_raw": [],
        "L_graph_diagnostic": [],
        "L_marg0": [],
        "L_marg1": [],
        "L_marg": [],
        "L_pair_mse": [],
        "gradient_norm": [],
    }

    print("\n" + "=" * 100)
    print(f"METHOD: {method} | SEED: {seed_value}")
    print("folder:", run_folder.resolve())
    print("=" * 100)

    start_time = time.perf_counter()
    net.train()

    for step in range(Nsteps):
        fraction = step / max(1, Nsteps - 1)
        blur = blur_start + (blur_end - blur_start) * fraction
        loss_sink = make_sinkhorn(blur)

        if method == "DIM_only":
            stage = "DIM"
        else:
            cycle_length = stage2_steps_per_cycle + stage1_steps_per_cycle
            cycle_position = step % cycle_length
            stage = (
                "enhanced"
                if cycle_position < stage2_steps_per_cycle
                else "DIM"
            )

        optimizer = opt1 if stage == "DIM" else opt2
        optimizer.zero_grad(set_to_none=True)

        L_DIM = compute_L_DIM(loss_sink)

        zero = torch.zeros((), dtype=torch.float32, device=device)
        L_graph_diag = zero
        L_marg0 = zero
        L_marg1 = zero
        L_marg = zero
        L_pair_mse = zero
        selected_raw = zero
        selected_norm = zero

        if stage == "DIM":
            L_total = L_DIM

        else:
            L_hits = min(L_hits_fullbatch, len(context["hits"]))
            k0max = len(context["hits"]) - L_hits

            if k0max < 0:
                raise RuntimeError(
                    f"Seed {seed_value}: insufficient recurrence hits"
                )

            block_components = []

            for _ in range(BATCH_SIZE_STAGE2):
                k0 = np.random.randint(0, k0max + 1)
                block_components.append(
                    build_recurrence_components(k0, L_hits, loss_sink)
                )

            def mean_component(name):
                return torch.stack(
                    [block[name] for block in block_components]
                ).mean()

            L_graph_diag = mean_component("L_graph")
            L_marg0 = mean_component("L_marg0")
            L_marg1 = mean_component("L_marg1")
            L_marg = mean_component("L_marg")
            L_pair_mse = mean_component("L_pair_mse")
            defect = mean_component("defect")

            if method == "DIM_plus_recurrence_marginals":
                selected_raw = L_marg + w_defect * defect

            elif method == "DIM_plus_pairMSE_recurrence_marginals":
                selected_raw = (
                    pair_mse_weight * L_pair_mse
                    + L_marg
                    + w_defect * defect
                )

            else:
                raise ValueError(f"Unknown method: {method}")

            # Same PRIME normalization by the number of event atoms.
            selected_norm = selected_raw / float(L_hits)

            # Same outer coefficient as PRIME.
            L_total = L_DIM + float(w2_const) * selected_norm

        L_total.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            net.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        values = {
            "L_total": float(L_total.detach().cpu()),
            "L_DIM": float(L_DIM.detach().cpu()),
            "L_selected_norm": float(selected_norm.detach().cpu()),
            "L_selected_raw": float(selected_raw.detach().cpu()),
            "L_graph_diagnostic": float(L_graph_diag.detach().cpu()),
            "L_marg0": float(L_marg0.detach().cpu()),
            "L_marg1": float(L_marg1.detach().cpu()),
            "L_marg": float(L_marg.detach().cpu()),
            "L_pair_mse": float(L_pair_mse.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }

        history["stage"].append(stage)

        for key in values:
            history[key].append(values[key])

        # BEST_TOTAL is selected over all steps for DIM-only,
        # and over enhanced steps for the recurrence-based methods,
        # matching the original PRIME checkpoint-selection logic.
        eligible_for_best = (
            method == "DIM_only"
            or stage == "enhanced"
        )

        if eligible_for_best and values["L_total"] < best_total:
            best_total = values["L_total"]
            best_total_step = step
            best_total_sd = cpu_state_dict(net)
            best_total_components = dict(values)

            print(
                f"  --> [New BEST_TOTAL] step={step} "
                f"L_total={best_total:.6f} "
                f"L_DIM={values['L_DIM']:.6f} "
                f"selected_raw={values['L_selected_raw']:.6f}"
            )

        if (
            stage == "enhanced"
            and values["L_selected_raw"] < best_selected_raw
        ):
            best_selected_raw = values["L_selected_raw"]
            best_selected_step = step
            best_selected_sd = cpu_state_dict(net)
            best_selected_components = dict(values)

        if step % PRINT_EVERY == 0:
            print(
                f"step {step:4d}/{Nsteps - 1} "
                f"[{stage}] "
                f"L_total={values['L_total']:.6f} "
                f"L_DIM={values['L_DIM']:.6f} "
                f"L_marg={values['L_marg']:.6f} "
                f"L_pairMSE={values['L_pair_mse']:.6f} "
                f"blur={blur:.3f}"
            )

    train_seconds = time.perf_counter() - start_time

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = method_folder_name(method)

    final_path = run_folder / f"{prefix}_FINAL_{stamp}.p"
    best_total_path = run_folder / f"{prefix}_BEST_TOTAL_{stamp}.p"
    best_selected_path = run_folder / f"{prefix}_BEST_STAGE2_{stamp}.p"

    config = {
        "method": method,
        "seed": int(seed_value),
        "dt_obs": float(dt_obs),
        "dt_base": float(dt_base),
        "sub": int(sub),
        "Nsteps": int(Nsteps),
        "warm_steps": int(warm_steps),
        "stage2_steps_per_cycle": int(stage2_steps_per_cycle),
        "stage1_steps_per_cycle": int(stage1_steps_per_cycle),
        "tau_pair": int(tau_pair),
        "stage1_batch": int(stage1_batch),
        "lr1": float(lr1),
        "lr2": float(lr2),
        "L_hits_fullbatch": int(L_hits_fullbatch),
        "BATCH_SIZE_STAGE2": int(BATCH_SIZE_STAGE2),
        "alpha_pair_state": float(alpha_pair_state),
        "alpha_pair_delay": float(alpha_pair_delay),
        "w_marg_u0": float(w_marg_u0),
        "w_marg_u1": float(w_marg_u1),
        "w_defect": float(w_defect),
        "pair_mse_weight": float(pair_mse_weight),
        "w2_const": float(w2_const),
        "blur_start": float(blur_start),
        "blur_end": float(blur_end),
        "cand_frac": float(cand_frac),
        "q_speed": float(q_speed),
        "M_landmarks": int(M_landmarks),
        "L_sec": int(L_sec),
        "tau_delay": int(tau_delay),
        "dim_delay": int(dim_delay),
        "eps_in_mult": float(eps_in_mult),
        "eps_out_mult": float(eps_out_mult),
        "min_t": int(context["min_t"]),
        "eps_in": float(context["eps_in"]),
        "eps_out": float(context["eps_out"]),
        "nn_med": float(context["nn_med"]),
        "N_hits": int(len(context["hits"])),
        "train_seconds": float(train_seconds),
    }

    stats = {
        "hits": context["hits"],
        "taus_steps": context["taus"],
        "visit_lens": context["visit_lens"],
        "C_dw": context["C_dw"],
        "u_mean_c": context["u_mean_c"],
        "u_std_c": context["u_std_c"],
        "u_mean": context["u_mean"],
        "u_std": context["u_std"],
    }

    history_serializable = {
        "stage": np.asarray(history["stage"], dtype=object),
    }

    for key, value in history.items():
        if key != "stage":
            history_serializable[key] = np.asarray(value, dtype=np.float64)

    base_payload = {
        "config": config,
        "model": {
            "transformer": context["scaler"],
            "state_dict": None,
        },
        "stats": stats,
        "train_hist": history_serializable,
    }

    payload_final = copy.deepcopy(base_payload)
    payload_final["model"]["state_dict"] = cpu_state_dict(net)

    with final_path.open("wb") as handle:
        pickle.dump(payload_final, handle)

    if best_total_sd is None:
        raise RuntimeError(
            f"No BEST_TOTAL was recorded for {method}, seed {seed_value}"
        )

    payload_best_total = copy.deepcopy(base_payload)
    payload_best_total["config"].update(
        {
            "selection": "minimum_training_total",
            "best_total_step": int(best_total_step),
            "best_total": float(best_total),
            "best_total_components": best_total_components,
        }
    )
    payload_best_total["model"]["state_dict"] = best_total_sd

    with best_total_path.open("wb") as handle:
        pickle.dump(payload_best_total, handle)

    if best_selected_sd is not None:
        payload_best_selected = copy.deepcopy(base_payload)
        payload_best_selected["config"].update(
            {
                "selection": "minimum_selected_stage2_objective",
                "best_selected_step": int(best_selected_step),
                "best_selected_raw": float(best_selected_raw),
                "best_selected_components": best_selected_components,
            }
        )
        payload_best_selected["model"]["state_dict"] = best_selected_sd

        with best_selected_path.open("wb") as handle:
            pickle.dump(payload_best_selected, handle)
    else:
        best_selected_path = None

    # Loss plots.
    plt.figure(figsize=(11, 4), dpi=140)
    plt.plot(history["L_total"], label="L_total")
    plt.plot(history["L_DIM"], label="L_DIM")
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.title(f"{method}, seed {seed_value}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        run_folder / f"{prefix}_loss_total_DIM.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    if method != "DIM_only":
        plt.figure(figsize=(11, 4), dpi=140)
        plt.plot(history["L_marg"], label="recurrence marginals")
        plt.plot(history["L_pair_mse"], label="pair MSE")
        plt.plot(
            history["L_graph_diagnostic"],
            label="graph OT (diagnostic only)",
        )
        plt.xlabel("Update")
        plt.ylabel("Loss component")
        plt.title(f"Stage-2 diagnostics: {method}, seed {seed_value}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            run_folder / f"{prefix}_stage2_components.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close()

    run_summary = {
        "method": method,
        "seed": int(seed_value),
        "N_hits": int(len(context["hits"])),
        "median_K": float(np.median(context["taus"])),
        "mean_K": float(np.mean(context["taus"])),
        "best_total_step": int(best_total_step),
        "best_total": float(best_total),
        "train_seconds": float(train_seconds),
        "final_checkpoint": str(final_path.resolve()),
        "best_total_checkpoint": str(best_total_path.resolve()),
        "best_stage2_checkpoint": (
            str(best_selected_path.resolve())
            if best_selected_path is not None
            else None
        ),
        "w_marg_u0": float(w_marg_u0),
        "w_marg_u1": float(w_marg_u1),
        "pair_mse_weight": float(pair_mse_weight),
        "w2_const": float(w2_const),
    }

    with (run_folder / "run_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(run_summary, handle, indent=2)

    print("Saved FINAL:", final_path)
    print("Saved BEST_TOTAL:", best_total_path)

    if best_selected_path is not None:
        print("Saved BEST_STAGE2:", best_selected_path)

    print(f"Training time: {train_seconds:.2f} seconds")

    del net, opt1, opt2, compute_L_DIM, build_recurrence_components
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_summary


# ============================================================
# 6. RESUME ONLY MISSING RUNS
# ============================================================

# Reduce notebook output and memory pressure.
PRINT_EVERY = 500


def get_run_folder(method, seed_value):
    return (
        OUTPUT_ROOT
        / method_folder_name(method)
        / f"seed_{seed_value}"
    )


def is_run_complete(method, seed_value):
    """
    A run is regarded as complete only when it contains:
      1. run_summary.json
      2. at least one BEST_TOTAL checkpoint
      3. at least one FINAL checkpoint
    """

    run_folder = get_run_folder(
        method,
        seed_value,
    )

    summary_path = (
        run_folder
        / "run_summary.json"
    )

    best_total_files = (
        list(
            run_folder.glob(
                "*_BEST_TOTAL_*.p"
            )
        )
        if run_folder.exists()
        else []
    )

    final_files = (
        list(
            run_folder.glob(
                "*_FINAL_*.p"
            )
        )
        if run_folder.exists()
        else []
    )

    return (
        summary_path.is_file()
        and len(best_total_files) > 0
        and len(final_files) > 0
    )


def load_completed_run_summary(
    method,
    seed_value,
):
    summary_path = (
        get_run_folder(
            method,
            seed_value,
        )
        / "run_summary.json"
    )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def collect_all_completed_summaries():
    """
    Reload every completed run from disk so that the final table
    includes both old runs and newly resumed runs.
    """

    summaries = []

    for method in METHODS:
        for seed_value in SEEDS:

            if is_run_complete(
                method,
                seed_value,
            ):
                summaries.append(
                    load_completed_run_summary(
                        method,
                        seed_value,
                    )
                )

    return summaries


def save_current_summary_table():
    """
    Save an up-to-date summary after every newly completed run.
    """

    completed_summaries = (
        collect_all_completed_summaries()
    )

    if len(completed_summaries) == 0:
        return None

    dataframe = pd.DataFrame(
        completed_summaries
    )

    dataframe = dataframe.sort_values(
        [
            "method",
            "seed",
        ]
    ).reset_index(
        drop=True
    )

    summary_path = (
        OUTPUT_ROOT
        / "all_runs_summary.csv"
    )

    dataframe.to_csv(
        summary_path,
        index=False,
    )

    return dataframe


# ------------------------------------------------------------
# Show status before resuming
# ------------------------------------------------------------

status_rows = []

for method in METHODS:
    for seed_value in SEEDS:

        completed = is_run_complete(
            method,
            seed_value,
        )

        status_rows.append(
            {
                "method": method,
                "seed": int(seed_value),
                "completed": bool(completed),
            }
        )

status_dataframe = pd.DataFrame(
    status_rows
)

print(
    "\nStatus before resuming:"
)

display(
    status_dataframe
)

missing_dataframe = status_dataframe.loc[
    ~status_dataframe["completed"]
].copy()

print(
    "\nRuns that will be trained:"
)

display(
    missing_dataframe
)


# ------------------------------------------------------------
# Resume missing runs
# ------------------------------------------------------------

for seed_value in SEEDS:

    methods_to_run = [
        method
        for method in METHODS
        if not is_run_complete(
            method,
            seed_value,
        )
    ]

    if len(methods_to_run) == 0:

        print(
            f"\n[SKIP SEED {seed_value}] "
            "All three methods are already complete."
        )

        continue

    print(
        "\n"
        + "#" * 110
    )

    print(
        f"RESUMING SEED {seed_value}"
    )

    print(
        "Methods to train:",
        methods_to_run,
    )

    print(
        "#" * 110
    )

    # Recreate exactly the same Algorithm-2 events and all
    # seed-specific normalization statistics.
    context = prepare_seed_context(
        seed_value
    )

    print(
        f"Seed {seed_value}: "
        f"N_hits={len(context['hits'])}, "
        f"median_K={np.median(context['taus']):.3f}, "
        f"mean_K={np.mean(context['taus']):.3f}"
    )

    # Recreate exactly the same network initialization used
    # by the original complete experiment.
    set_seed(
        seed_value
        + 1_000_000
    )

    base_network = build_net(
        context["d"],
        width=NETWORK_WIDTH,
    )

    initial_state_dict = cpu_state_dict(
        base_network
    )

    del base_network

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for method in methods_to_run:

        # Recheck immediately before training, so rerunning this
        # cell can never overwrite an already completed run.
        if is_run_complete(
            method,
            seed_value,
        ):

            print(
                f"[SKIP COMPLETED] "
                f"method={method}, "
                f"seed={seed_value}"
            )

            continue

        print(
            "\n"
            + "=" * 110
        )

        print(
            f"STARTING MISSING RUN: "
            f"method={method}, "
            f"seed={seed_value}"
        )

        print(
            "=" * 110
        )

        summary = train_one_method(
            method=method,
            seed_value=seed_value,
            context=context,
            initial_state_dict=initial_state_dict,
        )

        print(
            f"[COMPLETED AND SAVED] "
            f"method={method}, "
            f"seed={seed_value}"
        )

        # Immediately rebuild the summary CSV after every run.
        current_dataframe = (
            save_current_summary_table()
        )

        if current_dataframe is not None:

            print(
                f"Currently completed: "
                f"{len(current_dataframe)}/"
                f"{len(SEEDS) * len(METHODS)}"
            )

        plt.close(
            "all"
        )

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass

    del context
    del initial_state_dict

    plt.close(
        "all"
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 7. FINAL STATUS AND COMPLETE SUMMARY
# ============================================================

final_status_rows = []

for method in METHODS:
    for seed_value in SEEDS:

        final_status_rows.append(
            {
                "method": method,
                "seed": int(seed_value),
                "completed": is_run_complete(
                    method,
                    seed_value,
                ),
            }
        )

final_status_dataframe = pd.DataFrame(
    final_status_rows
)

print(
    "\n"
    + "=" * 110
)

print(
    "FINAL COMPLETION STATUS"
)

print(
    "=" * 110
)

display(
    final_status_dataframe
)


still_missing = final_status_dataframe.loc[
    ~final_status_dataframe[
        "completed"
    ]
]

if len(still_missing) > 0:

    print(
        "\nThe following runs are still incomplete:"
    )

    display(
        still_missing
    )

else:

    print(
        "\nAll 15 runs are complete."
    )


summary_dataframe = (
    save_current_summary_table()
)

if summary_dataframe is not None:

    print(
        "\n"
        + "=" * 110
    )

    print(
        "COMPLETE GRAPH-MEASURE ABLATION SUMMARY"
    )

    print(
        "=" * 110
    )

    columns_to_show = [
        column
        for column in [
            "method",
            "seed",
            "N_hits",
            "median_K",
            "mean_K",
            "best_total_step",
            "best_total",
            "train_seconds",
            "best_total_checkpoint",
        ]
        if column in summary_dataframe.columns
    ]

    display(
        summary_dataframe[
            columns_to_show
        ]
    )

    print(
        "\nSaved complete summary:"
    )

    print(
        (
            OUTPUT_ROOT
            / "all_runs_summary.csv"
        ).resolve()
    )
