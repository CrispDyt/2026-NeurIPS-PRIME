# Extracted notebook code; CLI/path/device/display adaptations only.
from _notebook_bootstrap import prepare
globals().update(prepare(__file__))

# ============================================================
# DIM warm-up -> PRIME-only ablation
#
# One complete Jupyter Notebook cell.
#
# For each seed:
#   1) train with L_DIM only for 3000 updates;
#   2) continue with L_PRIME only for 4000 updates.
#
# Seeds:
#   33, 123, 123456, 1000, 13532
#
# Data files expected relative to the notebook:
#   ../train_data.npy
#   ../test_data.npy
# ============================================================

import copy
import datetime
import gc
import json
import os
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as tnn
from geomloss import SamplesLoss
from sklearn.preprocessing import MaxAbsScaler
from torch import optim


# ============================================================
# 0. DATA
# ============================================================

TRAIN_PATH = PORT_DATA / "train_data.npy"
TEST_PATH = PORT_DATA / "test_data.npy"

if not TRAIN_PATH.is_file():
    raise FileNotFoundError(f"Training data not found: {TRAIN_PATH.resolve()}")

if not TEST_PATH.is_file():
    raise FileNotFoundError(f"Test data not found: {TEST_PATH.resolve()}")

train_data = np.load(TRAIN_PATH)
test_data = np.load(TEST_PATH)

train_data = np.asarray(train_data, dtype=np.float64)
test_data = np.asarray(test_data, dtype=np.float64)

if train_data.ndim != 2:
    raise ValueError(f"train_data must have shape (T,d), got {train_data.shape}")

if test_data.ndim != 2:
    raise ValueError(f"test_data must have shape (T,d), got {test_data.shape}")

if train_data.shape[1] != test_data.shape[1]:
    raise ValueError(
        f"train/test dimension mismatch: "
        f"{train_data.shape[1]} vs {test_data.shape[1]}"
    )

if not np.all(np.isfinite(train_data)):
    raise ValueError("train_data contains NaN or Inf")

if not np.all(np.isfinite(test_data)):
    raise ValueError("test_data contains NaN or Inf")


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEEDS = [PORT_ARGS.seed] if PORT_ARGS.seed is not None else [33,123,123456,1000,13532]

dt_obs = 0.1
dt_base = 0.1
sub = int(round(dt_obs / dt_base))

if sub < 1 or not np.isclose(sub * dt_base, dt_obs):
    raise ValueError(
        f"dt_obs/dt_base must be a positive integer; "
        f"dt_obs={dt_obs}, dt_base={dt_base}, sub={sub}"
    )

# Algorithm 2: unchanged
cand_frac = 0.5
q_speed = 0.3
M_landmarks = 200
L_sec = 5
tau_delay = 1
dim_delay = 5
eps_in_mult = 0.35
eps_out_mult = 1.8
DELAY_CH = 0

# Training schedule requested by the user:
# 3000 DIM-only updates, followed by 4000 PRIME-only updates.
DIM_WARMUP_STEPS = PORT_ARGS.warm_steps if PORT_ARGS.warm_steps is not None else 3000
PRIME_ONLY_STEPS = PORT_ARGS.steps if PORT_ARGS.steps is not None else 2000
TOTAL_STEPS = DIM_WARMUP_STEPS + PRIME_ONLY_STEPS

# DIM loss: unchanged
tau_pair = 2
stage1_batch = 512
lr1 = 1e-3

# PRIME loss: unchanged
L_hits_fullbatch = 32
BATCH_SIZE_STAGE2 = 16
lr2 = 1e-3

alpha_pair_state = 1.0
alpha_pair_delay = 0.7
w_marg_u0 = 0.20
w_marg_u1 = 0.20
w_defect = 0.0

blur_start = 0.2
blur_end = 0.1

# Retained exactly from the supplied implementation.
# During PRIME-only training the optimized objective is:
#     w2_const * L_PRIME
w2_const = 0.5

network_width = 100
gradient_clip = 1.0
print_every = 100

# New output root. Each seed receives its own subfolder.
OUTPUT_ROOT = PORT_OUTPUT / "Ablation_DIMWarmup_then_PRIMEOnly"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DEVICE
# ============================================================

def select_training_device():
    if not torch.cuda.is_available():
        print("CUDA is not visible; using CPU.")
        return torch.device("cpu")

    try:
        probe = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        del probe
        torch.cuda.empty_cache()
        return torch.device("cuda")
    except RuntimeError as error:
        print("CUDA is visible but unavailable; using CPU.")
        print("CUDA error:", error)
        return torch.device("cpu")


device = torch.device(PORT_DEVICE)
print("device:", device)
print("output root:", OUTPUT_ROOT.resolve())


# ============================================================
# 3. UTILITIES
# ============================================================

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed_value)

    # The network is an MLP, but these flags improve reproducibility
    # if the environment invokes cuDNN elsewhere.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_sinkhorn(blur):
    return SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=float(blur),
        backend="tensorized",
    )


def build_net(d, width=100):
    return tnn.Sequential(
        tnn.Linear(d, width),
        tnn.Tanh(),
        tnn.Linear(width, width),
        tnn.Tanh(),
        tnn.Linear(width, width),
        tnn.Tanh(),
        tnn.Linear(width, d),
    )


def cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


def build_delay_full_onech(y_scaled, tau, dim, ch):
    """
    D_t = [y_ch(t), y_ch(t-tau), ..., y_ch(t-(dim-1)tau)].
    """
    T_local, d_local = y_scaled.shape

    if not (0 <= ch < d_local):
        raise ValueError(f"DELAY_CH={ch} is invalid for d={d_local}")

    min_t_local = (dim - 1) * tau

    if T_local <= min_t_local:
        raise ValueError(
            f"Trajectory length {T_local} is too short for min_t={min_t_local}"
        )

    D = np.zeros((T_local, dim), dtype=np.float64)
    series = y_scaled[:, ch]

    for time_index in range(min_t_local, T_local):
        D[time_index] = np.asarray(
            [
                series[time_index - lag_index * tau]
                for lag_index in range(dim)
            ],
            dtype=np.float64,
        )

    return D, min_t_local


def farthest_point_sampling(X, M, start=0):
    N = int(X.shape[0])
    M = int(min(int(M), N))

    if M < 1:
        raise ValueError("Farthest-point sampling received an empty set")

    indices = np.zeros((M,), dtype=np.int64)
    indices[0] = int(start)

    distance_squared = np.sum(
        (X - X[indices[0]]) ** 2,
        axis=1,
    )

    for landmark_index in range(1, M):
        selected = int(np.argmax(distance_squared))
        indices[landmark_index] = selected

        new_distance_squared = np.sum(
            (X - X[selected]) ** 2,
            axis=1,
        )

        distance_squared = np.minimum(
            distance_squared,
            new_distance_squared,
        )

    return indices


def extract_hits_from_landmarks(
    U_dw,
    C_dw,
    eps_in,
    eps_out,
    minimum_gap,
):
    """
    Original Algorithm-2 hysteresis and minimum-gap hit extraction.
    """
    T_local = int(U_dw.shape[0])

    hits_local = []
    visit_lengths = []

    last_hit = -10**12
    active = False
    active_landmark = -1
    entrance_time = -1

    for time_index in range(T_local):
        distance_squared = np.sum(
            (C_dw - U_dw[time_index]) ** 2,
            axis=1,
        )

        nearest_landmark = int(np.argmin(distance_squared))
        nearest_distance = float(
            np.sqrt(distance_squared[nearest_landmark])
        )

        if not active:
            if (
                nearest_distance <= eps_in
                and (time_index - last_hit) >= int(minimum_gap)
            ):
                hits_local.append(time_index)
                last_hit = time_index
                active = True
                active_landmark = nearest_landmark
                entrance_time = time_index

        else:
            active_distance = float(
                np.sqrt(
                    np.sum(
                        (
                            U_dw[time_index]
                            - C_dw[active_landmark]
                        ) ** 2
                    )
                )
            )

            if active_distance >= eps_out:
                active = False
                active_landmark = -1

                if entrance_time >= 0:
                    visit_lengths.append(
                        time_index - entrance_time
                    )

                entrance_time = -1

    return (
        np.asarray(hits_local, dtype=np.int64),
        np.asarray(visit_lengths, dtype=np.int64),
    )


def build_delay_data_tensor_onech(
    y_np,
    delay_dimension,
    delay_lag,
    channel,
    target_device,
):
    """
    Original one-channel DIM delay cloud.
    """
    Y = torch.as_tensor(
        y_np,
        dtype=torch.float32,
        device=target_device,
    )

    series = Y[:, channel]
    T_local = int(series.shape[0])

    length = T_local - delay_lag * (delay_dimension - 1)

    delayed = torch.zeros(
        (1, length, delay_dimension),
        dtype=torch.float32,
        device=target_device,
    )

    for lag_index in range(delay_dimension):
        start = (
            delay_dimension * delay_lag
            - (lag_index + 1) * delay_lag
        )

        delayed[0, :, lag_index] = series[
            start : start + length
        ]

    return delayed


def save_loss_plots(
    seed_folder,
    seed_value,
    total_history,
    dim_history,
    prime_raw_history,
    prime_graph_history,
    prime_source_history,
    prime_target_history,
):
    warmup_boundary = DIM_WARMUP_STEPS

    figure = plt.figure(figsize=(10, 3), dpi=140)
    plt.plot(total_history, label="optimized objective")
    plt.axvline(
        warmup_boundary,
        linestyle="--",
        linewidth=1,
        label="PRIME-only starts",
    )
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.title(f"Seed {seed_value}: optimized objective")
    plt.legend()
    plt.tight_layout()
    figure.savefig(
        seed_folder / f"seed{seed_value}_optimized_objective.png",
        dpi=160,
    )
    plt.close(figure)

    figure = plt.figure(figsize=(10, 3), dpi=140)
    plt.plot(dim_history, label=r"$L^{DIM}$ (warm-up only)")
    plt.axvline(
        warmup_boundary,
        linestyle="--",
        linewidth=1,
        label="PRIME-only starts",
    )
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.title(f"Seed {seed_value}: DIM warm-up loss")
    plt.legend()
    plt.tight_layout()
    figure.savefig(
        seed_folder / f"seed{seed_value}_DIM_warmup_loss.png",
        dpi=160,
    )
    plt.close(figure)

    figure = plt.figure(figsize=(10, 3), dpi=140)
    plt.plot(prime_raw_history, label=r"$L^{PRIME}_{raw}$")
    plt.plot(prime_graph_history, label="graph")
    plt.plot(prime_source_history, label="source marginal")
    plt.plot(prime_target_history, label="target marginal")
    plt.axvline(
        warmup_boundary,
        linestyle="--",
        linewidth=1,
        label="PRIME-only starts",
    )
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.title(f"Seed {seed_value}: PRIME-only refinement")
    plt.legend()
    plt.tight_layout()
    figure.savefig(
        seed_folder / f"seed{seed_value}_PRIME_components.png",
        dpi=160,
    )
    plt.close(figure)


# ============================================================
# 4. ONE-SEED TRAINING FUNCTION
# ============================================================

def train_one_seed(seed_value):
    set_seed(seed_value)

    seed_folder = OUTPUT_ROOT / f"seed_{seed_value}"
    seed_folder.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"SEED {seed_value}")
    print(f"folder: {seed_folder.resolve()}")
    print("=" * 90)

    train = train_data.copy()
    T, d = train.shape

    scaler = MaxAbsScaler().fit(train)
    y_scaled = scaler.transform(train).astype(np.float64)

    # --------------------------------------------------------
    # Algorithm 2: unchanged
    # --------------------------------------------------------

    D_full, min_t = build_delay_full_onech(
        y_scaled,
        tau=tau_delay,
        dim=dim_delay,
        ch=DELAY_CH,
    )

    U_full = np.concatenate(
        [y_scaled, D_full],
        axis=1,
    )

    T_eff = T - min_t

    speed = np.linalg.norm(
        y_scaled[1:] - y_scaled[:-1],
        axis=1,
    )

    speed = np.concatenate(
        [[speed[0]], speed],
        axis=0,
    )

    effective_indices = np.arange(
        min_t,
        T,
        dtype=np.int64,
    )

    speed_threshold = np.quantile(
        speed[effective_indices],
        q_speed,
    )

    speed_pool = effective_indices[
        speed[effective_indices] >= speed_threshold
    ]

    candidate_count = max(
        1,
        int(round(cand_frac * T_eff)),
    )

    if len(speed_pool) >= candidate_count:
        candidate_indices = np.random.choice(
            speed_pool,
            size=candidate_count,
            replace=False,
        )
    else:
        candidate_indices = speed_pool.copy()

    candidate_indices = np.sort(
        candidate_indices
    )

    if len(candidate_indices) == 0:
        raise RuntimeError(
            f"Seed {seed_value}: empty Algorithm-2 candidate pool"
        )

    U_candidates = U_full[
        candidate_indices
    ]

    candidate_mean = U_candidates.mean(axis=0)
    candidate_std = U_candidates.std(axis=0) + 1e-6

    U_candidates_whitened = (
        U_candidates - candidate_mean[None, :]
    ) / candidate_std[None, :]

    U_full_whitened = (
        U_full - candidate_mean[None, :]
    ) / candidate_std[None, :]

    fps_start = int(
        np.random.randint(
            0,
            len(candidate_indices),
        )
    )

    fps_local_indices = farthest_point_sampling(
        U_candidates_whitened,
        M=M_landmarks,
        start=fps_start,
    )

    landmarks_whitened = U_candidates_whitened[
        fps_local_indices
    ]

    nearest_landmark_distances = []

    for landmark_index in range(
        len(landmarks_whitened)
    ):
        distance_squared = np.sum(
            (
                landmarks_whitened
                - landmarks_whitened[landmark_index]
            ) ** 2,
            axis=1,
        )

        distance_squared[landmark_index] = np.inf

        nearest_landmark_distances.append(
            np.sqrt(np.min(distance_squared))
        )

    nearest_landmark_median = float(
        np.median(nearest_landmark_distances)
    )

    eps_in = eps_in_mult * nearest_landmark_median
    eps_out = eps_out_mult * eps_in

    hits_raw, visit_lengths = extract_hits_from_landmarks(
        U_full_whitened,
        landmarks_whitened,
        eps_in,
        eps_out,
        minimum_gap=L_sec,
    )

    hits = hits_raw[
        hits_raw >= min_t
    ]

    hits = np.sort(hits)

    if len(hits) < 2:
        raise RuntimeError(
            f"Seed {seed_value}: Algorithm 2 produced "
            f"only {len(hits)} hit(s)"
        )

    intervals = np.diff(hits).astype(np.float64)

    data_states_at_hits = y_scaled[hits]
    data_delays_at_hits = D_full[hits]

    data_atoms_at_hits = np.concatenate(
        [
            data_states_at_hits,
            data_delays_at_hits,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Tensors and whitening: unchanged
    # --------------------------------------------------------

    Y_all_t = torch.as_tensor(
        y_scaled,
        dtype=torch.float32,
        device=device,
    )

    atom_mean = data_atoms_at_hits.mean(axis=0)
    atom_std = data_atoms_at_hits.std(axis=0) + 1e-6

    atom_mean_t = torch.as_tensor(
        atom_mean,
        dtype=torch.float32,
        device=device,
    )

    atom_std_t = torch.as_tensor(
        atom_std,
        dtype=torch.float32,
        device=device,
    )

    state_mean = y_scaled.mean(axis=0)
    state_std = y_scaled.std(axis=0) + 1e-6

    state_mean_t = torch.as_tensor(
        state_mean,
        dtype=torch.float32,
        device=device,
    )

    state_std_t = torch.as_tensor(
        state_std,
        dtype=torch.float32,
        device=device,
    )

    def whiten_atom(atom):
        return (
            atom - atom_mean_t[None, :]
        ) / atom_std_t[None, :]

    def whiten_state(state):
        return (
            state - state_mean_t[None, :]
        ) / state_std_t[None, :]

    delayed_data = build_delay_data_tensor_onech(
        y_scaled,
        delay_dimension=dim_delay,
        delay_lag=tau_delay,
        channel=DELAY_CH,
        target_device=device,
    )

    # --------------------------------------------------------
    # Model and two optimizers: unchanged
    # --------------------------------------------------------

    model = build_net(
        d,
        width=network_width,
    ).to(device)

    dim_optimizer = optim.Adam(
        model.parameters(),
        lr=lr1,
    )

    prime_optimizer = optim.Adam(
        model.parameters(),
        lr=lr2,
    )

    # --------------------------------------------------------
    # L_DIM: unchanged
    # --------------------------------------------------------

    def compute_DIM_loss(loss_sink):
        maximum_start = (
            len(y_scaled)
            - tau_pair
            - 1
        )

        if maximum_start <= 0:
            raise RuntimeError(
                "Trajectory is too short for tau_pair"
            )

        indices = np.random.randint(
            0,
            maximum_start,
            size=stage1_batch,
        )

        x0 = torch.as_tensor(
            y_scaled[indices],
            dtype=torch.float32,
            device=device,
        )

        xT_data = torch.as_tensor(
            y_scaled[indices + tau_pair],
            dtype=torch.float32,
            device=device,
        )

        x = x0.clone()

        for _ in range(int(tau_pair * sub)):
            x = x + float(dt_base) * model(x)

        xT_model = x

        data_state_pairs = torch.cat(
            [
                whiten_state(x0),
                whiten_state(xT_data),
            ],
            dim=1,
        )

        model_state_pairs = torch.cat(
            [
                whiten_state(x0),
                whiten_state(xT_model),
            ],
            dim=1,
        )

        state_graph_loss = loss_sink(
            model_state_pairs,
            data_state_pairs,
        )

        Kfine = tau_delay * dim_delay * sub
        yy = Y_all_t.clone()
        solutions = []

        for _ in range(Kfine):
            solutions.append(yy)

            for _ in range(sub):
                yy = yy + float(dt_base) * model(yy)

        solution_tensor = torch.stack(
            solutions,
            dim=0,
        )

        solution_tensor = solution_tensor[
            :,
            min_t:,
            :,
        ]

        fine_delay_step = tau_delay * sub

        solution_channel = solution_tensor[
            :,
            :,
            DELAY_CH,
        ].permute(
            1,
            0,
        ).contiguous()

        model_delay_cloud = torch.flip(
            solution_channel[
                :,
                ::fine_delay_step,
            ],
            dims=[1],
        ).unsqueeze(
            0
        ).contiguous()

        delay_loss = loss_sink(
            model_delay_cloud,
            delayed_data,
        ).sum()

        glue_defect = torch.mean(
            (
                model_delay_cloud[:, 0, :-1]
                - delayed_data[:, 0, :-1]
            ) ** 2
        )

        return (
            state_graph_loss
            + delay_loss
            + glue_defect
        )

    # --------------------------------------------------------
    # L_PRIME: unchanged
    # --------------------------------------------------------

    def build_PRIME_block(
        block_start,
        block_length,
        loss_sink,
    ):
        hit_times = hits[
            block_start : block_start + block_length
        ].astype(np.int64)

        data_state_block = torch.as_tensor(
            data_states_at_hits[
                block_start : block_start + block_length
            ],
            dtype=torch.float32,
            device=device,
        )

        data_delay_block = torch.as_tensor(
            data_delays_at_hits[
                block_start : block_start + block_length
            ],
            dtype=torch.float32,
            device=device,
        )

        first_hit = int(hit_times[0])
        rollout_start = (
            first_hit
            - int((dim_delay - 1) * tau_delay)
        )

        final_hit = int(hit_times[-1])

        total_observation_steps = int(
            final_hit - rollout_start
        )

        x = torch.as_tensor(
            y_scaled[rollout_start],
            dtype=torch.float32,
            device=device,
        )

        hit_to_position = {
            int(hit_time): position
            for position, hit_time
            in enumerate(hit_times.tolist())
        }

        model_states_at_hits = [
            None
        ] * block_length

        model_history = [
            x.clone()
        ]

        for observation_step in range(
            total_observation_steps + 1
        ):
            absolute_time = (
                rollout_start + observation_step
            )

            if absolute_time in hit_to_position:
                position = hit_to_position[
                    absolute_time
                ]

                model_states_at_hits[
                    position
                ] = x

                # Preserve the original PRIME multi-shoot cut.
                # The numerical state is not replaced.
                x = x.detach()

            if (
                observation_step
                == total_observation_steps
            ):
                break

            for _ in range(sub):
                x = x + float(dt_base) * model(x)

            model_history.append(
                x.clone()
            )

        if any(
            state is None
            for state in model_states_at_hits
        ):
            raise AssertionError(
                "At least one PRIME event state was not recorded"
            )

        model_state_block = torch.stack(
            model_states_at_hits,
            dim=0,
        )

        model_delay_vectors = []

        for absolute_hit_time in hit_times:
            history_position = int(
                absolute_hit_time - rollout_start
            )

            lagged_values = []

            for lag_index in range(dim_delay):
                lagged_values.append(
                    model_history[
                        history_position
                        - lag_index * tau_delay
                    ][DELAY_CH]
                )

            model_delay_vectors.append(
                torch.stack(
                    lagged_values,
                    dim=0,
                )
            )

        model_delay_block = torch.stack(
            model_delay_vectors,
            dim=0,
        )

        model_atoms = torch.cat(
            [
                model_state_block,
                model_delay_block,
            ],
            dim=1,
        )

        data_atoms = torch.cat(
            [
                data_state_block,
                data_delay_block,
            ],
            dim=1,
        )

        model_atoms_whitened = whiten_atom(
            model_atoms
        )

        data_atoms_whitened = whiten_atom(
            data_atoms
        )

        model_atoms_weighted = torch.cat(
            [
                alpha_pair_state
                * model_atoms_whitened[:, :d],

                alpha_pair_delay
                * model_atoms_whitened[:, d:],
            ],
            dim=1,
        )

        data_atoms_weighted = torch.cat(
            [
                alpha_pair_state
                * data_atoms_whitened[:, :d],

                alpha_pair_delay
                * data_atoms_whitened[:, d:],
            ],
            dim=1,
        )

        model_sources = model_atoms_weighted[:-1]
        model_targets = model_atoms_weighted[1:]

        data_sources = data_atoms_weighted[:-1]
        data_targets = data_atoms_weighted[1:]

        model_pairs = torch.cat(
            [
                model_sources,
                model_targets,
            ],
            dim=1,
        )

        data_pairs = torch.cat(
            [
                data_sources,
                data_targets,
            ],
            dim=1,
        )

        graph_loss = loss_sink(
            model_pairs,
            data_pairs,
        )

        source_marginal_loss = loss_sink(
            model_sources,
            data_sources,
        )

        target_marginal_loss = loss_sink(
            model_targets,
            data_targets,
        )

        defect = torch.mean(
            (
                model_atoms_whitened[:, :d]
                - data_atoms_whitened[:, :d]
            ) ** 2
        )

        prime_raw = (
            graph_loss
            + w_marg_u0 * source_marginal_loss
            + w_marg_u1 * target_marginal_loss
            + w_defect * defect
        )

        prime_normalized = (
            prime_raw / float(block_length)
        )

        return (
            prime_normalized,
            prime_raw,
            graph_loss,
            source_marginal_loss,
            target_marginal_loss,
            defect,
        )

    # --------------------------------------------------------
    # Histories
    # --------------------------------------------------------

    optimized_objective_history = []
    dim_loss_history = []
    prime_normalized_history = []
    prime_raw_history = []
    prime_graph_history = []
    prime_source_history = []
    prime_target_history = []
    gradient_norm_history = []
    phase_history = []

    # --------------------------------------------------------
    # Phase A: 3000 DIM-only warm-up updates
    # --------------------------------------------------------

    print(
        f"Seed {seed_value}: DIM warm-up "
        f"({DIM_WARMUP_STEPS} updates)"
    )

    model.train()

    for warmup_step in range(
        DIM_WARMUP_STEPS
    ):
        global_step = warmup_step

        fraction = (
            global_step
            / max(1, TOTAL_STEPS - 1)
        )

        blur = (
            blur_start
            + (blur_end - blur_start) * fraction
        )

        loss_sink = make_sinkhorn(blur)

        dim_optimizer.zero_grad(
            set_to_none=True
        )

        dim_loss = compute_DIM_loss(
            loss_sink
        )

        objective = dim_loss

        objective_value = float(
            objective.detach().cpu()
        )

        objective.backward()

        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip,
            ).detach().cpu()
        )

        dim_optimizer.step()

        optimized_objective_history.append(
            objective_value
        )

        dim_loss_history.append(
            objective_value
        )

        prime_normalized_history.append(
            np.nan
        )

        prime_raw_history.append(
            np.nan
        )

        prime_graph_history.append(
            np.nan
        )

        prime_source_history.append(
            np.nan
        )

        prime_target_history.append(
            np.nan
        )

        gradient_norm_history.append(
            gradient_norm
        )

        phase_history.append(
            "DIM_warmup"
        )

        if (
            warmup_step % print_every == 0
            or warmup_step
            == DIM_WARMUP_STEPS - 1
        ):
            print(
                f"  warm-up {warmup_step:4d}/"
                f"{DIM_WARMUP_STEPS - 1} | "
                f"L_DIM={objective_value:.6f} | "
                f"blur={blur:.3f}"
            )

    # Save the exact branch point for a paired joint-objective arm.
    warmup_path = (
        seed_folder
        / f"DIMWarmup_seed{seed_value}_step{DIM_WARMUP_STEPS}.p"
    )

    warmup_payload = {
        "config": {
            "seed": int(seed_value),
            "method": "DIM_warmup_branch_point",
            "DIM_warmup_steps": int(DIM_WARMUP_STEPS),
            "PRIME_only_steps_planned": int(PRIME_ONLY_STEPS),
            "dt_obs": float(dt_obs),
            "dt_base": float(dt_base),
            "sub": int(sub),
            "network_width": int(network_width),
        },
        "model": {
            "transformer": scaler,
            "state_dict": cpu_state_dict(model),
        },
        "stats": {
            "hits": hits,
            "taus_steps": intervals,
            "visit_lens": visit_lengths,
            "C_dw": landmarks_whitened,
            "u_mean_c": candidate_mean,
            "u_std_c": candidate_std,
            "u_mean": atom_mean,
            "u_std": atom_std,
            "x_mean": state_mean,
            "x_std": state_std,
        },
        "rng_state": capture_rng_state(),
    }

    with warmup_path.open("wb") as handle:
        pickle.dump(
            warmup_payload,
            handle,
        )

    print("  Saved warm-up branch point:", warmup_path)

    # --------------------------------------------------------
    # Phase B: 4000 PRIME-only updates
    # --------------------------------------------------------

    print(
        f"Seed {seed_value}: PRIME-only refinement "
        f"({PRIME_ONLY_STEPS} updates)"
    )

    block_length = min(
        L_hits_fullbatch,
        len(hits),
    )

    if block_length < 2:
        raise RuntimeError(
            f"Seed {seed_value}: not enough hits for PRIME blocks"
        )

    maximum_block_start = (
        len(hits) - block_length
    )

    best_prime_raw = float("inf")
    best_prime_iteration = -1
    best_prime_state_dict = None
    best_prime_normalized = None
    best_prime_graph = None
    best_prime_source = None
    best_prime_target = None

    for prime_step in range(
        PRIME_ONLY_STEPS
    ):
        global_step = (
            DIM_WARMUP_STEPS + prime_step
        )

        fraction = (
            global_step
            / max(1, TOTAL_STEPS - 1)
        )

        blur = (
            blur_start
            + (blur_end - blur_start) * fraction
        )

        loss_sink = make_sinkhorn(blur)

        prime_optimizer.zero_grad(
            set_to_none=True
        )

        blocks = []

        for _ in range(
            BATCH_SIZE_STAGE2
        ):
            block_start = int(
                np.random.randint(
                    0,
                    maximum_block_start + 1,
                )
            )

            blocks.append(
                build_PRIME_block(
                    block_start,
                    block_length,
                    loss_sink,
                )
            )

        prime_normalized = torch.stack(
            [block[0] for block in blocks]
        ).mean()

        prime_raw = torch.stack(
            [block[1] for block in blocks]
        ).mean()

        prime_graph = torch.stack(
            [block[2] for block in blocks]
        ).mean()

        prime_source = torch.stack(
            [block[3] for block in blocks]
        ).mean()

        prime_target = torch.stack(
            [block[4] for block in blocks]
        ).mean()

        # CRITICAL ABLATION CHANGE:
        # No L_DIM is computed or optimized after warm-up.
        objective = (
            float(w2_const)
            * prime_normalized
        )

        objective_value = float(
            objective.detach().cpu()
        )

        prime_normalized_value = float(
            prime_normalized.detach().cpu()
        )

        prime_raw_value = float(
            prime_raw.detach().cpu()
        )

        prime_graph_value = float(
            prime_graph.detach().cpu()
        )

        prime_source_value = float(
            prime_source.detach().cpu()
        )

        prime_target_value = float(
            prime_target.detach().cpu()
        )

        # Save the exact pre-update parameters corresponding
        # to the reported PRIME loss.
        if prime_raw_value < best_prime_raw:
            best_prime_raw = prime_raw_value
            best_prime_iteration = prime_step
            best_prime_state_dict = cpu_state_dict(model)
            best_prime_normalized = prime_normalized_value
            best_prime_graph = prime_graph_value
            best_prime_source = prime_source_value
            best_prime_target = prime_target_value

            print(
                f"  --> [New BEST_PRIME] "
                f"prime_step={prime_step} | "
                f"L_PRIME_raw={best_prime_raw:.6f} | "
                f"L_PRIME_norm={best_prime_normalized:.6f}"
            )

        objective.backward()

        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip,
            ).detach().cpu()
        )

        prime_optimizer.step()

        optimized_objective_history.append(
            objective_value
        )

        dim_loss_history.append(
            np.nan
        )

        prime_normalized_history.append(
            prime_normalized_value
        )

        prime_raw_history.append(
            prime_raw_value
        )

        prime_graph_history.append(
            prime_graph_value
        )

        prime_source_history.append(
            prime_source_value
        )

        prime_target_history.append(
            prime_target_value
        )

        gradient_norm_history.append(
            gradient_norm
        )

        phase_history.append(
            "PRIME_only"
        )

        if (
            prime_step % print_every == 0
            or prime_step
            == PRIME_ONLY_STEPS - 1
        ):
            print(
                f"  PRIME-only {prime_step:4d}/"
                f"{PRIME_ONLY_STEPS - 1} | "
                f"objective={objective_value:.6f} | "
                f"L_PRIME_raw={prime_raw_value:.6f} | "
                f"blur={blur:.3f}"
            )

    if best_prime_state_dict is None:
        raise RuntimeError(
            f"Seed {seed_value}: no BEST_PRIME state was recorded"
        )

    # --------------------------------------------------------
    # Save checkpoints
    # --------------------------------------------------------

    timestamp = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    final_path = (
        seed_folder
        / f"DIMWarmup_PRIMEOnly_seed{seed_value}_FINAL_{timestamp}.p"
    )

    best_path = (
        seed_folder
        / f"DIMWarmup_PRIMEOnly_seed{seed_value}_BEST_PRIME_{timestamp}.p"
    )

    config = {
        "seed": int(seed_value),
        "method": "DIM_warmup_then_PRIME_only",
        "post_warmup_objective": "w2_const * L_PRIME_norm",
        "includes_DIM_after_warmup": False,
        "DIM_warmup_steps": int(DIM_WARMUP_STEPS),
        "PRIME_only_steps": int(PRIME_ONLY_STEPS),
        "total_updates": int(TOTAL_STEPS),
        "dt_obs": float(dt_obs),
        "dt_base": float(dt_base),
        "sub": int(sub),
        "cand_frac": float(cand_frac),
        "q_speed": float(q_speed),
        "M_landmarks": int(M_landmarks),
        "L_sec": int(L_sec),
        "tau_delay": int(tau_delay),
        "dim_delay": int(dim_delay),
        "delay_channel": int(DELAY_CH),
        "eps_in_mult": float(eps_in_mult),
        "eps_out_mult": float(eps_out_mult),
        "tau_pair": int(tau_pair),
        "stage1_batch": int(stage1_batch),
        "lr1": float(lr1),
        "L_hits_fullbatch": int(L_hits_fullbatch),
        "BATCH_SIZE_STAGE2": int(BATCH_SIZE_STAGE2),
        "lr2": float(lr2),
        "alpha_pair_state": float(alpha_pair_state),
        "alpha_pair_delay": float(alpha_pair_delay),
        "w_marg_u0": float(w_marg_u0),
        "w_marg_u1": float(w_marg_u1),
        "w_defect": float(w_defect),
        "blur_start": float(blur_start),
        "blur_end": float(blur_end),
        "w2_const": float(w2_const),
        "network_width": int(network_width),
        "gradient_clip": float(gradient_clip),
        "N_hits": int(len(hits)),
        "min_t": int(min_t),
        "eps_in": float(eps_in),
        "eps_out": float(eps_out),
        "nn_med": float(nearest_landmark_median),
        "best_prime_step_pre_update": int(
            best_prime_iteration
        ),
        "best_prime_raw": float(
            best_prime_raw
        ),
        "best_prime_normalized": float(
            best_prime_normalized
        ),
    }

    common_payload = {
        "config": config,
        "model": {
            "transformer": scaler,
            "state_dict": None,
        },
        "stats": {
            "hits": hits,
            "taus_steps": intervals,
            "visit_lens": visit_lengths,
            "C_dw": landmarks_whitened,
            "u_mean_c": candidate_mean,
            "u_std_c": candidate_std,
            "u_mean": atom_mean,
            "u_std": atom_std,
            "x_mean": state_mean,
            "x_std": state_std,
        },
        "train_hist": {
            "phase": np.asarray(
                phase_history,
                dtype=object,
            ),
            "optimized_objective": np.asarray(
                optimized_objective_history,
                dtype=np.float64,
            ),
            "L_DIM": np.asarray(
                dim_loss_history,
                dtype=np.float64,
            ),
            "L_PRIME_norm": np.asarray(
                prime_normalized_history,
                dtype=np.float64,
            ),
            "L_PRIME_raw": np.asarray(
                prime_raw_history,
                dtype=np.float64,
            ),
            "L_PRIME_graph": np.asarray(
                prime_graph_history,
                dtype=np.float64,
            ),
            "L_PRIME_source_marginal": np.asarray(
                prime_source_history,
                dtype=np.float64,
            ),
            "L_PRIME_target_marginal": np.asarray(
                prime_target_history,
                dtype=np.float64,
            ),
            "gradient_norm": np.asarray(
                gradient_norm_history,
                dtype=np.float64,
            ),
        },
    }

    final_payload = copy.deepcopy(
        common_payload
    )

    final_payload["config"]["selection"] = (
        "final_post_update"
    )

    final_payload["model"]["state_dict"] = (
        cpu_state_dict(model)
    )

    with final_path.open("wb") as handle:
        pickle.dump(
            final_payload,
            handle,
        )

    best_payload = copy.deepcopy(
        common_payload
    )

    best_payload["config"]["selection"] = (
        "minimum_PRIME_raw_pre_update"
    )

    best_payload["config"].update(
        {
            "best_prime_graph": float(
                best_prime_graph
            ),
            "best_prime_source_marginal": float(
                best_prime_source
            ),
            "best_prime_target_marginal": float(
                best_prime_target
            ),
        }
    )

    best_payload["model"]["state_dict"] = (
        best_prime_state_dict
    )

    with best_path.open("wb") as handle:
        pickle.dump(
            best_payload,
            handle,
        )

    save_loss_plots(
        seed_folder=seed_folder,
        seed_value=seed_value,
        total_history=optimized_objective_history,
        dim_history=dim_loss_history,
        prime_raw_history=prime_raw_history,
        prime_graph_history=prime_graph_history,
        prime_source_history=prime_source_history,
        prime_target_history=prime_target_history,
    )

    result = {
        "seed": int(seed_value),
        "N_hits": int(len(hits)),
        "median_K": float(
            np.median(intervals)
        ),
        "mean_K": float(
            np.mean(intervals)
        ),
        "best_prime_step": int(
            best_prime_iteration
        ),
        "best_prime_raw": float(
            best_prime_raw
        ),
        "warmup_checkpoint": str(
            warmup_path.resolve()
        ),
        "best_checkpoint": str(
            best_path.resolve()
        ),
        "final_checkpoint": str(
            final_path.resolve()
        ),
    }

    with (
        seed_folder / "run_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            result,
            handle,
            indent=2,
        )

    print("  Saved BEST_PRIME:", best_path)
    print("  Saved FINAL:", final_path)
    print(f"Seed {seed_value} completed.")

    del (
        model,
        dim_optimizer,
        prime_optimizer,
        Y_all_t,
        delayed_data,
        atom_mean_t,
        atom_std_t,
        state_mean_t,
        state_std_t,
    )

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ============================================================
# 5. RUN ALL FIVE SEEDS
# ============================================================

print("=" * 90)
print("DIM warm-up -> PRIME-only ablation")
print(f"Seeds: {SEEDS}")
print(
    f"Per seed: {DIM_WARMUP_STEPS} DIM-only updates "
    f"+ {PRIME_ONLY_STEPS} PRIME-only updates "
    f"= {TOTAL_STEPS} total updates"
)
print(
    "Post-warm-up optimized objective: "
    "w2_const * L_PRIME_norm"
)
print("=" * 90)

all_results = []

for seed_value in SEEDS:
    try:
        all_results.append(
            train_one_seed(seed_value)
        )
    except Exception as error:
        failure_record = {
            "seed": int(seed_value),
            "error_type": type(error).__name__,
            "error": str(error),
        }

        failure_path = (
            OUTPUT_ROOT
            / f"seed_{seed_value}"
            / "FAILED.json"
        )

        failure_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with failure_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                failure_record,
                handle,
                indent=2,
            )

        print(
            f"\n[FAILED] seed {seed_value}: "
            f"{type(error).__name__}: {error}"
        )

        # Continue to the remaining seeds.
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# 6. AGGREGATE SUMMARY
# ============================================================

summary_dataframe = pd.DataFrame(
    all_results
)

summary_csv = (
    OUTPUT_ROOT
    / "DIMWarmup_then_PRIMEOnly_all_seeds_summary.csv"
)

summary_dataframe.to_csv(
    summary_csv,
    index=False,
)

print("\nCompleted-seed summary:")
display(summary_dataframe)

print("\nSaved summary:")
print(summary_csv.resolve())

print("\nAll requested runs finished.")

