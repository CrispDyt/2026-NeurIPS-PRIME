# Extracted notebook code; CLI/path/device/display adaptations only.
from _notebook_bootstrap import prepare
globals().update(prepare(__file__))
# ============================================================
# ONE COMPLETE JUPYTER CELL
#
# Compare across seeds:
#   1. PRIME
#   2. DIM only
#   3. DIM + recurrence marginals
#   4. DIM + pair-MSE + recurrence marginals
#
# Metrics:
#   - Attractor W2
#   - PSD log-L1
#
# Seeds:
#   33, 123, 1000, 13532, 123456
# ============================================================

import io
import json
import pickle
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from scipy.signal import welch
from sklearn.preprocessing import MaxAbsScaler
from geomloss import SamplesLoss
def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT_DIR = PORT_REPO / "checkpoints/lorenz63"

ABLATION_ROOT = PORT_CHECKPOINTS / "notebook/Ablation_GraphMeasureSpecificity"

SEEDS = [PORT_ARGS.seed] if PORT_ARGS.seed is not None else [33,123,1000,13532,123456]

METHOD_SPECS = OrderedDict(
    [
        ("PRIME", None),

        (
            "DIM only",
            "DIM_only",
        ),

        (
            "DIM + recurrence marginals",
            "DIM_plus_recurrence_marginals",
        ),

        (
            "DIM + pair-MSE + recurrence marginals",
            "DIM_plus_pairMSE_recurrence_marginals",
        ),
    ]
)

DT_OBS_DEFAULT = 0.1
W2_BLUR = 0.05
PSD_NFFT = 2048

# CPU is slower but avoids CUDA-busy errors and guarantees
# exactly the same evaluation device for every checkpoint.
DEVICE = torch.device(PORT_DEVICE)

PLOT_ATTRACTORS = PORT_ARGS.plots

EVALUATION_DIR = PORT_OUTPUT

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("Evaluation device:", DEVICE)
print("Evaluation output:", EVALUATION_DIR)


# ============================================================
# 2. LOAD TRAIN AND TEST DATA
# ============================================================

TRAIN_DATA_PATH = PORT_DATA / "train_data.npy"

TEST_DATA_PATH = PORT_DATA / "test_data.npy"


# Prefer variables already loaded in the Notebook.
if "train_data" not in globals():

    if not TRAIN_DATA_PATH.is_file():
        raise FileNotFoundError(
            "train_data is not defined and the fallback file "
            f"does not exist:\n{TRAIN_DATA_PATH}"
        )

    train_data = np.load(
        TRAIN_DATA_PATH
    )


if "test_data" not in globals():

    if not TEST_DATA_PATH.is_file():
        raise FileNotFoundError(
            "test_data is not defined and the fallback file "
            f"does not exist:\n{TEST_DATA_PATH}"
        )

    test_data = np.load(
        TEST_DATA_PATH
    )


train_array = np.asarray(
    train_data,
    dtype=np.float64,
)

test_array = np.asarray(
    test_data,
    dtype=np.float64,
)


if train_array.ndim != 2:
    raise ValueError(
        f"train_data must have shape (T,d), "
        f"got {train_array.shape}"
    )

if test_array.ndim != 2:
    raise ValueError(
        f"test_data must have shape (T,d), "
        f"got {test_array.shape}"
    )

if train_array.shape[1] != test_array.shape[1]:
    raise ValueError(
        "train_data and test_data dimensions differ: "
        f"{train_array.shape[1]} vs {test_array.shape[1]}"
    )

if not np.isfinite(train_array).all():
    raise ValueError(
        "train_data contains NaN or Inf"
    )

if not np.isfinite(test_array).all():
    raise ValueError(
        "test_data contains NaN or Inf"
    )


STATE_DIM = int(
    test_array.shape[1]
)

T_EVAL = int(
    len(test_array)
)

KMAX = (
    T_EVAL - 1
)

REFERENCE = test_array.copy()


print(
    f"T_eval={T_EVAL}, "
    f"state_dim={STATE_DIM}, "
    f"KMAX={KMAX}"
)


# ============================================================
# 3. CHECKPOINT DISCOVERY
# ============================================================

def resolve_prime_checkpoint(seed_value):
    """
    Prefer:
        1-step MSE {seed}/Implicit_BEST.p

    Otherwise use the newest ordinary BEST checkpoint,
    excluding BEST_L2, FINAL, and NoWarmUp checkpoints.
    """

    seed_folder = (
        ROOT_DIR
        / f"seed_{seed_value}"
    )

    if not seed_folder.is_dir():
        raise FileNotFoundError(
            f"PRIME seed folder was not found:\n"
            f"{seed_folder}"
        )

    exact_path = (
        seed_folder
        / "Implicit_BEST.p"
    )

    if exact_path.is_file():
        return exact_path.resolve()

    candidates = []

    for checkpoint in seed_folder.glob(
        "Implicit_BEST*.p"
    ):

        name_upper = checkpoint.name.upper()

        if "BEST_L2" in name_upper:
            continue

        if "FINAL" in name_upper:
            continue

        if "NOWARMUP" in name_upper:
            continue

        candidates.append(
            checkpoint.resolve()
        )

    if not candidates:

        available_files = sorted(
            path.name
            for path in seed_folder.glob("*")
        )

        raise FileNotFoundError(
            f"No PRIME BEST checkpoint was found for "
            f"seed {seed_value} inside:\n"
            f"{seed_folder}\n\n"
            "Available files:\n"
            + "\n".join(available_files)
        )

    candidates.sort(
        key=lambda path: path.stat().st_mtime
    )

    if len(candidates) > 1:

        print(
            f"[WARNING] Multiple PRIME BEST checkpoints "
            f"found for seed {seed_value}."
        )

        for checkpoint in candidates:
            print("   ", checkpoint)

        print(
            "Using newest:",
            candidates[-1],
        )

    return candidates[-1]


def resolve_ablation_checkpoint(
    method_folder_name,
    seed_value,
):
    """
    Find the BEST_TOTAL checkpoint saved by the graph-measure
    ablation training code.
    """

    run_folder = (
        ABLATION_ROOT
        / method_folder_name
        / f"seed_{seed_value}"
    )

    if not run_folder.is_dir():
        raise FileNotFoundError(
            f"Ablation folder was not found:\n"
            f"{run_folder}"
        )

    # First try run_summary.json.
    run_summary_path = (
        run_folder
        / "run_summary.json"
    )

    if run_summary_path.is_file():

        with run_summary_path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            run_summary = json.load(
                handle
            )

        possible_keys = [
            "best_total_checkpoint",
            "best_checkpoint",
            "BEST_TOTAL_checkpoint",
        ]

        for key in possible_keys:

            saved_value = run_summary.get(
                key
            )

            if not saved_value:
                continue

            saved_path = Path(
                saved_value
            ).expanduser()

            possible_paths = [
                saved_path,
                run_folder / saved_path,
                run_folder / saved_path.name,
            ]

            for possible_path in possible_paths:

                if possible_path.is_file():
                    return possible_path.resolve()

    # Fallback: discover BEST_TOTAL checkpoint from folder.
    candidates = sorted(
        run_folder.glob(
            "*_BEST_TOTAL_*.p"
        ),
        key=lambda path: path.stat().st_mtime,
    )

    if not candidates:

        available_files = sorted(
            path.name
            for path in run_folder.glob("*")
        )

        raise FileNotFoundError(
            f"No BEST_TOTAL checkpoint was found inside:\n"
            f"{run_folder}\n\n"
            "Available files:\n"
            + "\n".join(available_files)
        )

    if len(candidates) > 1:

        print(
            f"[WARNING] Multiple BEST_TOTAL checkpoints "
            f"found inside:\n{run_folder}"
        )

        for checkpoint in candidates:
            print("   ", checkpoint.name)

        print(
            "Using newest:",
            candidates[-1].name,
        )

    return candidates[-1].resolve()


# Build checkpoint manifest.
manifest_rows = []

for seed_value in SEEDS:

    manifest_rows.append(
        {
            "seed": int(seed_value),
            "method": "PRIME",
            "checkpoint": resolve_prime_checkpoint(
                seed_value
            ),
        }
    )

    for method_name, folder_name in METHOD_SPECS.items():

        if method_name == "PRIME":
            continue

        manifest_rows.append(
            {
                "seed": int(seed_value),
                "method": method_name,
                "checkpoint": resolve_ablation_checkpoint(
                    folder_name,
                    seed_value,
                ),
            }
        )


checkpoint_manifest = pd.DataFrame(
    manifest_rows
)

METHOD_ORDER = list(
    METHOD_SPECS.keys()
)

checkpoint_manifest["method"] = pd.Categorical(
    checkpoint_manifest["method"],
    categories=METHOD_ORDER,
    ordered=True,
)

checkpoint_manifest = (
    checkpoint_manifest
    .sort_values(
        [
            "seed",
            "method",
        ]
    )
    .reset_index(
        drop=True
    )
)


print("\nCheckpoint manifest:")

display(
    checkpoint_manifest
)


# ============================================================
# 4. CPU-SAFE CHECKPOINT LOADER
# ============================================================

class CPUUnpickler(pickle.Unpickler):

    def find_class(
        self,
        module,
        name,
    ):

        if (
            module == "torch.storage"
            and name == "_load_from_bytes"
        ):

            return lambda data: torch.load(
                io.BytesIO(data),
                map_location="cpu",
                weights_only=False,
            )

        return super().find_class(
            module,
            name,
        )


def load_pickle_cpu(checkpoint_path):

    with Path(checkpoint_path).open(
        "rb"
    ) as handle:

        return CPUUnpickler(
            handle
        ).load()


def smart_load(checkpoint_path):

    torch_error = None

    try:

        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    except Exception as error:

        torch_error = error

    try:

        return load_pickle_cpu(
            checkpoint_path
        )

    except Exception as pickle_error:

        raise RuntimeError(
            f"Failed to load checkpoint:\n"
            f"{checkpoint_path}\n\n"
            f"torch.load error:\n"
            f"{torch_error}\n\n"
            f"pickle error:\n"
            f"{pickle_error}"
        )


def looks_like_state_dict(obj):

    return (
        isinstance(obj, dict)
        and len(obj) > 0
        and any(
            isinstance(key, str)
            and key.endswith(".weight")
            for key in obj.keys()
        )
        and all(
            torch.is_tensor(value)
            for value in obj.values()
        )
    )


def load_checkpoint_components(
    checkpoint_path,
):

    checkpoint_object = smart_load(
        checkpoint_path
    )

    state_dict = None
    scaler = None
    config = {}


    if looks_like_state_dict(
        checkpoint_object
    ):

        state_dict = checkpoint_object


    elif isinstance(
        checkpoint_object,
        dict,
    ):

        config = checkpoint_object.get(
            "config",
            {},
        )

        model_block = checkpoint_object.get(
            "model"
        )

        if isinstance(
            model_block,
            dict,
        ):

            state_dict = model_block.get(
                "state_dict"
            )

            scaler = model_block.get(
                "transformer",
                model_block.get("scaler"),
            )


        if state_dict is None:

            state_dict = checkpoint_object.get(
                "state_dict"
            )


        if state_dict is None:

            state_dict = checkpoint_object.get(
                "model_state_dict"
            )


        if scaler is None:

            scaler = checkpoint_object.get(
                "transformer"
            )


        if scaler is None:

            scaler = checkpoint_object.get(
                "scaler"
            )


    else:

        state_dict = checkpoint_object


    if state_dict is None:

        raise RuntimeError(
            "No usable state_dict was found in:\n"
            f"{checkpoint_path}"
        )


    # Remove DataParallel prefix if present.
    if all(
        key.startswith("module.")
        for key in state_dict.keys()
    ):

        state_dict = {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }


    if scaler is None:

        warnings.warn(
            f"No scaler stored in {checkpoint_path}; "
            "fitting MaxAbsScaler from train_data."
        )

        scaler = MaxAbsScaler().fit(
            train_array
        )


    return (
        checkpoint_object,
        state_dict,
        scaler,
        config,
    )


# ============================================================
# 5. NETWORK RECONSTRUCTION
# ============================================================

def infer_mlp_dimensions(state_dict):

    weight_keys = []

    for key in state_dict.keys():

        if not key.endswith(
            ".weight"
        ):
            continue

        prefix = key.split(".")[0]

        if prefix.isdigit():

            weight_keys.append(
                (
                    int(prefix),
                    key,
                )
            )

    weight_keys.sort(
        key=lambda item: item[0]
    )

    if not weight_keys:

        raise ValueError(
            "Could not infer MLP dimensions from state_dict."
        )

    first_weight = state_dict[
        weight_keys[0][1]
    ]

    last_weight = state_dict[
        weight_keys[-1][1]
    ]

    input_dimension = int(
        first_weight.shape[1]
    )

    hidden_width = int(
        first_weight.shape[0]
    )

    output_dimension = int(
        last_weight.shape[0]
    )

    return (
        input_dimension,
        hidden_width,
        output_dimension,
    )


def build_network(
    dimension,
    width,
):

    return nn.Sequential(

        nn.Linear(
            dimension,
            width,
        ),
        nn.Tanh(),

        nn.Linear(
            width,
            width,
        ),
        nn.Tanh(),

        nn.Linear(
            width,
            width,
        ),
        nn.Tanh(),

        nn.Linear(
            width,
            dimension,
        ),
    )


# ============================================================
# 6. AUTONOMOUS ROLLOUT
# ============================================================

@torch.inference_mode()
def autonomous_rollout(
    network,
    scaler,
    initial_state_raw,
    number_of_steps,
    dt_value,
):

    initial_state_scaled = scaler.transform(
        np.asarray(
            initial_state_raw
        ).reshape(
            1,
            -1,
        )
    ).astype(
        np.float32
    )

    state = torch.as_tensor(
        initial_state_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )

    prediction_scaled = np.zeros(
        (
            number_of_steps + 1,
            state.shape[1],
        ),
        dtype=np.float32,
    )

    for step in range(
        number_of_steps + 1
    ):

        prediction_scaled[step] = (
            state.detach()
            .cpu()
            .numpy()[0]
        )

        state = (
            state
            + float(dt_value)
            * network(state)
        )

    prediction = scaler.inverse_transform(
        prediction_scaled
    ).astype(
        np.float64
    )

    return prediction


# ============================================================
# 7. STRUCTURE METRICS
# ============================================================

def compute_attractor_w2(
    reference,
    prediction,
    blur=0.05,
):

    reference = np.asarray(
        reference,
        dtype=np.float32,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )

    common_length = min(
        len(reference),
        len(prediction),
    )

    reference = reference[
        :common_length
    ]

    prediction = prediction[
        :common_length
    ]

    if not np.isfinite(
        prediction
    ).all():

        return float("nan")

    reference_tensor = torch.as_tensor(
        reference,
        dtype=torch.float32,
        device=DEVICE,
    )

    prediction_tensor = torch.as_tensor(
        prediction,
        dtype=torch.float32,
        device=DEVICE,
    )

    sinkhorn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=float(blur),
        backend="tensorized",
    )

    value = sinkhorn(
        reference_tensor,
        prediction_tensor,
    )

    return float(
        value.detach().cpu()
    )


def compute_psd_logl1(
    reference,
    prediction,
    dt_value,
    nfft=2048,
):

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    common_length = min(
        len(reference),
        len(prediction),
    )

    reference = reference[
        :common_length
    ]

    prediction = prediction[
        :common_length
    ]

    sampling_frequency = (
        1.0 / float(dt_value)
    )

    epsilon = 1e-12

    channel_distances = []


    for channel in range(
        reference.shape[1]
    ):

        reference_channel = (
            reference[:, channel]
            - np.mean(
                reference[:, channel]
            )
        )

        prediction_channel = (
            prediction[:, channel]
            - np.mean(
                prediction[:, channel]
            )
        )

        usable_length = min(
            int(nfft),
            common_length,
        )

        reference_channel = (
            reference_channel[
                :usable_length
            ]
        )

        prediction_channel = (
            prediction_channel[
                :usable_length
            ]
        )

        if usable_length < 8:
            continue

        _, reference_psd = welch(
            reference_channel,
            fs=sampling_frequency,
            nperseg=min(
                512,
                usable_length,
            ),
            noverlap=min(
                256,
                usable_length // 2,
            ),
        )

        _, prediction_psd = welch(
            prediction_channel,
            fs=sampling_frequency,
            nperseg=min(
                512,
                usable_length,
            ),
            noverlap=min(
                256,
                usable_length // 2,
            ),
        )

        psd_length = min(
            len(reference_psd),
            len(prediction_psd),
        )

        if psd_length == 0:
            continue

        distance = np.mean(
            np.abs(
                np.log10(
                    reference_psd[
                        :psd_length
                    ] + epsilon
                )
                -
                np.log10(
                    prediction_psd[
                        :psd_length
                    ] + epsilon
                )
            )
        )

        channel_distances.append(
            float(distance)
        )


    if not channel_distances:
        return float("nan")


    return float(
        np.mean(
            channel_distances
        )
    )


# ============================================================
# 8. EVALUATE EVERY CHECKPOINT
# ============================================================

result_rows = []
rollouts = {}


for manifest_row in checkpoint_manifest.itertuples(
    index=False
):

    seed_value = int(
        manifest_row.seed
    )

    method_name = str(
        manifest_row.method
    )

    checkpoint_path = Path(
        manifest_row.checkpoint
    )


    (
        checkpoint_object,
        state_dict,
        scaler,
        config,
    ) = load_checkpoint_components(
        checkpoint_path
    )


    (
        input_dimension,
        hidden_width,
        output_dimension,
    ) = infer_mlp_dimensions(
        state_dict
    )


    if (
        input_dimension != STATE_DIM
        or output_dimension != STATE_DIM
    ):

        raise ValueError(
            f"Dimension mismatch for seed={seed_value}, "
            f"method={method_name}: "
            f"{input_dimension}->{output_dimension}, "
            f"test dimension={STATE_DIM}"
        )


    network = build_network(
        dimension=STATE_DIM,
        width=hidden_width,
    ).to(
        DEVICE
    )

    network.load_state_dict(
        state_dict
    )

    network.eval()

    for parameter in network.parameters():
        parameter.requires_grad_(
            False
        )


    checkpoint_dt = float(
        config.get(
            "dt_obs",
            DT_OBS_DEFAULT,
        )
        if isinstance(config, dict)
        else DT_OBS_DEFAULT
    )


    print(
        f"\nEvaluating seed={seed_value}, "
        f"method={method_name}"
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )


    prediction = autonomous_rollout(
        network=network,
        scaler=scaler,
        initial_state_raw=test_array[0],
        number_of_steps=KMAX,
        dt_value=checkpoint_dt,
    )


    w2_value = compute_attractor_w2(
        reference=REFERENCE,
        prediction=prediction,
        blur=W2_BLUR,
    )


    psd_value = compute_psd_logl1(
        reference=REFERENCE,
        prediction=prediction,
        dt_value=checkpoint_dt,
        nfft=PSD_NFFT,
    )


    result_rows.append(
        {
            "seed": seed_value,
            "method": method_name,
            "Attractor_W2": w2_value,
            "PSD_logL1": psd_value,
            "checkpoint": str(
                checkpoint_path
            ),
        }
    )


    rollouts[
        (
            seed_value,
            method_name,
        )
    ] = prediction


    print(
        f"Attractor W2 = {w2_value:.6f}"
    )

    print(
        f"PSD log-L1  = {psd_value:.6f}"
    )


    del network


# ============================================================
# 9. PER-SEED TABLE
# ============================================================

results_dataframe = pd.DataFrame(
    result_rows
)

results_dataframe["method"] = pd.Categorical(
    results_dataframe["method"],
    categories=METHOD_ORDER,
    ordered=True,
)

results_dataframe = (
    results_dataframe
    .sort_values(
        [
            "seed",
            "method",
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "PER-SEED RESULTS "
    "(one full autonomous rollout; lower is better)"
)

print(
    "=" * 100
)


display(
    results_dataframe[
        [
            "seed",
            "method",
            "Attractor_W2",
            "PSD_logL1",
        ]
    ].style.format(
        {
            "Attractor_W2": "{:.6f}",
            "PSD_logL1": "{:.6f}",
        }
    )
)


# ============================================================
# 10. FIVE-SEED SUMMARY
# ============================================================

summary_rows = []


for method_name in METHOD_ORDER:

    group = results_dataframe.loc[
        results_dataframe[
            "method"
        ] == method_name
    ]

    if len(group) == 0:
        continue


    w2_values = group[
        "Attractor_W2"
    ].to_numpy(
        dtype=np.float64
    )

    psd_values = group[
        "PSD_logL1"
    ].to_numpy(
        dtype=np.float64
    )


    summary_rows.append(
        {
            "method": method_name,
            "n_seeds": int(
                len(group)
            ),
            "Attractor_W2_mean": float(
                np.nanmean(
                    w2_values
                )
            ),
            "Attractor_W2_std": float(
                np.nanstd(
                    w2_values,
                    ddof=0,
                )
            ),
            "PSD_logL1_mean": float(
                np.nanmean(
                    psd_values
                )
            ),
            "PSD_logL1_std": float(
                np.nanstd(
                    psd_values,
                    ddof=0,
                )
            ),
        }
    )


summary_dataframe = pd.DataFrame(
    summary_rows
)


print(
    "\n"
    + "=" * 100
)

print(
    "SUMMARY ACROSS FIVE SEEDS"
)

print(
    "=" * 100
)


display(
    summary_dataframe.style.format(
        {
            "Attractor_W2_mean": "{:.6f}",
            "Attractor_W2_std": "{:.6f}",
            "PSD_logL1_mean": "{:.6f}",
            "PSD_logL1_std": "{:.6f}",
        }
    )
)


print(
    "\nMarkdown summary table:\n"
)


print(
    summary_dataframe.to_markdown(
        index=False,
        floatfmt=".6f",
    )
)


# ============================================================
# 11. SAVE CSV RESULTS
# ============================================================

PER_SEED_CSV = (
    EVALUATION_DIR
    / "graph_measure_ablation_per_seed.csv"
)

SUMMARY_CSV = (
    EVALUATION_DIR
    / "graph_measure_ablation_summary.csv"
)


results_dataframe.to_csv(
    PER_SEED_CSV,
    index=False,
)

summary_dataframe.to_csv(
    SUMMARY_CSV,
    index=False,
)


print("\nSaved:")

print(PER_SEED_CSV)
print(SUMMARY_CSV)


# ============================================================
# 12. ATTRACTOR VISUALIZATION
# ============================================================

if PLOT_ATTRACTORS and STATE_DIM >= 2:

    figure, axes = plt.subplots(
        len(SEEDS),
        len(METHOD_ORDER),
        figsize=(
            4.1 * len(METHOD_ORDER),
            3.6 * len(SEEDS),
        ),
        dpi=150,
        squeeze=False,
    )


    for row_index, seed_value in enumerate(
        SEEDS
    ):

        for column_index, method_name in enumerate(
            METHOD_ORDER
        ):

            axis = axes[
                row_index,
                column_index,
            ]

            prediction = rollouts[
                (
                    seed_value,
                    method_name,
                )
            ]


            axis.plot(
                REFERENCE[:, 0],
                REFERENCE[:, 1],
                linewidth=0.8,
                alpha=0.28,
                label="Reference",
            )


            axis.plot(
                prediction[:, 0],
                prediction[:, 1],
                linewidth=0.9,
                alpha=0.92,
                label=method_name,
            )


            axis.set_title(
                f"Seed {seed_value}\n"
                f"{method_name}"
            )

            axis.set_xlabel(
                "Channel 0"
            )

            axis.set_ylabel(
                "Channel 1"
            )


            if (
                row_index == 0
                and column_index == 0
            ):

                axis.legend(
                    fontsize=8
                )


    figure.suptitle(
        "Graph-measure specificity ablation: "
        "full autonomous rollouts",
        y=1.002,
    )

    figure.tight_layout()


    FIGURE_PATH = (
        EVALUATION_DIR
        / "graph_measure_ablation_attractors.png"
    )


    figure.savefig(
        FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


    print(
        "Saved attractor figure:"
    )

    print(
        FIGURE_PATH
    )


print("\nDone.")
