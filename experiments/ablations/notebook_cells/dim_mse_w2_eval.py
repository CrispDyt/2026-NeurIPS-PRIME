# Extracted notebook code; CLI/path/device/display adaptations only.
from _notebook_bootstrap import prepare
globals().update(prepare(__file__))
# ============================================================
# ONE COMPLETE JUPYTER CELL
#
# Evaluate Attractor W2 for:
#   1-step MSE 33/DIM_MSE_BEST.pt
#   1-step MSE 123/DIM_MSE_BEST.pt
#   1-step MSE 1000/DIM_MSE_BEST.pt
#   1-step MSE 13532/DIM_MSE_BEST.pt
#   1-step MSE 123456/DIM_MSE_BEST.pt
#
# Protocol:
#   - start from test_data[0]
#   - autonomous rollout over the full test length
#   - raw-coordinate Sinkhorn W2
#   - p = 2, blur = 0.05
# ============================================================

import io
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import MaxAbsScaler
from geomloss import SamplesLoss
def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT_DIR = PORT_REPO / "checkpoints/lorenz63"

DATA_ROOT = PORT_DATA

SEEDS = [PORT_ARGS.seed] if PORT_ARGS.seed is not None else [33,123,1000,13532,123456]

CHECKPOINT_NAME = "DIM_MSE_BEST.pt"

DT_OBS_DEFAULT = 0.1
W2_BLUR = 0.05

DEVICE = torch.device(PORT_DEVICE)

OUTPUT_DIR = PORT_OUTPUT

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("device:", DEVICE)
print("output directory:", OUTPUT_DIR)


# ============================================================
# 2. LOAD DATA
# ============================================================

TRAIN_DATA_PATH = PORT_DATA / "train_data.npy"

TEST_DATA_PATH = PORT_DATA / "test_data.npy"


if "train_data" not in globals():

    if not TRAIN_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"train_data was not found:\n{TRAIN_DATA_PATH}"
        )

    train_data = np.load(
        TRAIN_DATA_PATH
    )


if "test_data" not in globals():

    if not TEST_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"test_data was not found:\n{TEST_DATA_PATH}"
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
        "train/test dimension mismatch: "
        f"{train_array.shape[1]} vs "
        f"{test_array.shape[1]}"
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

# 1000 test points -> initial point + 999 updates.
KMAX = T_EVAL - 1

REFERENCE_ATTRACTOR = test_array.copy()


print(
    f"train shape={train_array.shape}, "
    f"test shape={test_array.shape}, "
    f"state_dim={STATE_DIM}, "
    f"rollout points={T_EVAL}, "
    f"updates={KMAX}"
)


# ============================================================
# 3. CHECKPOINT MANIFEST
# ============================================================

checkpoint_manifest = {}


for seed_value in SEEDS:

    checkpoint_path = (
        ROOT_DIR
        / f"seed_{seed_value}"
        / CHECKPOINT_NAME
    )

    if not checkpoint_path.is_file():

        seed_folder = (
            checkpoint_path.parent
        )

        available_files = (
            sorted(
                path.name
                for path in seed_folder.glob("*")
            )
            if seed_folder.exists()
            else []
        )

        raise FileNotFoundError(
            f"\n{CHECKPOINT_NAME} was not found for "
            f"seed {seed_value}:\n"
            f"{checkpoint_path}\n\n"
            "Available files:\n"
            + (
                "\n".join(available_files)
                if available_files
                else "<none>"
            )
        )

    checkpoint_manifest[
        seed_value
    ] = checkpoint_path.resolve()


print("\n" + "=" * 95)
print("DIM + MSE checkpoint manifest")
print("=" * 95)

for seed_value, checkpoint_path in checkpoint_manifest.items():

    print(
        f"seed {seed_value:>6}: "
        f"{checkpoint_path}"
    )

print("=" * 95)


# ============================================================
# 4. CPU-SAFE CHECKPOINT LOADER
# ============================================================

class CPUUnpickler(
    pickle.Unpickler
):

    def find_class(
        self,
        module,
        name,
    ):

        if (
            module == "torch.storage"
            and name == "_load_from_bytes"
        ):

            return lambda byte_data: torch.load(
                io.BytesIO(byte_data),
                map_location="cpu",
                weights_only=False,
            )

        return super().find_class(
            module,
            name,
        )


def load_pickle_cpu(
    checkpoint_path,
):

    with Path(
        checkpoint_path
    ).open(
        "rb"
    ) as handle:

        return CPUUnpickler(
            handle
        ).load()


def smart_load(
    checkpoint_path,
):

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


def looks_like_state_dict(
    obj,
):

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


def extract_checkpoint_information(
    checkpoint_path,
):

    checkpoint_object = smart_load(
        checkpoint_path
    )

    state_dict = None
    scaler = None
    config = {}


    if isinstance(
        checkpoint_object,
        nn.Module,
    ):

        state_dict = (
            checkpoint_object.state_dict()
        )


    elif looks_like_state_dict(
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
            nn.Module,
        ):

            state_dict = (
                model_block.state_dict()
            )


        elif isinstance(
            model_block,
            dict,
        ):

            if looks_like_state_dict(
                model_block
            ):

                state_dict = model_block

            else:

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


    if state_dict is None:
        raise RuntimeError(
            "No usable state_dict was found in:\n"
            f"{checkpoint_path}"
        )


    # Remove DataParallel prefix.
    if (
        len(state_dict) > 0
        and all(
            key.startswith("module.")
            for key in state_dict.keys()
        )
    ):

        state_dict = {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }


    if scaler is None:

        print(
            f"[WARNING] No scaler stored in:\n"
            f"{checkpoint_path}\n"
            "Using MaxAbsScaler fitted on train_data."
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
# 5. REBUILD NETWORK
# ============================================================

def rebuild_network_from_state_dict(
    original_state_dict,
):

    detected_layers = []


    for key, tensor in original_state_dict.items():

        if (
            not key.endswith(".weight")
            or not torch.is_tensor(tensor)
            or tensor.ndim != 2
        ):
            continue

        match = re.search(
            r"(?:^|\.)(\d+)\.weight$",
            key,
        )

        if match is not None:

            detected_layers.append(
                (
                    int(match.group(1)),
                    key,
                )
            )


    detected_layers.sort(
        key=lambda item: item[0]
    )


    if len(detected_layers) == 0:
        raise ValueError(
            "Could not detect linear layers "
            "from checkpoint state_dict."
        )


    modules = []
    remapped_state_dict = {}


    for layer_position, (
        _,
        weight_key,
    ) in enumerate(
        detected_layers
    ):

        weight = original_state_dict[
            weight_key
        ]

        input_features = int(
            weight.shape[1]
        )

        output_features = int(
            weight.shape[0]
        )

        new_module_index = (
            2 * layer_position
        )


        modules.append(
            nn.Linear(
                input_features,
                output_features,
            )
        )


        remapped_state_dict[
            f"{new_module_index}.weight"
        ] = weight


        bias_key = (
            weight_key[
                :-len("weight")
            ]
            + "bias"
        )

        if bias_key not in original_state_dict:
            raise KeyError(
                f"Bias corresponding to {weight_key} "
                f"was not found:\n{bias_key}"
            )


        remapped_state_dict[
            f"{new_module_index}.bias"
        ] = original_state_dict[
            bias_key
        ]


        if (
            layer_position
            < len(detected_layers) - 1
        ):

            modules.append(
                nn.Tanh()
            )


    network = nn.Sequential(
        *modules
    )

    network.load_state_dict(
        remapped_state_dict,
        strict=True,
    )


    input_dimension = int(
        original_state_dict[
            detected_layers[0][1]
        ].shape[1]
    )

    output_dimension = int(
        original_state_dict[
            detected_layers[-1][1]
        ].shape[0]
    )


    return (
        network,
        input_dimension,
        output_dimension,
    )


# ============================================================
# 6. AUTONOMOUS ROLLOUT
# ============================================================

@torch.inference_mode()
def autonomous_rollout(
    network,
    scaler,
    initial_state_raw,
    number_of_updates,
    dt_value,
):

    initial_scaled = scaler.transform(
        np.asarray(
            initial_state_raw,
            dtype=np.float64,
        ).reshape(
            1,
            -1,
        )
    ).astype(
        np.float32
    )


    state = torch.as_tensor(
        initial_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )


    prediction_scaled = np.zeros(
        (
            number_of_updates + 1,
            state.shape[1],
        ),
        dtype=np.float32,
    )


    for step in range(
        number_of_updates + 1
    ):

        prediction_scaled[step] = (
            state.detach()
            .cpu()
            .numpy()[0]
        )


        if step == number_of_updates:
            break


        state = (
            state
            + float(dt_value)
            * network(state)
        )


    prediction_raw = scaler.inverse_transform(
        prediction_scaled
    ).astype(
        np.float64
    )


    return prediction_raw


# ============================================================
# 7. ATTRACTOR W2
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


    sinkhorn_loss = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=float(blur),
        backend="tensorized",
    )


    value = sinkhorn_loss(
        reference_tensor,
        prediction_tensor,
    )


    return float(
        value.detach().cpu()
    )


# ============================================================
# 8. EVALUATE FIVE DIM + MSE CHECKPOINTS
# ============================================================

result_rows = []


for seed_value in SEEDS:

    checkpoint_path = checkpoint_manifest[
        seed_value
    ]


    (
        checkpoint_object,
        state_dict,
        scaler,
        config,
    ) = extract_checkpoint_information(
        checkpoint_path
    )


    (
        network,
        input_dimension,
        output_dimension,
    ) = rebuild_network_from_state_dict(
        state_dict
    )


    if (
        input_dimension != STATE_DIM
        or output_dimension != STATE_DIM
    ):

        raise ValueError(
            f"Seed {seed_value}: dimension mismatch. "
            f"Checkpoint={input_dimension}->{output_dimension}, "
            f"test dimension={STATE_DIM}."
        )


    network = network.to(
        DEVICE
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
        "\n"
        + "-" * 95
    )

    print(
        f"Evaluating DIM + MSE, seed {seed_value}"
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )


    prediction = autonomous_rollout(
        network=network,
        scaler=scaler,
        initial_state_raw=test_array[0],
        number_of_updates=KMAX,
        dt_value=checkpoint_dt,
    )


    w2_value = compute_attractor_w2(
        reference=REFERENCE_ATTRACTOR,
        prediction=prediction,
        blur=W2_BLUR,
    )


    print(
        f"Attractor W2 = {w2_value:.6f}"
    )


    result_rows.append(
        {
            "Method": "DIM + MSE",
            "Seed": int(seed_value),
            "Attractor_W2": float(
                w2_value
            ),
            "Rollout_points": int(
                len(prediction)
            ),
            "Checkpoint": str(
                checkpoint_path
            ),
        }
    )


    del network


# ============================================================
# 9. RESULTS
# ============================================================

results_dataframe = pd.DataFrame(
    result_rows
)


print(
    "\n"
    + "=" * 95
)

print(
    "DIM + MSE Attractor W2 results "
    "(lower is better)"
)

print(
    "=" * 95
)


display(
    results_dataframe[
        [
            "Method",
            "Seed",
            "Attractor_W2",
            "Rollout_points",
        ]
    ].style.format(
        {
            "Attractor_W2": "{:.6f}",
        }
    )
)


# ============================================================
# 10. SUMMARY
# ============================================================

w2_values = results_dataframe[
    "Attractor_W2"
].to_numpy(
    dtype=np.float64
)

w2_values = w2_values[
    np.isfinite(
        w2_values
    )
]


mean_w2 = float(
    np.mean(w2_values)
)

std_w2 = float(
    np.std(
        w2_values,
        ddof=0,
    )
)

median_w2 = float(
    np.median(w2_values)
)


print(
    f"\nMean ± STD: "
    f"{mean_w2:.6f} ± {std_w2:.6f}"
)

print(
    f"Median: "
    f"{median_w2:.6f}"
)


summary_dataframe = pd.DataFrame(
    [
        {
            "Method": "DIM + MSE",
            "N_seeds": int(
                len(w2_values)
            ),
            "Attractor_W2_mean": mean_w2,
            "Attractor_W2_std": std_w2,
            "Attractor_W2_median": median_w2,
        }
    ]
)


print(
    "\nMarkdown table:\n"
)

print(
    results_dataframe[
        [
            "Method",
            "Seed",
            "Attractor_W2",
        ]
    ].to_markdown(
        index=False,
        floatfmt=".6f",
    )
)


# ============================================================
# 11. SAVE CSV
# ============================================================

PER_SEED_CSV = (
    OUTPUT_DIR
    / "DIM_MSE_Attractor_W2_per_seed.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "DIM_MSE_Attractor_W2_summary.csv"
)


results_dataframe.to_csv(
    PER_SEED_CSV,
    index=False,
)

summary_dataframe.to_csv(
    SUMMARY_CSV,
    index=False,
)


print(
    "\nSaved:"
)

print(
    PER_SEED_CSV
)

print(
    SUMMARY_CSV
)

print(
    "\nDone."
)
