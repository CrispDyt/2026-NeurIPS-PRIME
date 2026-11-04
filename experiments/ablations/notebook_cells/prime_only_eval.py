# Extracted notebook code; CLI/path/device/display adaptations only.
from _notebook_bootstrap import prepare
globals().update(prepare(__file__))
# ============================================================
# ONE COMPLETE JUPYTER CELL
#
# Evaluate:
#   DIM warm-up -> PRIME-only
#   seeds = [33, 123, 123456, 1000, 13532]
#
# Metric:
#   Attractor W2 only
#
# Checkpoint selection:
#   BEST_PRIME checkpoint recorded in each run_summary.json
# ============================================================

import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import MaxAbsScaler
from geomloss import SamplesLoss


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

SEEDS = [PORT_ARGS.seed] if PORT_ARGS.seed is not None else [33,123,123456,1000,13532]

OUTPUT_ROOT = PORT_CHECKPOINTS / "notebook/Ablation_DIMWarmup_then_PRIMEOnly"

# Same data-loading convention as the training notebook.
TRAIN_DATA_PATH = PORT_DATA / "train_data.npy"
TEST_DATA_PATH = PORT_DATA / "test_data.npy"

DT_OBS = 0.1
W2_BLUR = 0.05

# Force CPU to avoid CUDA-busy errors during evaluation.
DEVICE = torch.device(PORT_DEVICE)

print("device:", DEVICE)
print("output root:", OUTPUT_ROOT)


# ============================================================
# 2. LOAD DATA
# ============================================================

if not TRAIN_DATA_PATH.is_file():
    raise FileNotFoundError(
        f"train_data was not found:\n"
        f"{TRAIN_DATA_PATH.resolve()}"
    )

if not TEST_DATA_PATH.is_file():
    raise FileNotFoundError(
        f"test_data was not found:\n"
        f"{TEST_DATA_PATH.resolve()}"
    )

train_data = np.load(
    TRAIN_DATA_PATH
).astype(
    np.float64
)

test_data = np.load(
    TEST_DATA_PATH
).astype(
    np.float64
)

if train_data.ndim != 2:
    raise ValueError(
        f"train_data must have shape (T,d), got {train_data.shape}"
    )

if test_data.ndim != 2:
    raise ValueError(
        f"test_data must have shape (T,d), got {test_data.shape}"
    )

if train_data.shape[1] != test_data.shape[1]:
    raise ValueError(
        "train_data and test_data dimensions differ: "
        f"{train_data.shape[1]} vs {test_data.shape[1]}"
    )

if not np.all(np.isfinite(train_data)):
    raise ValueError(
        "train_data contains NaN or Inf"
    )

if not np.all(np.isfinite(test_data)):
    raise ValueError(
        "test_data contains NaN or Inf"
    )

T_EVAL = int(
    len(test_data)
)

STATE_DIM = int(
    test_data.shape[1]
)

KMAX = T_EVAL - 1

REFERENCE_ATTRACTOR = test_data.copy()

print(
    f"T_eval={T_EVAL}, "
    f"state_dim={STATE_DIM}, "
    f"dt_obs={DT_OBS}"
)


# ============================================================
# 3. FIND EACH SEED'S BEST_PRIME CHECKPOINT
# ============================================================

def find_best_checkpoint(
    seed_value,
):
    """
    First read best_checkpoint from run_summary.json.

    If run_summary.json is unavailable, fall back to the most
    recently modified BEST_PRIME checkpoint in the seed folder.
    """

    seed_folder = (
        OUTPUT_ROOT
        / f"seed_{seed_value}"
    )

    if not seed_folder.is_dir():
        raise FileNotFoundError(
            f"Seed folder was not found:\n"
            f"{seed_folder}"
        )

    summary_path = (
        seed_folder
        / "run_summary.json"
    )

    if summary_path.is_file():

        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            summary = json.load(
                handle
            )

        checkpoint_string = summary.get(
            "best_checkpoint"
        )

        if checkpoint_string:

            checkpoint_path = Path(
                checkpoint_string
            ).expanduser()

            # Handle a relative path if one was saved.
            if not checkpoint_path.is_absolute():
                checkpoint_path = (
                    seed_folder
                    / checkpoint_path
                )

            if checkpoint_path.is_file():
                return (
                    checkpoint_path.resolve(),
                    summary,
                )

            print(
                f"[WARNING] Seed {seed_value}: "
                f"run_summary.json points to a missing file:\n"
                f"{checkpoint_path}\n"
                "Falling back to folder discovery."
            )

    candidates = sorted(
        seed_folder.glob(
            f"DIMWarmup_PRIMEOnly_seed"
            f"{seed_value}_BEST_PRIME_*.p"
        ),
        key=lambda path: path.stat().st_mtime,
    )

    if len(candidates) == 0:

        available_files = sorted(
            path.name
            for path in seed_folder.glob("*")
        )

        raise FileNotFoundError(
            f"No BEST_PRIME checkpoint was found for "
            f"seed {seed_value} inside:\n"
            f"{seed_folder}\n\n"
            "Available files:\n"
            + "\n".join(available_files)
        )

    if len(candidates) > 1:

        print(
            f"[WARNING] Seed {seed_value}: "
            f"multiple BEST_PRIME checkpoints found."
        )

        for candidate in candidates:
            print("   ", candidate.name)

        print(
            "Using the most recently modified checkpoint:",
            candidates[-1].name,
        )

    return (
        candidates[-1].resolve(),
        {},
    )


checkpoint_manifest = {}

print("\n" + "=" * 90)
print("BEST_PRIME checkpoint manifest")
print("=" * 90)

for seed_value in SEEDS:

    checkpoint_path, run_summary = (
        find_best_checkpoint(
            seed_value
        )
    )

    checkpoint_manifest[
        seed_value
    ] = {
        "path": checkpoint_path,
        "run_summary": run_summary,
    }

    print(
        f"seed {seed_value:>6}: "
        f"{checkpoint_path}"
    )

print("=" * 90)


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

    with checkpoint_path.open(
        "rb"
    ) as handle:

        return CPUUnpickler(
            handle
        ).load()


def smart_load(
    checkpoint_path,
):
    """
    Try torch.load first, then CPU-safe pickle loading.
    """

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


def extract_model_information(
    checkpoint_path,
):
    """
    Return:
        checkpoint_object,
        state_dict,
        scaler,
        config
    """

    checkpoint_object = smart_load(
        checkpoint_path
    )

    state_dict = None
    scaler = None
    config = {}

    if isinstance(
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
            "No state_dict was found in checkpoint:\n"
            f"{checkpoint_path}"
        )

    if scaler is None:

        print(
            f"[WARNING] No scaler was stored in:\n"
            f"{checkpoint_path}\n"
            "Fitting MaxAbsScaler from train_data."
        )

        scaler = MaxAbsScaler().fit(
            train_data
        )

    return (
        checkpoint_object,
        state_dict,
        scaler,
        config,
    )


# ============================================================
# 5. REBUILD THE MLP
# ============================================================

def infer_mlp_dimensions(
    state_dict,
):

    weight_keys = []

    for key in state_dict.keys():

        if not key.endswith(
            ".weight"
        ):
            continue

        prefix = key.split(
            "."
        )[0]

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

    if len(weight_keys) == 0:

        raise ValueError(
            "Could not infer MLP dimensions "
            "from the checkpoint state_dict."
        )

    first_weight = state_dict[
        weight_keys[0][1]
    ]

    final_weight = state_dict[
        weight_keys[-1][1]
    ]

    input_dimension = int(
        first_weight.shape[1]
    )

    network_width = int(
        first_weight.shape[0]
    )

    output_dimension = int(
        final_weight.shape[0]
    )

    return (
        input_dimension,
        network_width,
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
# 6. LONG AUTONOMOUS ROLLOUT
# ============================================================

@torch.no_grad()
def autonomous_rollout(
    network,
    scaler,
    initial_state_raw,
    number_of_steps,
    dt_value,
):

    initial_state_raw = np.asarray(
        initial_state_raw,
        dtype=np.float64,
    ).reshape(
        1,
        -1,
    )

    initial_state_scaled = scaler.transform(
        initial_state_raw
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
# 7. ATTRACTOR W2
# ============================================================

def attractor_w2(
    reference,
    prediction,
    blur,
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

    if not np.all(
        np.isfinite(prediction)
    ):

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
# 8. EVALUATE ALL FIVE BEST_PRIME MODELS
# ============================================================

results = []

for seed_value in SEEDS:

    checkpoint_path = (
        checkpoint_manifest[
            seed_value
        ][
            "path"
        ]
    )

    (
        checkpoint_object,
        state_dict,
        scaler,
        config,
    ) = extract_model_information(
        checkpoint_path
    )

    (
        input_dimension,
        network_width,
        output_dimension,
    ) = infer_mlp_dimensions(
        state_dict
    )

    if (
        input_dimension != STATE_DIM
        or output_dimension != STATE_DIM
    ):

        raise ValueError(
            f"Seed {seed_value}: model dimension mismatch. "
            f"Checkpoint input={input_dimension}, "
            f"output={output_dimension}, "
            f"test dimension={STATE_DIM}"
        )

    network = build_network(
        dimension=STATE_DIM,
        width=network_width,
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
            DT_OBS,
        )
    )

    print(
        "\n"
        + "-" * 90
    )

    print(
        f"Evaluating seed {seed_value}"
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Method:",
        config.get("method"),
    )

    print(
        "DIM warm-up steps:",
        config.get("DIM_warmup_steps"),
    )

    print(
        "PRIME-only steps:",
        config.get("PRIME_only_steps"),
    )

    prediction = autonomous_rollout(
        network=network,
        scaler=scaler,
        initial_state_raw=test_data[0],
        number_of_steps=KMAX,
        dt_value=checkpoint_dt,
    )

    w2_value = attractor_w2(
        reference=REFERENCE_ATTRACTOR,
        prediction=prediction,
        blur=W2_BLUR,
    )

    print(
        f"Attractor W2 = "
        f"{w2_value:.6f}"
    )

    results.append(
        {
            "seed": int(seed_value),
            "Attractor_W2": float(
                w2_value
            ),
            "best_prime_step": config.get(
                "best_prime_step_pre_update",
                checkpoint_manifest[
                    seed_value
                ][
                    "run_summary"
                ].get(
                    "best_prime_step"
                ),
            ),
            "checkpoint": str(
                checkpoint_path
            ),
        }
    )

    del network


# ============================================================
# 9. RESULT TABLE AND SUMMARY
# ============================================================

results_dataframe = pd.DataFrame(
    results
)

print(
    "\n"
    + "=" * 90
)

print(
    "Attractor W2 results "
    "(lower is better)"
)

print(
    "=" * 90
)

display(
    results_dataframe[
        [
            "seed",
            "Attractor_W2",
            "best_prime_step",
        ]
    ].style.format(
        {
            "Attractor_W2": "{:.6f}",
        }
    )
)

finite_w2 = results_dataframe[
    "Attractor_W2"
].to_numpy(
    dtype=np.float64
)

finite_w2 = finite_w2[
    np.isfinite(
        finite_w2
    )
]

if len(finite_w2) == 0:

    mean_w2 = float("nan")
    std_w2 = float("nan")
    median_w2 = float("nan")

else:

    mean_w2 = float(
        np.mean(finite_w2)
    )

    std_w2 = float(
        np.std(
            finite_w2,
            ddof=0,
        )
    )

    median_w2 = float(
        np.median(finite_w2)
    )

print(
    f"\nMean Attractor W2:   "
    f"{mean_w2:.6f}"
)

print(
    f"STD Attractor W2:    "
    f"{std_w2:.6f}"
)

print(
    f"Median Attractor W2: "
    f"{median_w2:.6f}"
)

print(
    f"\nMean ± STD: "
    f"{mean_w2:.6f} ± {std_w2:.6f}"
)


# ============================================================
# 10. SAVE CSV
# ============================================================

RESULT_CSV = (
    PORT_OUTPUT
    / "BEST_PRIME_Attractor_W2_all_seeds.csv"
)

results_dataframe.to_csv(
    RESULT_CSV,
    index=False,
)

print(
    "\nSaved results:"
)

print(
    RESULT_CSV
)

print(
    "\nDone."
)
