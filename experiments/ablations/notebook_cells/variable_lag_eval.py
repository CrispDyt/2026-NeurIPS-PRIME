# Extracted notebook code; CLI/path/device/display adaptations only.
from _notebook_bootstrap import prepare
globals().update(prepare(__file__))
# ============================================================
# ONE COMPLETE JUPYTER CELL
#
# Compare only:
#   1. PRIME
#   2. Sequential empirical-K
#   3. Empirical variable-K
#
# Metrics:
#   - Attractor W2
#   - PSD log-L1
#
# Required variables from the data-loading cell:
#   train_data, test_data, dt_obs
# ============================================================

import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.signal import welch
from sklearn.preprocessing import MaxAbsScaler
from geomloss import SamplesLoss


# ============================================================
# 1. CHECK REQUIRED DATA
# ============================================================

required_variables = (
    "train_data",
    "test_data",
    "dt_obs",
)

missing_variables = [
    name
    for name in required_variables
    if name not in globals()
]

if missing_variables:
    raise RuntimeError(
        "Run the data-loading cell first. Missing variables: "
        + ", ".join(missing_variables)
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
        f"train_data must have shape (T,d), got {train_array.shape}"
    )

if test_array.ndim != 2:
    raise ValueError(
        f"test_data must have shape (T,d), got {test_array.shape}"
    )

if train_array.shape[1] != test_array.shape[1]:
    raise ValueError(
        "train_data and test_data have different dimensions: "
        f"{train_array.shape[1]} vs {test_array.shape[1]}"
    )

if not np.all(np.isfinite(train_array)):
    raise ValueError(
        "train_data contains NaN or Inf"
    )

if not np.all(np.isfinite(test_array)):
    raise ValueError(
        "test_data contains NaN or Inf"
    )


# ============================================================
# 2. EXACT CHECKPOINT PATHS
# ============================================================

ROOT_DIR = PORT_CHECKPOINTS / "notebook"


# PRIME checkpoint used in your previous W2/PSD comparison.
PRIME_CKPT_PATH = (
    ROOT_DIR
    / "1-step MSE 123"
    / "Implicit_NoWarmUp_BEST.p"
)


SEQUENTIAL_K_CKPT_PATH = (
    ROOT_DIR
    / "OrdinaryStartSequentialEmpiricalK_seed123"
    / "OrdinaryStartSequentialEmpiricalK_BEST_20260728_012801.p"
)


EMPIRICAL_VARIABLE_K_CKPT_PATH = (
    ROOT_DIR
    / "EmpiricalVariableLag_seed123456"
    / "EmpiricalVariableLag_BEST_20260726_235526.p"
)


MODEL_CKPTS = {
    "PRIME": PRIME_CKPT_PATH,
    "Sequential empirical-K": SEQUENTIAL_K_CKPT_PATH,
    "Empirical variable-K": EMPIRICAL_VARIABLE_K_CKPT_PATH,
}


def resolve_checkpoint(
    model_name,
    checkpoint_path,
):
    """
    Check that a checkpoint exists.

    If it does not exist, print all files in its parent folder.
    """

    checkpoint_path = (
        Path(checkpoint_path)
        .expanduser()
    )

    if checkpoint_path.is_file():
        return checkpoint_path.resolve()

    parent_directory = checkpoint_path.parent

    available_files = (
        sorted(
            path.name
            for path in parent_directory.glob("*")
        )
        if parent_directory.exists()
        else []
    )

    raise FileNotFoundError(
        f"\n{model_name} checkpoint was not found:\n"
        f"{checkpoint_path}\n\n"
        f"Parent directory exists: "
        f"{parent_directory.exists()}\n\n"
        "Files currently inside the parent directory:\n"
        + (
            "\n".join(available_files)
            if available_files
            else "<none>"
        )
    )


for model_name in list(MODEL_CKPTS):

    MODEL_CKPTS[model_name] = (
        resolve_checkpoint(
            model_name=model_name,
            checkpoint_path=MODEL_CKPTS[model_name],
        )
    )


print("=" * 90)
print("Checkpoint manifest")
print("=" * 90)

for model_name, checkpoint_path in MODEL_CKPTS.items():

    print(f"{model_name}:")
    print(checkpoint_path)
    print()

print("=" * 90)


# ============================================================
# 3. EVALUATION CONFIGURATION
# ============================================================

# Force CPU because your CUDA device was busy/unavailable.
DEVICE = torch.device(PORT_DEVICE)

W2_BLUR = 0.05
PSD_NFFT = 2048

T_eval = int(
    len(test_array)
)

state_dim = int(
    test_array.shape[1]
)

dt = float(
    dt_obs
)

# One long autonomous rollout from test_data[0].
reference_long = test_array[
    :T_eval
].copy()

number_of_rollout_steps = (
    T_eval - 1
)


print("device:", DEVICE)

print(
    f"T_eval={T_eval}, "
    f"state_dim={state_dim}, "
    f"dt_obs={dt}, "
    f"rollout_steps={number_of_rollout_steps}"
)


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

    with open(
        checkpoint_path,
        "rb",
    ) as file_handle:

        return CPUUnpickler(
            file_handle
        ).load()


def smart_load(
    checkpoint_path,
):
    """
    Try torch.load first, then CPU-safe pickle.
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


def extract_state_dict_and_scaler(
    checkpoint_path,
):
    """
    Extract:
      - checkpoint object
      - model state_dict
      - scaler
    """

    checkpoint_object = smart_load(
        checkpoint_path
    )

    state_dict = None
    scaler = None


    if isinstance(
        checkpoint_object,
        dict,
    ):

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
            f"[WARNING] No scaler stored in:\n"
            f"{checkpoint_path}\n"
            "Fitting MaxAbsScaler from train_data."
        )

        scaler = MaxAbsScaler().fit(
            train_array
        )


    return (
        checkpoint_object,
        state_dict,
        scaler,
    )


# ============================================================
# 5. REBUILD NETWORK
# ============================================================

def infer_mlp_dimensions(
    state_dict,
):
    """
    Infer input dimension, hidden width, and output dimension
    from the Sequential MLP state_dict.
    """

    weight_keys = []

    for key in state_dict:

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


    if not weight_keys:

        raise ValueError(
            "Could not infer MLP dimensions "
            "from state_dict."
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

    network_width = int(
        first_weight.shape[0]
    )

    output_dimension = int(
        last_weight.shape[0]
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
):
    """
    Autonomous Euler rollout:

        x_{t+1} = x_t + dt_obs * network(x_t)
    """

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


    current_state = torch.as_tensor(
        initial_state_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )


    prediction_scaled = np.zeros(
        (
            number_of_steps + 1,
            current_state.shape[1],
        ),
        dtype=np.float32,
    )


    for step in range(
        number_of_steps + 1
    ):

        prediction_scaled[step] = (
            current_state
            .detach()
            .cpu()
            .numpy()[0]
        )


        current_state = (
            current_state
            + dt * network(
                current_state
            )
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
    """
    Sinkhorn distance between the raw-coordinate point clouds.
    Lower is better.
    """

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


# ============================================================
# 8. PSD LOG-L1
# ============================================================

def compute_psd_log_l1(
    reference,
    prediction,
    dt_value,
    nfft=2048,
):
    """
    Mean absolute difference between log10 PSD values,
    averaged across all channels.

    Lower is better.
    """

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

    usable_length = min(
        int(nfft),
        common_length,
    )

    epsilon = 1e-12

    channel_distances = []


    for channel in range(
        reference.shape[1]
    ):

        reference_channel = (
            reference[
                :usable_length,
                channel
            ]
            -
            np.mean(
                reference[
                    :usable_length,
                    channel
                ]
            )
        )


        prediction_channel = (
            prediction[
                :usable_length,
                channel
            ]
            -
            np.mean(
                prediction[
                    :usable_length,
                    channel
                ]
            )
        )


        if usable_length < 8:

            channel_distances.append(
                np.nan
            )

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


        distance = np.mean(

            np.abs(

                np.log10(
                    reference_psd[
                        :psd_length
                    ]
                    + epsilon
                )

                -

                np.log10(
                    prediction_psd[
                        :psd_length
                    ]
                    + epsilon
                )
            )
        )


        channel_distances.append(
            float(distance)
        )


    channel_distances = np.asarray(
        channel_distances,
        dtype=np.float64,
    )


    channel_distances = channel_distances[
        np.isfinite(
            channel_distances
        )
    ]


    if len(channel_distances) == 0:

        return float("nan")


    return float(
        np.mean(
            channel_distances
        )
    )


# ============================================================
# 9. LOAD AND EVALUATE THE THREE MODELS
# ============================================================

results = []

long_rollouts = {}


for model_name, checkpoint_path in MODEL_CKPTS.items():

    (
        checkpoint_object,
        state_dict,
        scaler,
    ) = extract_state_dict_and_scaler(
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
        input_dimension != state_dim
        or output_dimension != state_dim
    ):

        raise ValueError(
            f"{model_name} dimension mismatch: "
            f"input={input_dimension}, "
            f"output={output_dimension}, "
            f"test dimension={state_dim}"
        )


    network = build_network(
        dimension=state_dim,
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


    config = (
        checkpoint_object.get(
            "config",
            {},
        )
        if isinstance(
            checkpoint_object,
            dict,
        )
        else {}
    )


    print(
        "\n"
        + "-" * 90
    )

    print(
        "Evaluating:",
        model_name,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Stored seed:",
        config.get("seed"),
    )

    print(
        "Stored method:",
        config.get("method"),
    )


    prediction = autonomous_rollout(

        network=network,

        scaler=scaler,

        initial_state_raw=test_array[0],

        number_of_steps=number_of_rollout_steps,
    )


    w2_value = compute_attractor_w2(

        reference=reference_long,

        prediction=prediction,

        blur=W2_BLUR,
    )


    psd_value = compute_psd_log_l1(

        reference=reference_long,

        prediction=prediction,

        dt_value=dt,

        nfft=PSD_NFFT,
    )


    results.append(
        {
            "Model": model_name,
            "Attractor_W2": w2_value,
            "PSD_logL1": psd_value,
        }
    )


    long_rollouts[
        model_name
    ] = prediction


    print(
        f"Attractor W2: "
        f"{w2_value:.6f}"
    )

    print(
        f"PSD log-L1:  "
        f"{psd_value:.6f}"
    )


# ============================================================
# 10. FINAL TABLE
# ============================================================

results_dataframe = pd.DataFrame(
    results
)


print(
    "\n"
    "==================== "
    "STRUCTURE METRICS "
    "(1 Long Autonomous Rollout; lower is better) "
    "===================="
)


display(
    results_dataframe.style.format(
        {
            "Attractor_W2": "{:.6f}",
            "PSD_logL1": "{:.6f}",
        }
    )
)


print(
    "\nMarkdown table:\n"
)


print(
    results_dataframe.to_markdown(
        index=False,
        floatfmt=".6f",
    )
)


# ============================================================
# 11. SAVE CSV
# ============================================================

RESULT_CSV = PORT_OUTPUT / "PRIME_SequentialK_EmpiricalVariableK_W2_PSD.csv"


results_dataframe.to_csv(
    RESULT_CSV,
    index=False,
)


print(
    "\nSaved CSV:"
)

print(
    RESULT_CSV
)


print(
    "\nDone."
)
