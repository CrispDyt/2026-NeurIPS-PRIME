# Extracted notebook code; CLI/path/device/display adaptations only.
from _notebook_bootstrap import prepare
globals().update(prepare(__file__))
# Dependency functions from Lorenz.ipynb cell 11, unchanged.
import io

import pickle

import numpy as np

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

from scipy.signal import welch

from sklearn.preprocessing import MaxAbsScaler

from geomloss import SamplesLoss

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)

def load_p_cpu(path):
    with open(path, "rb") as f:
        return CPU_Unpickler(f).load()

def smart_load(path):
    """
    .pt / torch.save(...) -> torch.load
    .p  / pickle.dump(...) -> CPU-safe pickle fallback
    """
    last_err = None

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        last_err = e

    try:
        return load_p_cpu(path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint: {path}\n"
            f"torch.load error: {last_err}\n"
            f"pickle error: {e}"
        )

def psd_logl1_distance(ref, pred, dt, nfft=2048, channels=None):
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    L = min(len(ref), len(pred))
    ref = ref[:L]
    pred = pred[:L]

    if channels is None:
        channels = list(range(ref.shape[1]))

    fs = 1.0 / float(dt)
    eps = 1e-12
    dists = []

    for ch in channels:
        x = ref[:, ch] - np.mean(ref[:, ch])
        y = pred[:, ch] - np.mean(pred[:, ch])

        n = min(nfft, L)
        x = x[:n]
        y = y[:n]

        if len(x) < 8 or len(y) < 8:
            dists.append(np.nan)
            continue

        _, Pxx = welch(x, fs=fs, nperseg=min(512, n), noverlap=min(256, n // 2))
        _, Pyy = welch(y, fs=fs, nperseg=min(512, n), noverlap=min(256, n // 2))

        mlen = min(len(Pxx), len(Pyy))
        if mlen == 0:
            dists.append(np.nan)
            continue

        Pxx = Pxx[:mlen] + eps
        Pyy = Pyy[:mlen] + eps
        dists.append(float(np.mean(np.abs(np.log10(Pxx) - np.log10(Pyy)))))

    dists = np.asarray(dists, dtype=np.float64)
    dists = dists[np.isfinite(dists)]
    return float(np.mean(dists)) if len(dists) else float("nan")

def sinkhorn_w2_cloud(A, B, blur=0.05, device="cpu"):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    XA = torch.tensor(A, dtype=torch.float32, device=device)
    XB = torch.tensor(B, dtype=torch.float32, device=device)
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=float(blur), backend="tensorized")
    return float(loss(XA, XB).detach().cpu())

def infer_mlp_dims_from_state_dict(sd: dict):
    wkeys = []
    for k in sd.keys():
        if k.endswith(".weight"):
            p = k.split(".")[0]
            if p.isdigit():
                wkeys.append((int(p), k))
    wkeys = sorted(wkeys, key=lambda x: x[0])
    if len(wkeys) == 0:
        raise ValueError("Could not infer MLP dims from state_dict.")
    W0 = sd[wkeys[0][1]]
    Wlast = sd[wkeys[-1][1]]
    width = int(W0.shape[0])
    d_in  = int(W0.shape[1])
    d_out = int(Wlast.shape[0])
    return d_in, width, d_out

def build_net(d, width=100):
    return nn.Sequential(
        nn.Linear(d, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(),
        nn.Linear(width, d),
    )

def load_model_ckpt(path):
    obj = smart_load(path)

    sd = None
    scaler = None

    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"].get("state_dict", None)
            scaler = obj["model"].get("transformer", obj["model"].get("scaler", None))

        if sd is None and "state_dict" in obj:
            sd = obj["state_dict"]

        if sd is None and "model_state_dict" in obj:
            sd = obj["model_state_dict"]

        if scaler is None and "transformer" in obj:
            scaler = obj["transformer"]
        if scaler is None and "scaler" in obj:
            scaler = obj["scaler"]
    else:
        sd = obj

    if sd is None:
        raise RuntimeError(f"Could not find usable state_dict in checkpoint: {path}")

    if scaler is None:
        scaler = MaxAbsScaler().fit(np.asarray(train_data, dtype=np.float64))

    return obj, sd, scaler

def rollout_from_x0(net, scaler, x0_raw, K, dt_obs, device):
    x0_s = scaler.transform(np.asarray(x0_raw).reshape(1, -1)).astype(np.float32)
    x = torch.tensor(x0_s, dtype=torch.float32, device=device)

    pred_s = np.zeros((K + 1, x.shape[1]), dtype=np.float32)
    for k in range(K + 1):
        pred_s[k] = x.detach().cpu().numpy()[0]
        x = x + float(dt_obs) * net(x)

    pred = scaler.inverse_transform(pred_s).astype(np.float64)
    return pred

def evaluate_model(name, net, scaler, test_data_full, ref_full, dt, KMAX, n_starts=50):
    row = {"model": name}
    test_data_full = np.asarray(test_data_full, dtype=np.float64)
    
    # -------------------------------------------------------------
    # 1. Short/Mid-term Metrics (MSE, NMSE) via 50 Different Starts
    # -------------------------------------------------------------
    max_k = max(MSE_WINDOWS)
    valid_len = len(test_data_full) - max_k - 1
    
    if valid_len < n_starts:
        indices = np.arange(max(0, valid_len))
    else:
        np.random.seed(42) # fixed seed for reproducible comparison
        indices = np.random.choice(valid_len, n_starts, replace=False)

    results_mse = {k: [] for k in MSE_WINDOWS}
    results_nmse = {k: [] for k in MSE_WINDOWS}

    for idx in indices:
        x0 = test_data_full[idx]
        gt = test_data_full[idx : idx + max_k + 1]
        
        # Rollout specifically for the required window length
        pred = rollout_from_x0(net, scaler, x0, max_k, dt, device)

        for k in MSE_WINDOWS:
            err = gt[:k+1] - pred[:k+1]
            mse = np.mean(err**2)
            nmse = mse / (np.mean(gt[:k+1]**2) + 1e-12)
            
            results_mse[k].append(mse)
            results_nmse[k].append(nmse)

    # 格式化为 Mean ± STD 的字符串
    for k in MSE_WINDOWS:
        if len(results_mse[k]) > 0:
            mean_mse = np.mean(results_mse[k])
            std_mse = np.std(results_mse[k])
            row[f"MSE_{k}"] = f"{mean_mse:.4g} ± {std_mse:.4g}"
            
            mean_nmse = np.mean(results_nmse[k])
            std_nmse = np.std(results_nmse[k])
            row[f"NMSE_{k}"] = f"{mean_nmse:.4g} ± {std_nmse:.4g}"
        else:
            row[f"MSE_{k}"] = "nan"
            row[f"NMSE_{k}"] = "nan"

    # -------------------------------------------------------------
    # 2. Long-term Metrics (W2, PSD) via Single Long Rollout
    # -------------------------------------------------------------
    pred_long = rollout_from_x0(net, scaler, test_data_full[0], KMAX, dt, device)
    
    row["Attractor_W2"] = sinkhorn_w2_cloud(ref_full, pred_long, blur=W2_BLUR, device=device)
    row["PSD_logL1"] = psd_logl1_distance(ref_full, pred_long, dt, nfft=PSD_NFFT, channels=PSD_CHANNELS)

    return row, pred_long

def fmt(x):
    if isinstance(x, float):
        return "nan" if not np.isfinite(x) else f"{x:.6g}"
    return str(x)

# ============================================================
# checkpoint discovery
# 自动寻找：
#   FixedTime_seed123456_K3
#   FixedTime_seed123456_K6
#   FixedTime_seed123456_K9
#   FixedTime_seed123456_K12
#   以及其他所有符合命名规则的 K
# ============================================================
import re
from pathlib import Path

ROOT_DIR = PORT_CHECKPOINTS / "notebook"

PRIME_CKPT_PATH = PORT_REPO / "checkpoints/lorenz63/seed_123456/Implicit_BEST.p"

if not PRIME_CKPT_PATH.exists():
    raise FileNotFoundError(
        f"PRIME checkpoint not found:\n{PRIME_CKPT_PATH}"
    )


def extract_fixed_lag_from_folder(folder: Path) -> int:
    match = re.fullmatch(
        r"FixedTime_seed123456_K(\d+)",
        folder.name,
    )

    if match is None:
        raise ValueError(
            f"Cannot parse fixed lag from folder: {folder}"
        )

    return int(match.group(1))


def resolve_fixed_best_checkpoint(
    folder: Path,
    fixed_lag: int,
) -> Path:
    """
    Prefer the exact non-timestamped file:
        FixedTime_K{K}_BEST.p

    Otherwise accept timestamped files such as:
        FixedTime_K{K}_BEST_20260726_221648.p

    Explicitly exclude:
        BEST_L2
        FINAL
    """

    exact_path = (
        folder
        / f"FixedTime_K{fixed_lag}_BEST.p"
    )

    if exact_path.exists():
        return exact_path.resolve()

    candidates = []

    for path in folder.glob(
        f"FixedTime_K{fixed_lag}_BEST*.p"
    ):
        upper_name = path.name.upper()

        if "BEST_L2" in upper_name:
            continue

        if "FINAL" in upper_name:
            continue

        candidates.append(
            path.resolve()
        )

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No BEST_TOTAL checkpoint found for K={fixed_lag} "
            f"inside:\n{folder}"
        )

    # 若同一个 K 重复训练过，明确显示候选文件，
    # 然后使用最新生成的一个。
    candidates.sort(
        key=lambda path: path.stat().st_mtime
    )

    if len(candidates) > 1:
        print(
            f"\n[WARNING] Multiple BEST_TOTAL files found "
            f"for K={fixed_lag}:"
        )

        for candidate in candidates:
            print("  ", candidate)

        print(
            "Using the most recently modified file:",
            candidates[-1],
        )

    return candidates[-1]


fixed_folders = sorted(
    [
        folder
        for folder in ROOT_DIR.glob(
            "FixedTime_seed123456_K*"
        )
        if folder.is_dir()
    ],
    key=extract_fixed_lag_from_folder,
)

if len(fixed_folders) == 0:
    raise FileNotFoundError(
        f"No FixedTime_seed123456_K* directories found under:\n"
        f"{ROOT_DIR}"
    )


# 最终模型清单：
# PRIME + 所有被发现的 Fixed-Time K
MODEL_CKPTS = {
    "PRIME": PRIME_CKPT_PATH.resolve(),
}

for folder in fixed_folders:
    fixed_lag = extract_fixed_lag_from_folder(
        folder
    )

    checkpoint = resolve_fixed_best_checkpoint(
        folder,
        fixed_lag,
    )

    MODEL_CKPTS[
        f"Fixed-Time K={fixed_lag}"
    ] = checkpoint


print(
    "============================================================"
)
print("Checkpoint manifest")
print(
    "============================================================"
)

for model_name, checkpoint_path in MODEL_CKPTS.items():
    print(
        f"{model_name:>20s} : {checkpoint_path}"
    )

print(
    "============================================================"
)


# ============================================================
# 原 evaluation configuration 保持不变
# ============================================================
MSE_WINDOWS = [
    10,
    20,
    50,
    100,
    200,
    300,
    400,
    500,
]

PSD_NFFT = 2048
PSD_CHANNELS = None
W2_BLUR = 0.05

PLOT_ON = PORT_ARGS.plots
PLOT_N = 1500

assert "train_data" in globals()
assert "test_data" in globals()
assert "dt_obs" in globals()

device = PORT_DEVICE

print("\ndevice:", device)

T_eval = int(
    len(test_data)
)

dt = float(
    dt_obs
)

ref = np.asarray(
    test_data[:T_eval],
    dtype=np.float64,
)

m = ref.shape[1]

print(
    f"eval: T_eval={T_eval}, "
    f"dim={m}, "
    f"dt_obs={dt}"
)


# ============================================================
# 从这里到 evaluate_model() 结束：
# 保留你原代码中的以下函数，完全不要修改：
#
# CPU_Unpickler
# load_p_cpu
# smart_load
# psd_logl1_distance
# sinkhorn_w2_cloud
# infer_mlp_dims_from_state_dict
# build_net
# load_model_ckpt
# rollout_from_x0
# evaluate_model
# ============================================================


# ============================================================
# load + evaluate
# PRIME + 所有 Fixed-Time K
# ============================================================
rows = []
models_pred = {}
model_objects = {}

KMAX = T_eval - 1


def load_and_evaluate_one_model(
    model_name: str,
    checkpoint_path: Path,
):
    obj, state_dict, scaler = load_model_ckpt(
        str(checkpoint_path)
    )

    d_in, width, d_out = (
        infer_mlp_dims_from_state_dict(
            state_dict
        )
    )

    if d_in != m or d_out != m:
        raise ValueError(
            f"{model_name} dimension mismatch: "
            f"input={d_in}, output={d_out}, "
            f"test dimension={m}"
        )

    net = build_net(
        d_in,
        width=width,
    ).to(device)

    net.load_state_dict(
        state_dict
    )

    net.eval()

    for parameter in net.parameters():
        parameter.requires_grad_(False)

    config = (
        obj.get("config", {})
        if isinstance(obj, dict)
        else {}
    )

    print(
        "\n------------------------------------------------------------"
    )
    print("Evaluating:", model_name)
    print("Checkpoint:", checkpoint_path)
    print("Internal seed:", config.get("seed"))
    print(
        "Stored fixed lag:",
        config.get(
            "fixed_lag_observation_steps",
            config.get("fixed_lag"),
        ),
    )
    print(
        "BEST iteration:",
        config.get(
            "best_total_iteration_pre_update",
            config.get(
                "best_total_iter",
                config.get("selected_iter"),
            ),
        ),
    )
    print(
        "------------------------------------------------------------"
    )

    row, pred_long = evaluate_model(
        name=model_name,
        net=net,
        scaler=scaler,
        test_data_full=test_data,
        ref_full=ref,
        dt=dt,
        KMAX=KMAX,
        n_starts=50,
    )

    rows.append(
        row
    )

    models_pred[
        model_name
    ] = pred_long

    model_objects[
        model_name
    ] = {
        "checkpoint": str(checkpoint_path),
        "object": obj,
        "state_dict": state_dict,
        "scaler": scaler,
        "net": net,
    }


for model_name, checkpoint_path in MODEL_CKPTS.items():
    load_and_evaluate_one_model(
        model_name,
        checkpoint_path,
    )


# ============================================================
# print tables
# 保留原始评价协议和指标定义
# ============================================================
def fmt(value):
    if isinstance(value, float):
        return (
            "nan"
            if not np.isfinite(value)
            else f"{value:.6g}"
        )

    return str(value)


COL_WIDTH = 20


print(
    "\n"
    "==================== "
    "SHORT-WINDOW MSE "
    "(Mean ± STD over 50 starts, lower better) "
    "===================="
)

mse_cols = (
    ["model"]
    + [
        f"MSE_{k}"
        for k in MSE_WINDOWS
    ]
)

header = " | ".join(
    [
        f"{column:>{COL_WIDTH}s}"
        for column in mse_cols
    ]
)

print(header)
print("-" * len(header))

for row in rows:
    print(
        " | ".join(
            [
                f"{fmt(row[column]):>{COL_WIDTH}s}"
                for column in mse_cols
            ]
        )
    )


print(
    "\n"
    "==================== "
    "SHORT-WINDOW NMSE "
    "(Mean ± STD over 50 starts, lower better) "
    "===================="
)

nmse_cols = (
    ["model"]
    + [
        f"NMSE_{k}"
        for k in MSE_WINDOWS
    ]
)

header = " | ".join(
    [
        f"{column:>{COL_WIDTH}s}"
        for column in nmse_cols
    ]
)

print(header)
print("-" * len(header))

for row in rows:
    print(
        " | ".join(
            [
                f"{fmt(row[column]):>{COL_WIDTH}s}"
                for column in nmse_cols
            ]
        )
    )


print(
    "\n"
    "==================== "
    "STRUCTURE METRICS "
    "(1 Long Rollout, lower better) "
    "===================="
)

structure_cols = [
    "model",
    "Attractor_W2",
    "PSD_logL1",
]

header = " | ".join(
    [
        f"{column:>20s}"
        for column in structure_cols
    ]
)

print(header)
print("-" * len(header))

for row in rows:
    print(
        " | ".join(
            [
                f"{fmt(row[column]):>20s}"
                for column in structure_cols
            ]
        )
    )


# ============================================================
# 将结果整理成 DataFrame，并保存 CSV
# ============================================================
import pandas as pd

results_df = pd.DataFrame(
    rows
)

print("\nComplete result table:")
display(results_df)

RESULT_CSV = PORT_OUTPUT / "PRIME_vs_all_FixedTime_K_results.csv"

results_df.to_csv(
    RESULT_CSV,
    index=False,
)

print(
    "\nSaved CSV:",
    RESULT_CSV,
)


# ============================================================
# 可选：只画长期 W2 和 PSD 随 K 的变化
# PRIME 作为水平参考线
# ============================================================
fixed_rows = []

for row in rows:
    match = re.fullmatch(
        r"Fixed-Time K=(\d+)",
        row["model"],
    )

    if match is not None:
        fixed_rows.append(
            {
                "K": int(match.group(1)),
                "Attractor_W2": float(
                    row["Attractor_W2"]
                ),
                "PSD_logL1": float(
                    row["PSD_logL1"]
                ),
            }
        )

fixed_rows = sorted(
    fixed_rows,
    key=lambda item: item["K"],
)

prime_row = next(
    row
    for row in rows
    if row["model"] == "PRIME"
)

if len(fixed_rows) > 0:
    fixed_k_values = np.asarray(
        [
            item["K"]
            for item in fixed_rows
        ],
        dtype=np.int64,
    )

    fixed_w2_values = np.asarray(
        [
            item["Attractor_W2"]
            for item in fixed_rows
        ],
        dtype=np.float64,
    )

    fixed_psd_values = np.asarray(
        [
            item["PSD_logL1"]
            for item in fixed_rows
        ],
        dtype=np.float64,
    )

    plt.figure(
        figsize=(8, 4),
        dpi=150,
    )

    plt.plot(
        fixed_k_values,
        fixed_w2_values,
        marker="o",
        label="Fixed-Time",
    )

    plt.axhline(
        float(prime_row["Attractor_W2"]),
        linestyle="--",
        label="PRIME",
    )

    plt.xlabel(
        "Fixed lag K "
        f"(physical time = K × {dt_obs})"
    )

    plt.ylabel(
        "Attractor W2"
    )

    plt.title(
        "Attractor W2 versus fixed lag"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


    plt.figure(
        figsize=(8, 4),
        dpi=150,
    )

    plt.plot(
        fixed_k_values,
        fixed_psd_values,
        marker="o",
        label="Fixed-Time",
    )

    plt.axhline(
        float(prime_row["PSD_logL1"]),
        linestyle="--",
        label="PRIME",
    )

    plt.xlabel(
        "Fixed lag K "
        f"(physical time = K × {dt_obs})"
    )

    plt.ylabel(
        "PSD log-L1"
    )

    plt.title(
        "PSD error versus fixed lag"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 原始长期轨迹 overlay
# 所有模型都画在同一张图中
# ============================================================
if PLOT_ON:
    Nplot = min(
        PLOT_N,
        T_eval,
    )

    time_axis = (
        np.arange(Nplot)
        * dt
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        dpi=160,
        sharex=True,
    )

    axes[0].plot(
        time_axis,
        ref[:Nplot, 0],
        linewidth=1.5,
        label="Reference",
    )

    for model_name, prediction in models_pred.items():
        axes[0].plot(
            time_axis,
            prediction[:Nplot, 0],
            linewidth=1.0,
            label=model_name,
        )

    axes[0].set_title(
        "Long rollout: channel 0"
    )

    axes[0].legend(
        ncol=2,
    )

    for model_name, prediction in models_pred.items():
        error = np.linalg.norm(
            prediction[:Nplot]
            - ref[:Nplot],
            axis=1,
        )

        axes[1].plot(
            time_axis,
            error,
            linewidth=1.0,
            label=model_name,
        )

    axes[1].set_yscale(
        "log"
    )

    axes[1].set_xlabel(
        "Time"
    )

    axes[1].set_title(
        "Long-rollout trajectory error norm"
    )

    axes[1].legend(
        ncol=2,
    )

    plt.tight_layout()
    plt.show()


print("\nDone.")
