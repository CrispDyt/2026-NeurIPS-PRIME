#!/usr/bin/env python3
"""Table 21 heterogeneous training benchmark, recovered from pP3L_Rebuttal.ipynb cell index 6.
The timer includes model construction and training, but excludes loading/scaling,
saving, evaluation and plotting. Per-update cost is total timed cost / updates.
"""
import argparse
import hashlib
import json
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MaxAbsScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/baselines'))
import lorenz_heterogeneous_baselines as hb
from lorenz_heterogeneous_baselines import (
    set_seed, load_array, select_obs, train_one_baseline, save_model,
    NVARModel, ESNModel, LSTMRolloutModel, KoopmanRolloutModel, DySLIMRolloutModel,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=4000, help='Optimization steps for neural methods; ridge fits are unchanged.')
    p.add_argument('--seeds', nargs='+', default=['33','123','123456','1000','13532'])
    p.add_argument('--models', nargs='+', default=['ngrc','esn','lstm','koopman','dyslim'])
    p.add_argument('--device', choices=['cuda','cpu','auto'], default='cuda')
    p.add_argument('--data-dir', type=Path, default=ROOT/'data/lorenz63')
    p.add_argument('--output-dir', type=Path, default=ROOT/'runs/cost/heterogeneous')
    p.add_argument('--save-checkpoints', action='store_true', help='Save fitted models after timing.')
    p.add_argument('--print-every', type=int, default=500)
    p.add_argument('--threads', type=int, help='Optional CPU thread override; default preserves the runtime configuration.')
    cli = p.parse_args()
    cli.seeds = [int(x) for token in cli.seeds for x in token.split(',') if x]
    cli.models = [x for token in cli.models for x in token.split(',') if x]
    if cli.steps < 1 or cli.print_every < 1 or not cli.seeds or not cli.models:
        p.error('steps/print-every must be positive and seeds/models nonempty')
    if any(x not in ['ngrc','esn','lstm','koopman','dyslim'] for x in cli.models):
        p.error('models must be ngrc, esn, lstm, koopman or dyslim')
    if len(set(cli.seeds)) != len(cli.seeds) or len(set(cli.models)) != len(cli.models):
        p.error('Repeated seeds or models would duplicate summary records')
    if cli.threads is not None:
        if cli.threads < 1:
            p.error('--threads must be positive')
        torch.set_num_threads(cli.threads)
    import gc
    import platform
    import sys
    from types import SimpleNamespace

    # ----------------------------
    # Paths and benchmark settings
    # ----------------------------
    TRAIN_DATA_PATH = cli.data_dir / "train_data.npy"
    TEST_DATA_PATH = cli.data_dir / "test_data.npy"
    OUT_DIR = cli.output_dir
    if (OUT_DIR / "training_time_metadata.json").exists() or (OUT_DIR / "training_time_per_seed.csv").exists():
        raise FileExistsError(f"Choose a fresh output directory: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if cli.save_checkpoints:
        (OUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)

    SEEDS = cli.seeds
    MODEL_NAMES = cli.models

    DEVICE_MODE = cli.device     # "auto", "cuda", or "cpu"
    SAVE_CHECKPOINTS = cli.save_checkpoints  # saving occurs AFTER timing and is excluded
    PRINT_EVERY = cli.print_every

    # Same hyperparameters as the uploaded heterogeneous-baseline script.
    BASE_ARGS = dict(
        dt_obs=0.1,
        seq_len=10,
        steps=cli.steps,
        batch_size=256,
        lr=1e-3,
        grad_clip=1.0,
        print_every=PRINT_EVERY,
        rollout_clip=50.0,

        # NGRC / NVAR
        nvar_order=2,
        nvar_ridge=1e-3,
        nvar_feature_clip=20.0,

        # ESN
        esn_size=700,
        esn_spectral_radius=0.9,
        esn_leak=0.3,
        esn_input_scale=0.5,
        esn_density=0.05,
        esn_ridge=1e-5,
        esn_washout=100,

        # LSTM
        lstm_hidden=128,
        lstm_layers=2,

        # Koopman-AE
        koopman_hidden=128,
        koopman_latent=16,
        koopman_recon_w=0.25,
        koopman_latent_w=0.1,

        # DySLIM-style
        dyslim_hidden=128,
        dyslim_rollout=16,
        dyslim_mmd_w=0.2,
    )

    CATEGORY = {
        "ngrc": "history-window / nonlinear autoregression",
        "esn": "history-state / reservoir",
        "lstm": "history-window / recurrent model",
        "koopman": "latent-state",
        "dyslim": "measure matching (DySLIM-style)",
    }

    DISPLAY_NAME = {
        "ngrc": "NGRC / NVAR",
        "esn": "Reservoir / ESN",
        "lstm": "LSTM",
        "koopman": "Koopman-AE",
        "dyslim": "DySLIM-style",
    }

    # ----------------------------
    # Device selection
    # ----------------------------
    def select_working_device(mode: str) -> torch.device:
        mode = mode.lower()
        if mode == "cpu":
            return torch.device("cpu")
        if mode not in {"auto", "cuda"}:
            raise ValueError("DEVICE_MODE must be 'auto', 'cuda', or 'cpu'.")
        if not torch.cuda.is_available():
            if mode == "cuda":
                raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
            return torch.device("cpu")
        try:
            probe = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            del probe
            torch.cuda.empty_cache()
            return torch.device("cuda")
        except RuntimeError as exc:
            if mode == "cuda":
                raise
            print("[WARNING] CUDA is visible but unavailable; using CPU.")
            print("CUDA error:", exc)
            return torch.device("cpu")

    device = select_working_device(DEVICE_MODE)

    if device.type == "cuda":
        # One untimed device warm-up to exclude first-use kernel/context overhead.
        a = torch.randn(256, 256, device=device)
        b = torch.randn(256, 256, device=device)
        _ = a @ b
        torch.cuda.synchronize()
        del a, b
        torch.cuda.empty_cache()

    # ----------------------------
    # Load train/test arrays
    # ----------------------------
    if not TRAIN_DATA_PATH.is_file():
        raise FileNotFoundError(f"train_data.npy not found:\n{TRAIN_DATA_PATH.resolve()}")
    if not TEST_DATA_PATH.is_file():
        raise FileNotFoundError(f"test_data.npy not found:\n{TEST_DATA_PATH.resolve()}")

    train_raw0 = load_array(str(TRAIN_DATA_PATH))
    test_raw0 = load_array(str(TEST_DATA_PATH))

    # Match the original script:
    # - use both columns when d=2;
    # - otherwise default to Lorenz [X,Z] = columns [0,2].
    train_raw = select_obs(train_raw0, None)
    test_raw = select_obs(test_raw0, None)

    if train_raw.shape[1] != test_raw.shape[1]:
        raise ValueError(
            f"Train/test observation dimensions differ: "
            f"{train_raw.shape[1]} vs {test_raw.shape[1]}"
        )

    scaler = MaxAbsScaler().fit(train_raw)
    train_scaled = scaler.transform(train_raw).astype(np.float64)

    print("=" * 96)
    print("Heterogeneous baseline training-time benchmark")
    print("=" * 96)
    print("train path:             ", TRAIN_DATA_PATH.resolve())
    print("test path:              ", TEST_DATA_PATH.resolve())
    print("train shape used:       ", train_raw.shape)
    print("test shape used:        ", test_raw.shape)
    print("device:                 ", device)
    print(
        "device name:            ",
        torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else (platform.processor() or "CPU"),
    )
    print("seeds:                  ", SEEDS)
    print("models:                 ", MODEL_NAMES)
    print("neural training steps:  ", BASE_ARGS["steps"])
    print("timing scope:            model construction + training only")
    print("excluded:                data loading, scaling, save, eval, plots")
    print("=" * 96)

    # ----------------------------
    # Helpers for timing and size
    # ----------------------------
    def sync_device():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def model_parameter_count(model) -> int:
        if isinstance(model, NVARModel):
            return int(np.asarray(model.W).size)
        if isinstance(model, ESNModel):
            return int(
                np.asarray(model.Win).size
                + np.asarray(model.Wres).size
                + np.asarray(model.bias).size
                + np.asarray(model.Wout).size
            )
        if isinstance(model, LSTMRolloutModel):
            return int(sum(p.numel() for p in model.net.parameters()))
        if isinstance(model, KoopmanRolloutModel):
            return int(sum(p.numel() for p in model.net.parameters()))
        if isinstance(model, DySLIMRolloutModel):
            return int(sum(p.numel() for p in model.net.parameters()))
        return 0

    def backend_for(model_name: str) -> str:
        if model_name in {"ngrc", "esn"}:
            return "NumPy/CPU closed-form"
        return f"PyTorch/{device.type.upper()} iterative"

    # ----------------------------
    # Train and time all methods
    # ----------------------------
    rows = []

    for seed in SEEDS:
        for model_name in MODEL_NAMES:
            print("\n" + "#" * 96)
            print(
                f"Seed {seed} | {DISPLAY_NAME[model_name]} | "
                f"{CATEGORY[model_name]}"
            )
            print("#" * 96)

            set_seed(seed)
            args = SimpleNamespace(**BASE_ARGS, seed=int(seed))

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            sync_device()
            start = time.perf_counter()

            status = "ok"
            error_message = ""
            model = None

            try:
                model = train_one_baseline(
                    model_name,
                    train_scaled=train_scaled,
                    args=args,
                    device=device,
                )
                sync_device()
                elapsed_seconds = float(time.perf_counter() - start)

            except Exception as exc:
                sync_device()
                elapsed_seconds = float(time.perf_counter() - start)
                status = "failed"
                error_message = repr(exc)
                print("[FAILED]", error_message)

            parameter_count = model_parameter_count(model) if model is not None else np.nan

            checkpoint_path = None
            if model is not None and SAVE_CHECKPOINTS:
                safe_name = DISPLAY_NAME[model_name].replace("/", "-").replace(" ", "_")
                checkpoint_path = (
                    OUT_DIR
                    / "checkpoints"
                    / f"{safe_name}_seed{seed}.p"
                )
                # Save after timing so serialization is not counted.
                save_model(model, checkpoint_path)

            iterative_steps = (
                int(BASE_ARGS["steps"])
                if model_name in {"lstm", "koopman", "dyslim"}
                else 0
            )

            rows.append(
                {
                    "seed": int(seed),
                    "method": DISPLAY_NAME[model_name],
                    "category": CATEGORY[model_name],
                    "backend": backend_for(model_name),
                    "status": status,
                    "train_seconds": elapsed_seconds,
                    "train_minutes": elapsed_seconds / 60.0,
                    "optimization_steps": iterative_steps,
                    "milliseconds_per_update": (
                        1000.0 * elapsed_seconds / iterative_steps
                        if iterative_steps > 0
                        else np.nan
                    ),
                    "parameter_or_coefficient_count": parameter_count,
                    "checkpoint": str(checkpoint_path) if checkpoint_path else "",
                    "error": error_message,
                }
            )

            print(f"Pure training time: {elapsed_seconds:.4f} s")
            if iterative_steps > 0:
                print(
                    "Time per optimization update: "
                    f"{1000.0 * elapsed_seconds / iterative_steps:.4f} ms"
                )

            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    results_df = pd.DataFrame(rows)

    # ----------------------------
    # Per-run table
    # ----------------------------
    print("\n" + "=" * 96)
    print("PER-SEED PURE TRAINING TIME")
    print("=" * 96)


    # ----------------------------
    # Summary across seeds
    # ----------------------------
    valid_df = results_df[results_df["status"] == "ok"].copy()

    summary_df = (
        valid_df.groupby(
            ["method", "category", "backend"],
            sort=False,
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "count"),
            train_seconds_mean=("train_seconds", "mean"),
            train_seconds_std=("train_seconds", "std"),
            train_seconds_median=("train_seconds", "median"),
            train_seconds_min=("train_seconds", "min"),
            train_seconds_max=("train_seconds", "max"),
            milliseconds_per_update_mean=("milliseconds_per_update", "mean"),
            parameter_or_coefficient_count=(
                "parameter_or_coefficient_count",
                "median",
            ),
        )
    )

    # Ratios are useful once PRIME time is placed beside this table.
    # Here we normalize to DySLIM-style, the measure-matching baseline.
    dyslim_row = summary_df[summary_df["method"] == "DySLIM-style"]
    if len(dyslim_row) == 1:
        dyslim_time = float(dyslim_row["train_seconds_mean"].iloc[0])
        summary_df["relative_to_DySLIM_style"] = (
            summary_df["train_seconds_mean"] / dyslim_time
        )
    else:
        summary_df["relative_to_DySLIM_style"] = np.nan

    print("\n" + "=" * 96)
    print("TRAINING-TIME SUMMARY ACROSS SEEDS")
    print("=" * 96)



    # ----------------------------
    # Save results and metadata
    # ----------------------------
    PER_RUN_CSV = OUT_DIR / "training_time_per_seed.csv"
    SUMMARY_CSV = OUT_DIR / "training_time_summary.csv"
    METADATA_JSON = OUT_DIR / "training_time_metadata.json"

    results_df.to_csv(PER_RUN_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    metadata = {
        "train_path": str(TRAIN_DATA_PATH.resolve()),
        "test_path": str(TEST_DATA_PATH.resolve()),
        "train_shape_used": list(train_raw.shape),
        "test_shape_used": list(test_raw.shape),
        "selected_observation_rule": (
            "all columns if d=2, otherwise Lorenz columns [0,2]"
        ),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else (platform.processor() or "CPU")
        ),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "seeds": SEEDS,
        "models": MODEL_NAMES,
        "base_args": BASE_ARGS,
        "timing_scope": (
            "Model construction and pure training only. "
            "Data loading, scaling, checkpoint saving, evaluation, "
            "and plotting are excluded."
        ),
        "important_caveat": (
            "DySLIM-style is the practical MMD implementation in the uploaded "
            "script, not the verbatim official JAX implementation. "
            "NGRC and ESN are closed-form NumPy/CPU methods, whereas the neural "
            "baselines use iterative PyTorch training."
        ),
    }

    metadata.update(
        train_sha256=hashlib.sha256(TRAIN_DATA_PATH.read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256(TEST_DATA_PATH.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        save_checkpoints=SAVE_CHECKPOINTS,
        update_time_definition="1000 * total model-construction-plus-training seconds / number of optimization updates",
        source="NIPS/Lorenz/1-step MSE/pP3L_Rebuttal.ipynb, code cell index 6 (commented source)",
    )
    print(results_df.to_string(index=False))
    print(summary_df.to_string(index=False))
    with METADATA_JSON.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nSaved:")
    print(PER_RUN_CSV)
    print(SUMMARY_CSV)
    print(METADATA_JSON)
    print("\nDone.")
    if (results_df["status"] != "ok").any():
        raise SystemExit("One or more timing runs failed; inspect training_time_per_seed.csv.")


if __name__ == "__main__":
    main()
