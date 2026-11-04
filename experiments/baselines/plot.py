"""Figure 9: portable adaptation of plot_lorenz.ipynb cell 0."""
import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir', type=Path, default=ROOT/'checkpoints/baselines')
    p.add_argument('--prime-checkpoint', type=Path)
    p.add_argument('--test-path', type=Path, default=ROOT/'data/lorenz63/test_data.npy')
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/baselines/figure9')
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--horizon', type=int, default=700)
    args = p.parse_args()
    import os
    import io
    import re
    import csv
    import json
    import pickle

    import numpy as np
    import sys
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    import torch
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import lorenz_heterogeneous_baselines as hb


    # ============================================================
    # 1. Config
    # ============================================================
    OUT_DIR = args.output_dir
    TEST_PATH = str(args.test_path)

    # Heterogeneous baselines
    CKPT_DIR = args.checkpoint_dir
    MODEL_CKPTS = [
        ("NGRC-NVAR",     CKPT_DIR / "NGRC-NVAR_seed1000.p"),
        ("Reservoir-ESN", CKPT_DIR / "Reservoir-ESN_seed1000.p"),
        ("LSTM",          CKPT_DIR / "LSTM_seed1000.p"),
        ("Koopman-AE",    CKPT_DIR / "Koopman-AE_seed1000.p"),
        ("DySLIM-style",  CKPT_DIR / "DySLIM-style_seed1000.p"),
    ]

    # PRIME checkpoint (user specified)
    PRIME_CKPT = args.prime_checkpoint or CKPT_DIR / "PRIME_figure9_seed1000.p"

    START = args.start
    HORIZON = args.horizon

    SAVE_FIG = True
    OUT_PNG  = OUT_DIR / "lorenz_heterogeneous_plus_prime_2x3_rollouts.png"
    OUT_PDF  = OUT_DIR / "lorenz_heterogeneous_plus_prime_2x3_rollouts.pdf"

    # For unstable models
    CLIP_PRED_TO_GT_RANGE = False
    CLIP_MARGIN_MULT = 0.25

    DEVICE = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.set_num_threads(1)
    print("device:", DEVICE)


    # ============================================================
    # 2. Generic load helpers
    # ============================================================
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
            return super().find_class(module, name)


    def load_p_cpu(path):
        with open(path, "rb") as f:
            return CPU_Unpickler(f).load()


    def smart_load(path, device="cpu"):
        path = Path(path)
        last_err = None

        try:
            return torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            last_err = e

        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

        try:
            return load_p_cpu(path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint: {path}\n"
                f"torch.load error: {last_err}\n"
                f"pickle/cpu-unpickler error: {e}"
            )


    def strip_module_prefix(sd):
        if not isinstance(sd, dict):
            return sd
        if len(sd) > 0 and all(k.startswith("module.") for k in sd.keys()):
            return {k.replace("module.", "", 1): v for k, v in sd.items()}
        return sd


    def maybe_strip_prefix(sd, prefix):
        """
        If all keys start with prefix + '.', strip that prefix.
        """
        if not isinstance(sd, dict) or len(sd) == 0:
            return sd
        prefix_dot = prefix + "."
        if all(k.startswith(prefix_dot) for k in sd.keys()):
            return {k[len(prefix_dot):]: v for k, v in sd.items()}
        return sd


    def infer_lstm_dims(sd):
        hidden = sd["lstm.weight_hh_l0"].shape[1]
        layers = len([k for k in sd.keys() if k.startswith("lstm.weight_ih_l")])
        return int(hidden), int(layers)


    def natural_key(s):
        parts = re.split(r"(\d+)", s)
        out = []
        for p in parts:
            if p.isdigit():
                out.append(int(p))
            else:
                out.append(p)
        return out


    def extract_linear_weight_keys(sd):
        """
        Extract ordered linear weight keys from a Sequential-like state_dict.
        Works for keys like:
          0.weight, 2.weight, 4.weight, 6.weight
        """
        wkeys = [k for k in sd.keys() if k.endswith(".weight")]
        wkeys = sorted(wkeys, key=natural_key)
        return wkeys


    # ============================================================
    # 3. Heterogeneous baselines loader
    # ============================================================
    def load_saved_model(path, device):
        obj = smart_load(path, device="cpu")
        typ = obj.get("type", None)

        if typ == "nvar":
            return hb.NVARModel(
                W=obj["W"],
                seq_len=obj["seq_len"],
                dim=obj["dim"],
                order=obj["order"]
            )

        if typ == "esn":
            return hb.ESNModel(
                Win=obj["Win"], Wres=obj["Wres"], bias=obj["bias"], Wout=obj["Wout"],
                leak=obj["leak"], seq_len=obj["seq_len"], dim=obj["dim"]
            )

        if typ == "lstm":
            sd = obj["state_dict"]
            dim = int(obj["dim"])
            hidden, layers = infer_lstm_dims(sd)
            net = hb.LSTMForecaster(dim=dim, hidden=hidden, layers=layers).to(device)
            net.load_state_dict(sd)
            net.eval()
            return hb.LSTMRolloutModel(net=net, seq_len=obj["seq_len"], dim=dim, device=device)

        if typ == "koopman":
            sd = obj["state_dict"]
            seq_len = int(obj["seq_len"])
            dim = int(obj["dim"])
            in_dim = int(obj.get("in_dim", seq_len * dim))
            hidden = int(sd["encoder.0.weight"].shape[0])
            latent = int(sd["encoder.4.weight"].shape[0])
            net = hb.KoopmanAE(in_dim=in_dim, obs_dim=dim, latent=latent, hidden=hidden).to(device)
            net.load_state_dict(sd)
            net.eval()
            return hb.KoopmanRolloutModel(net=net, seq_len=seq_len, dim=dim, device=device)

        if typ == "dyslim":
            sd = obj["state_dict"]
            dim = int(obj["dim"])
            hidden = int(sd["net.0.weight"].shape[0])
            net = hb.MLPMap(dim=dim, hidden=hidden).to(device)
            net.load_state_dict(sd)
            net.eval()
            return hb.DySLIMRolloutModel(net=net, seq_len=obj["seq_len"], dim=dim, device=device)

        raise ValueError(f"Unknown checkpoint type in {path}: {typ}")


    # ============================================================
    # 4. PRIME loader
    # ============================================================
    def extract_state_dict_and_scaler_from_prime(obj):
        """
        Try to robustly extract state_dict and scaler from PRIME checkpoint.
        """
        sd = None
        scaler = None

        if isinstance(obj, dict):
            # possible scaler keys
            for k in ["scaler", "transformer"]:
                if k in obj:
                    scaler = obj[k]
                    break

            # possible state dict keys
            for k in [
                "net_state_dict",
                "state_dict",
                "model_state_dict",
                "f_state_dict",
                "vf_state_dict",
                "map_state_dict",
            ]:
                if k in obj:
                    sd = obj[k]
                    break

            # nested dict cases
            if sd is None and "net" in obj and isinstance(obj["net"], dict):
                sd = obj["net"]
            if sd is None and "model" in obj and isinstance(obj["model"], dict):
                if "state_dict" in obj["model"]:
                    sd = obj["model"]["state_dict"]

        # if checkpoint itself is a state_dict
        if sd is None and isinstance(obj, dict):
            has_tensor_values = any(torch.is_tensor(v) for v in obj.values())
            has_weight_keys = any(k.endswith(".weight") for k in obj.keys())
            if has_tensor_values and has_weight_keys:
                sd = obj

        if sd is None:
            raise RuntimeError("Could not extract PRIME state_dict from checkpoint.")

        sd = strip_module_prefix(sd)

        # strip common wrappers if present
        for pref in ["net", "model", "f", "vf", "map"]:
            sd_try = maybe_strip_prefix(sd, pref)
            if sd_try is not sd:
                sd = sd_try
                break

        return sd, scaler


    class GenericSequentialMap(nn.Sequential):
        """
        Build a plain MLP from sequential-like state_dict:
          0.weight, 0.bias, 2.weight, 2.bias, ...
        """
        def __init__(self, dims):
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.Tanh())
            super().__init__(*layers)


    def build_prime_model_from_state_dict(sd, device):
        """
        Build a generic Sequential MLP from a Sequential-like state_dict.

        Expected PRIME keys:
            0.weight, 0.bias, 2.weight, 2.bias, ...
        """
        sd = strip_module_prefix(sd)

        # If the checkpoint accidentally has a wrapper prefix, remove it.
        for pref in ["net", "model", "f", "vf", "map"]:
            pref_dot = pref + "."
            if len(sd) > 0 and all(k.startswith(pref_dot) for k in sd.keys()):
                sd = {k[len(pref_dot):]: v for k, v in sd.items()}
                break

        wkeys = extract_linear_weight_keys(sd)
        if len(wkeys) == 0:
            raise RuntimeError("Could not infer linear layers from PRIME state_dict.")

        # Example:
        # 0.weight: [100, 2]
        # 2.weight: [100, 100]
        # 4.weight: [100, 100]
        # 6.weight: [2, 100]
        layer_shapes = [tuple(sd[k].shape) for k in wkeys]
        dims = [layer_shapes[0][1]] + [sh[0] for sh in layer_shapes]

        input_dim = int(dims[0])
        output_dim = int(dims[-1])

        net = GenericSequentialMap(dims).to(device)

        net.load_state_dict(sd, strict=True)

        net.eval()

        return net, input_dim, output_dim, dims
    
    # ============================================================
    # PRIME-specific rollout wrapper
    # ============================================================
    class PrimeEulerRolloutModel:
        """
        PRIME implicit checkpoint rollout.

        Treat the loaded MLP as a vector field:
            x_{k+1} = x_k + dt_base * f_theta(x_k)

        This is different from a direct next-state map:
            x_{k+1} = f_theta(x_k)
        """
        def __init__(self, net, seq_len, dim, input_dim, device, dt_base=1.0):
            self.net = net
            self.seq_len = int(seq_len)
            self.dim = int(dim)
            self.input_dim = int(input_dim)
            self.device = device
            self.dt_base = float(dt_base)
            self.name = "PRIME"

        @torch.no_grad()
        def rollout_scaled(self, hist, horizon):
            hist = np.asarray(hist, dtype=np.float32)

            if hist.ndim != 2:
                raise ValueError(f"hist must be 2D, got shape={hist.shape}")
            if hist.shape[1] != self.dim:
                raise ValueError(f"hist dim mismatch: hist.shape[1]={hist.shape[1]}, model dim={self.dim}")
            if hist.shape[0] < self.seq_len:
                raise ValueError(f"hist too short: got {hist.shape[0]}, need seq_len={self.seq_len}")

            cur_hist = hist[-self.seq_len:].copy()
            preds = []

            for _ in range(int(horizon)):
                if self.input_dim == self.dim:
                    xin = cur_hist[-1].reshape(1, -1)
                elif self.input_dim == self.seq_len * self.dim:
                    xin = cur_hist.reshape(1, -1)
                else:
                    raise ValueError(
                        f"Unsupported PRIME input_dim={self.input_dim}, "
                        f"dim={self.dim}, seq_len={self.seq_len}"
                    )

                x = torch.tensor(xin, dtype=torch.float32, device=self.device)
                f = self.net(x)

                if f.shape[-1] != self.dim:
                    raise ValueError(f"PRIME output dim mismatch: expected {self.dim}, got {f.shape[-1]}")

                # Euler update in scaled space
                y = x[:, -self.dim:] + self.dt_base * f if self.input_dim != self.dim else x + self.dt_base * f

                y_np = y.detach().cpu().numpy()[0].astype(np.float32)
                preds.append(y_np.copy())

                if self.seq_len > 1:
                    cur_hist = np.vstack([cur_hist[1:], y_np[None, :]])
                else:
                    cur_hist = y_np.reshape(1, -1)

            return np.asarray(preds, dtype=np.float32)
    def load_prime_model(prime_ckpt_path, device, fallback_scaler):
        obj = smart_load(prime_ckpt_path, device="cpu")
        sd, scaler = extract_state_dict_and_scaler_from_prime(obj)

        if scaler is None:
            scaler = fallback_scaler

        net, input_dim, output_dim, dims = build_prime_model_from_state_dict(sd, device=device)

        dim = output_dim

        if input_dim == dim:
            seq_len = 1
        elif input_dim % dim == 0:
            seq_len = input_dim // dim
        else:
            raise RuntimeError(
                f"Cannot infer PRIME seq_len from input_dim={input_dim}, output_dim={output_dim}"
            )

        dt_base = 1.0
        if isinstance(obj, dict):
            for k in ["dt_base", "dt", "dt_obs"]:
                if k in obj:
                    try:
                        dt_base = float(obj[k])
                        break
                    except Exception:
                        pass

            if "config" in obj and isinstance(obj["config"], dict):
                for k in ["dt_base", "dt", "dt_obs"]:
                    if k in obj["config"]:
                        try:
                            dt_base = float(obj["config"][k])
                            break
                        except Exception:
                            pass

        wrapper = PrimeEulerRolloutModel(
            net=net,
            seq_len=seq_len,
            dim=dim,
            input_dim=input_dim,
            device=device,
            dt_base=dt_base,
        )

        print("[OK] PRIME loaded")
        print("     input_dim :", input_dim)
        print("     output_dim:", output_dim)
        print("     seq_len   :", seq_len)
        print("     dims      :", dims)
        print("     dt_base   :", dt_base)
        print("     mode      : euler  x_next = x + dt_base * net(x)")

        return wrapper, scaler


    # ============================================================
    # 5. Data helpers
    # ============================================================
    def load_test_observation(test_path, out_dir, scaler):
        X = np.load(test_path, allow_pickle=True)
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"test_data must be 2D, got {X.shape}")

        cfg_path = Path(out_dir) / "run_config.json"
        obs_cols = None

        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            obs_spec = str(cfg.get("obs_cols", "")).strip()
            if obs_spec:
                obs_cols = [int(s) for s in obs_spec.split(",")]

        n_features = getattr(scaler, "n_features_in_", None)

        if obs_cols is None:
            if X.shape[1] == n_features:
                obs_cols = list(range(X.shape[1]))
            elif X.shape[1] >= 3 and n_features == 2:
                # Lorenz partial observation often (u0, u2)
                obs_cols = [0, 2]
            else:
                obs_cols = list(range(min(X.shape[1], int(n_features))))

        return X[:, obs_cols], obs_cols


    def rollout_raw(model, scaler, test_raw, start, max_seq, horizon, scaled_clip=1e3):
        test_scaled = scaler.transform(test_raw)

        hist = test_scaled[start + max_seq - model.seq_len : start + max_seq]
        pred_s = model.rollout_scaled(hist, horizon)
        pred_s = np.asarray(pred_s, dtype=np.float64)

        pred_s = np.nan_to_num(
            pred_s,
            nan=0.0,
            posinf=scaled_clip,
            neginf=-scaled_clip,
        )
        pred_s = np.clip(pred_s, -scaled_clip, scaled_clip)

        pred_raw = scaler.inverse_transform(pred_s)
        return np.asarray(pred_raw, dtype=np.float64)


    def clip_to_ref_range(Y, ref, margin_mult=0.25):
        Y = np.asarray(Y, dtype=np.float64).copy()

        lo = np.nanmin(ref, axis=0)
        hi = np.nanmax(ref, axis=0)
        span = np.maximum(hi - lo, 1e-8)

        lo2 = lo - margin_mult * span
        hi2 = hi + margin_mult * span

        return np.clip(Y, lo2[None, :], hi2[None, :])


    def safe_xy(arr, x_idx=0, y_idx=1):
        arr = np.asarray(arr, dtype=np.float64)
        xy = arr[:, [x_idx, y_idx]]
        xy = xy[np.isfinite(xy).all(axis=1)]
        return xy


    # ============================================================
    # 6. Plot style (copied from your second script)
    # ============================================================
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.labelsize": 13,
        "axes.titlesize": 11,
        "font.size": 11,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    GT_BG_COLOR  = "#c8c8c8"
    GT_SEG_COLOR = "#6e6e6e"

    MODEL_COLORS = {
        "NGRC-NVAR":     "#0072B2",
        "Reservoir-ESN": "#D55E00",
        "LSTM":          "#009E73",
        "Koopman-AE":    "#CC79A7",
        "DySLIM-style":  "#56B4E9",
        "PRIME":         "#E69F00",
    }


    # ============================================================
    # 7. Load shared scaler / test data
    # ============================================================
    scaler_path = CKPT_DIR / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    with open(scaler_path, "rb") as f:
        shared_scaler = pickle.load(f)

    test_raw, obs_cols = load_test_observation(TEST_PATH, OUT_DIR, shared_scaler)

    print("[OK] test_raw shape:", test_raw.shape)
    print("[OK] obs_cols:", obs_cols)


    # ============================================================
    # 8. Load heterogeneous baselines
    # ============================================================
    loaded_models = []

    for model_name, ckpt_path in MODEL_CKPTS:
        if not ckpt_path.exists():
            print(f"[skip] missing checkpoint: {ckpt_path}")
            continue

        model = load_saved_model(ckpt_path, device=DEVICE)
        loaded_models.append({
            "name": model_name,
            "model": model,
            "scaler": shared_scaler,
        })
        print(f"[OK] loaded {model_name:14s} | seq_len={model.seq_len} | {ckpt_path}")


    # ============================================================
    # 9. Load PRIME
    # ============================================================
    if not PRIME_CKPT.exists():
        raise FileNotFoundError(
            f"Cannot find PRIME checkpoint: {PRIME_CKPT}\n"
            "Please check the folder name carefully."
        )

    prime_model, prime_scaler = load_prime_model(
        PRIME_CKPT,
        device=DEVICE,
        fallback_scaler=shared_scaler,
    )

    loaded_models.append({
        "name": "PRIME",
        "model": prime_model,
        "scaler": prime_scaler,
    })

    print(f"[OK] loaded PRIME | seq_len={prime_model.seq_len} | {PRIME_CKPT}")


    # ============================================================
    # 10. Determine common rollout origin and roll out
    # ============================================================

    max_seq = max(item["model"].seq_len for item in loaded_models)

    # All models use histories ending at this same raw data index.
    origin_idx = int(START) + max_seq - 1
    x0_common = test_raw[origin_idx].copy()

    # Number of future prediction steps after x0_common.
    horizon_eff = min(
        int(HORIZON),
        len(test_raw) - origin_idx - 1,
    )

    if horizon_eff <= 10:
        raise ValueError(
            f"Not enough test data for START={START}, max_seq={max_seq}, HORIZON={HORIZON}"
        )

    # GT local trajectory includes the shared initial point.
    # ref[0] is exactly x0_common.
    ref = test_raw[origin_idx : origin_idx + horizon_eff + 1]

    print("\n==================== COMMON ROLLOUT ORIGIN ====================")
    print("origin_idx:", origin_idx)
    print("x0_common:", x0_common)
    print("GT ref shape:", ref.shape)
    print("Future prediction steps:", horizon_eff)
    print("===============================================================")

    preds = []

    for item in loaded_models:
        model_name = item["name"]
        model      = item["model"]
        scaler     = item["scaler"]

        hist_start = int(START) + max_seq - model.seq_len
        hist_end   = int(START) + max_seq  # exclusive

        # This history always ends at origin_idx.
        print(
            f"[history] {model_name:14s} | seq_len={model.seq_len:3d} | "
            f"hist indices = [{hist_start}, {hist_end}) | "
            f"last hist idx = {hist_end - 1}"
        )

        pred_future = rollout_raw(
            model=model,
            scaler=scaler,
            test_raw=test_raw,
            start=START,
            max_seq=max_seq,
            horizon=horizon_eff,
        )

        pred_future = pred_future[:horizon_eff]

        # Force every plotted rollout to start from the identical observed x0.
        pred_full = np.vstack([
            x0_common.reshape(1, -1),
            pred_future,
        ])

        if pred_full.shape[0] != ref.shape[0]:
            raise RuntimeError(
                f"Length mismatch for {model_name}: pred_full={pred_full.shape}, ref={ref.shape}"
            )

        if CLIP_PRED_TO_GT_RANGE:
            pred_plot = clip_to_ref_range(pred_full, ref, margin_mult=CLIP_MARGIN_MULT)
            # Keep the initial point exact even after clipping.
            pred_plot[0] = x0_common
        else:
            pred_plot = pred_full

        preds.append({
            "name": model_name,
            "raw": pred_full,
            "plot": pred_plot,
            "seq_len": model.seq_len,
        })

        finite = np.isfinite(pred_full).all()
        same_start = np.allclose(pred_full[0], x0_common, atol=1e-10, rtol=1e-10)

        print(
            f"[rollout] {model_name:14s} | finite={finite} | "
            f"same_start={same_start} | pred shape={pred_full.shape}"
        )

    print("Effective future horizon:", horizon_eff)


    # ============================================================
    # 11. Axis labels
    # ============================================================
    X_IDX = 0
    Y_IDX = 1

    if len(obs_cols) >= 2:
        X_LABEL = rf"$u_0$"
        Y_LABEL = rf"$u_2$"
    else:
        X_LABEL = r"$y_0$"
        Y_LABEL = r"$y_1$"


    # ============================================================
    # 12. Draw one model per subplot
    # ============================================================
    def draw_model_subplot(ax, model_name, pred_plot):
        # Full GT background attractor
        ax.plot(
            test_raw[:, X_IDX],
            test_raw[:, Y_IDX],
            color=GT_BG_COLOR,
            alpha=0.26,
            linewidth=0.8,
            zorder=1,
        )

        # Local GT segment
        ax.plot(
            ref[:, X_IDX],
            ref[:, Y_IDX],
            color=GT_SEG_COLOR,
            alpha=0.95,
            linewidth=1.55,
            zorder=2,
        )

        color = MODEL_COLORS.get(model_name, "#0072B2")

        # Prediction rollout
        ax.plot(
            pred_plot[:, X_IDX],
            pred_plot[:, Y_IDX],
            color=color,
            alpha=0.95,
            linewidth=1.45,
            zorder=3,
        )

        # Initial point
        ax.scatter(
            pred_plot[0, X_IDX],
            pred_plot[0, Y_IDX],
            color=color,
            marker="o",
            s=34,
            zorder=5,
        )

        handles = [
            Line2D([0], [0], color=GT_BG_COLOR, lw=1.2, alpha=0.8, label="GT background"),
            Line2D([0], [0], color=GT_SEG_COLOR, lw=1.6, alpha=0.95, label="GT local"),
            Line2D([0], [0], color=color, lw=1.45, marker="o", markersize=4.8, label=model_name),
        ]

        xy_parts = []

        bg_xy = safe_xy(test_raw, X_IDX, Y_IDX)
        if len(bg_xy) > 0:
            bg_lo = np.percentile(bg_xy, 2, axis=0)
            bg_hi = np.percentile(bg_xy, 98, axis=0)
            xy_parts.append(np.vstack([bg_lo, bg_hi]))

        ref_xy = safe_xy(ref, X_IDX, Y_IDX)
        if len(ref_xy) > 0:
            xy_parts.append(ref_xy)

        pred_xy = safe_xy(pred_plot, X_IDX, Y_IDX)
        if len(pred_xy) > 0:
            xy_parts.append(pred_xy)

        if len(xy_parts) > 0:
            xy = np.concatenate(xy_parts, axis=0)
            xmin, ymin = np.min(xy, axis=0)
            xmax, ymax = np.max(xy, axis=0)
        else:
            xmin, xmax, ymin, ymax = -1, 1, -1, 1

        dx = max(xmax - xmin, 1e-8)
        dy = max(ymax - ymin, 1e-8)

        ax.set_xlim(xmin - 0.08 * dx, xmax + 0.08 * dx)
        ax.set_ylim(ymin - 0.08 * dy, ymax + 0.08 * dy)

        ax.set_title(model_name, fontsize=11)
        ax.tick_params(axis="both", which="major", labelsize=8.2)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

        ax.legend(
            handles=handles,
            loc="upper right",
            framealpha=0.92,
            edgecolor="gray",
            fontsize=7.0,
            labelspacing=0.22,
            handletextpad=0.4,
            borderpad=0.3,
        )


    # ============================================================
    # 13. Figure: 2 rows x 3 columns
    # ============================================================
    n_rows, n_cols = 2, 3

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15.0, 9.2),
        dpi=300,
        sharex=False,
        sharey=False,
    )

    axes = axes.ravel()

    # Keep the order: 5 heterogeneous baselines + PRIME
    desired_order = ["NGRC-NVAR", "Reservoir-ESN", "LSTM", "Koopman-AE", "DySLIM-style", "PRIME"]
    name_to_pack = {p["name"]: p for p in preds}
    ordered_preds = [name_to_pack[n] for n in desired_order if n in name_to_pack]

    for i, ax in enumerate(axes):
        if i < len(ordered_preds):
            pack = ordered_preds[i]
            draw_model_subplot(ax, pack["name"], pack["plot"])
        else:
            ax.axis("off")

    for r in range(n_rows):
        axes[r * n_cols].set_ylabel(Y_LABEL, fontsize=12)

    for c in range(n_cols):
        axes[(n_rows - 1) * n_cols + c].set_xlabel(X_LABEL, fontsize=12)

    plt.tight_layout(rect=[0.015, 0.02, 0.985, 0.985])

    if SAVE_FIG:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PNG, bbox_inches="tight")
        fig.savefig(OUT_PDF, bbox_inches="tight")
        print("Saved:")
        print(" ", OUT_PNG)
        print(" ", OUT_PDF)

    plt.close(fig)

if __name__ == "__main__":
    main()
