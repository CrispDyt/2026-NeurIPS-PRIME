#!/usr/bin/env python3
"""Rebuild Tables 2, 3 and seed-level appendix statistics from raw evaluation CSVs."""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["PRIME", "MSE", "DIM", "DIM+MSE"]
METRICS = {"TCM": "TCM", "NMSE_20": "NMSE_20", "NMSE_100": "NMSE_100",
           "NMSE_400": "NMSE_400", "SD": "Attractor_W2", "Log-PSD": "PSD_logL1"}
CONFIG = {
    "Lorenz63": ("lorenz63", "per_rollout_prediction_metrics_5seeds_32starts.csv",
                 "per_seed_full_horizon_poincare_metrics.csv",
                 ["tauW1_x10", "tauW1_z40", "tauW1_unimodal"]),
    "KS": ("ks", "per_rollout_prediction_metrics_KS.csv", "per_run_tauW1_original_logic.csv",
           ["Sigma1_u0_0", "Sigma3_u5_0", "Sigma2_u3_0"]),
    "CFD": ("cylinder", "per_rollout_prediction_metrics_CFD.csv", "per_seed_poincare_DRTD_CFD.csv",
            ["Sigma1_u0_4", "Sigma2_u3_4", "Sigma3_u2_4"]),
}
TABLES = {2: ["TCM", "DRTD_Sigma1", "DRTD_Sigma2", "DRTD_Sigma3"],
          3: ["NMSE_20", "NMSE_100", "NMSE_400", "SD", "Log-PSD"]}


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def statistics(values):
    valid = np.array([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    out = {"n_total": len(values), "n_valid": len(valid), "n_missing": len(values) - len(valid)}
    if not len(valid):
        return dict(out, mean=None, std=None, median=None, Q25=None, Q75=None)
    q = np.quantile(valid, [.25, .5, .75], method="linear")
    return dict(out, mean=float(valid.mean()), std=float(valid.std(ddof=0)),
                median=float(q[1]), Q25=float(q[0]), Q75=float(q[2]))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def seed_id(row):
    return int(re.search(r"(\d+)$", row["seed_folder"]).group(1))


def collect(input_root):
    observations = defaultdict(list)
    sources, seeds = [], {}
    for system, (directory, prediction, drtd, sections) in CONFIG.items():
        pred_path, drtd_path = input_root / directory / prediction, input_root / directory / drtd
        pred, tau = read_csv(pred_path), read_csv(drtd_path)
        for path, rows in [(pred_path, pred), (drtd_path, tau)]:
            sources.append(dict(path=str(path.relative_to(input_root)), rows=len(rows),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        seeds[system] = sorted({seed_id(row) for row in pred})
        if len(pred) != 640 or len(seeds[system]) != 5:
            raise ValueError(f"{system}: expected four models x five seeds x 32 starts; got {len(pred)} rows")
        seen = set()
        starts = defaultdict(set)
        for row in pred:
            seed, model, start = seed_id(row), row["model"], int(row["start_idx"])
            key = (model, seed, start)
            if key in seen or model not in MODELS:
                raise ValueError(f"Invalid or duplicate prediction row: {system} {key}")
            seen.add(key)
            starts[(model, seed)].add(start)
            for metric, column in METRICS.items():
                observations[(system, model, metric, seed)].append(number(row[column]))
        if len(starts) != 20 or any(len(v) != 32 for v in starts.values()):
            raise ValueError(f"{system}: incomplete model/seed groups")
        if len({tuple(sorted(v)) for v in starts.values()}) != 1:
            raise ValueError(f"{system}: rollout start indices differ between models/seeds")
        for row in tau:
            for index, section in enumerate(sections, 1):
                if system != "Lorenz63" and row["section_name"] != section:
                    continue
                column = section if system == "Lorenz63" else "tau_dist_W1"
                key = (system, row["model"], f"DRTD_Sigma{index}", seed_id(row))
                observations[key].append(number(row[column]))
        for model in MODELS:
            for seed in seeds[system]:
                for index in range(1, 4):
                    key = (system, model, f"DRTD_Sigma{index}", seed)
                    if len(observations[key]) != 1:
                        raise ValueError(f"Expected exactly one DRTD scalar slot: {key}")
        if len(tau) != (20 if system == "Lorenz63" else 60):
            raise ValueError(f"Unexpected DRTD row count: {system}")
    return observations, sources, seeds


def summarize(observations, seeds):
    pooled, per_seed, appendix = [], [], []
    for system in CONFIG:
        for table, metrics in TABLES.items():
            for model in MODELS:
                for metric in metrics:
                    identity = dict(table=table, system=system, model=model, metric=metric)
                    all_values, seed_values = [], []
                    for seed in seeds[system]:
                        values = observations[(system, model, metric, seed)]
                        summary = statistics(values)
                        per_seed.append(dict(identity, seed=seed, **summary))
                        all_values.extend(values)
                        seed_values.append(summary["mean"])
                    overall = statistics(all_values)
                    pooled.append(dict(identity, **overall))
                    cross_seed = statistics(seed_values)
                    matrix = dict(identity)
                    for index, (seed, value) in enumerate(zip(seeds[system], seed_values), 1):
                        matrix[f"seed_{index}"] = seed
                        matrix[f"value_{index}"] = value
                    matrix.update(overall_mean=overall["mean"], overall_std=overall["std"],
                                  overall_n=overall["n_valid"], median=cross_seed["median"],
                                  Q25=cross_seed["Q25"], Q75=cross_seed["Q75"],
                                  valid_seeds=cross_seed["n_valid"])
                    appendix.append(matrix)
    return pooled, per_seed, appendix


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return "--" if value is None else f"{value:.6g}"


def write_latex(path, pooled, appendix, seeds):
    lines = ["% Requires \\usepackage{booktabs,graphicx}",
             "% DRTD excludes missing seeds. Standard deviation uses ddof=0.",
             "% SD is the archived Attractor_W2 value, without an additional square root."]
    for table, metrics in TABLES.items():
        lines += [r"\begin{table*}[t]", r"\centering", f"\\caption{{Recomputed Table {table} statistics.}}",
                  r"\begin{tabular}{ll" + "r" * len(metrics) + "}", r"\toprule",
                  "System & Model & " + " & ".join(m.replace("_", r"\_") for m in metrics) + r" \\", r"\midrule"]
        for system in CONFIG:
            for model in MODELS:
                cells = [system, model]
                for metric in metrics:
                    row = next(r for r in pooled if (r["system"], r["model"], r["metric"]) == (system, model, metric))
                    entry = "$" + fmt(row["mean"]) + r"\pm " + fmt(row["std"]) + "$"
                    if metric.startswith("DRTD") and row["n_valid"] < 5:
                        entry += f" ($n={row['n_valid']}$)"
                    cells.append(entry)
                lines.append(" & ".join(cells) + r" \\")
            lines.append(r"\midrule")
        lines[-1] = r"\bottomrule"
        lines += [r"\end{tabular}", r"\end{table*}"]
    for system in CONFIG:
        for table in TABLES:
            lines += [r"\begin{table*}[t]", r"\centering", f"\\caption{{{system}: Table {table} seed-level results.}}",
                      r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
                      "Model & Metric & " + " & ".join(str(s) for s in seeds[system]) + r" & Overall mean $\pm$ std & Median & Q25 & Q75 \\", r"\midrule"]
            for row in appendix:
                if row["system"] != system or row["table"] != table:
                    continue
                cells = [row["model"], row["metric"].replace("_", r"\_")]
                cells += [fmt(row[f"value_{i}"]) for i in range(1, 6)]
                overall = "$" + fmt(row["overall_mean"]) + r"\pm " + fmt(row["overall_std"]) + "$"
                if row["metric"].startswith("DRTD") and row["valid_seeds"] < 5:
                    overall += f" ($n={row['valid_seeds']}$)"
                cells += [overall]
                cells += [fmt(row[k]) for k in ["median", "Q25", "Q75"]]
                lines.append(" & ".join(cells) + r" \\")
            lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "reference",
                        help="Root containing lorenz63/, ks/, cylinder/ raw evaluation CSVs.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tables")
    args = parser.parse_args()
    observations, sources, seeds = collect(args.input_root)
    pooled, per_seed, appendix = summarize(observations, seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "overall.csv", pooled)
    write_csv(args.output_dir / "per_seed.csv", per_seed)
    write_csv(args.output_dir / "appendix.csv", appendix)
    write_latex(args.output_dir / "tables.tex", pooled, appendix, seeds)
    manifest = dict(sources=sources, std="population, ddof=0; n=1 yields 0",
                    quartiles="linear interpolation across seed means/DRTD scalars",
                    prediction_grain="160 rollouts overall; 32 per seed", drtd_grain="up to 5 valid seed scalars",
                    missing_drtd=sum(r["n_missing"] for r in pooled if r["metric"].startswith("DRTD")),
                    sections={"Lorenz63": ["x=10 upward", "z=40 upward", "local maxima of z"],
                              "KS": ["u[0]=0 both", "u[1]=0 both", "u[2]=0 both"],
                              "CFD": ["u[6]=4 both", "u[4]=4 both", "u[5]=4 both"]})
    (args.output_dir / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(pooled)} overall rows, {len(per_seed)} seed rows, {len(appendix)} appendix rows to {args.output_dir}")
    print(f"Missing DRTD values (excluded): {manifest['missing_drtd']}")


if __name__ == "__main__":
    main()
