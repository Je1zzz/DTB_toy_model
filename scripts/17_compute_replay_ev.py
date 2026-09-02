#!/usr/bin/env python
"""Compute truth-free posterior/MAP predictive EV in the Stan feature space."""

import csv, json, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def stan_header(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                return next(csv.reader([line]))
    raise ValueError(f"No Stan header in {path}")


def selected_matrix(path, names):
    header = stan_header(path)
    indices = [header.index(name) for name in names]
    rows = []
    with path.open(encoding="utf-8") as handle:
        data_lines = (line for line in handle if not line.startswith("#"))
        reader = csv.reader(data_lines)
        next(reader)
        for row in reader:
            rows.append([float(row[index]) for index in indices])
    return np.atleast_2d(np.asarray(rows))


def predictive_columns(path):
    header = stan_header(path)
    found = []
    for name in header:
        match = re.fullmatch(r"xhat_q\.(\d+)\.(\d+)", name)
        if match:
            found.append((int(match.group(1)), int(match.group(2)), name))
    if not found:
        raise ValueError(f"xhat_q not found in {path}")
    return [name for _, _, name in sorted(found)]


def ev_rows(predictions, target):
    flat_target = target.reshape(-1)
    denominator = np.sum((flat_target - flat_target.mean()) ** 2)
    return 1.0 - np.sum((predictions - flat_target) ** 2, axis=1) / denominator


def evaluate_csv(csv_path, prepared):
    obs = np.load(prepared)["Obs"]
    target = np.log(obs)
    names = predictive_columns(csv_path)
    predictions = selected_matrix(csv_path, names)
    ev = ev_rows(predictions, target)
    mean_prediction = predictions.mean(axis=0)
    correlation = np.corrcoef(mean_prediction, target.reshape(-1))[0, 1]
    return {
        "draws": int(predictions.shape[0]),
        "ev_mean": float(ev.mean()),
        "ev_median": float(np.median(ev)),
        "ev_min": float(ev.min()),
        "ev_max": float(ev.max()),
        "posterior_mean_prediction_ev": float(ev_rows(mean_prediction[None, :], target)[0]),
        "posterior_mean_prediction_pearson": float(correlation),
    }


def main():
    started = time.perf_counter()
    rows = []
    for subject_dir in sorted((ROOT / "outputs/personalization_map").glob("sub-*")):
        report = json.loads((subject_dir / "map_report.json").read_text())
        csv_path = subject_dir / f"optimize-{report['best_start']}.csv"
        metrics = evaluate_csv(csv_path, subject_dir / "prepared_inputs.npz")
        rows.append({"subject": subject_dir.name, "method": "map", **metrics})

    nuts_dir = ROOT / "outputs/timing_nuts/sub-001"
    if nuts_dir.exists():
        per_chain = [evaluate_csv(path, nuts_dir / "prepared_inputs.npz") for path in sorted(nuts_dir.glob("chain-*.csv")) if "diagnostic" not in path.name]
        if per_chain:
            rows.append({
                "subject": "sub-001", "method": "nuts_smoke",
                "draws": sum(item["draws"] for item in per_chain),
                **{key: float(np.mean([item[key] for item in per_chain])) for key in per_chain[0] if key != "draws"},
            })

    frame = pd.DataFrame(rows)
    out = ROOT / "outputs/replay_ev"; out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "subject_metrics.csv", index=False)
    map_rows = frame[frame.method == "map"]
    summary = {
        "feature_space": "log(positive SEEG log-power features) used by reference_vep_mcmc.stan likelihood",
        "endpoint": "EV = 1 - SSE(predicted feature, observed feature) / SST(observed feature)",
        "map_subjects": int(len(map_rows)),
        "map_ev_mean": float(map_rows.posterior_mean_prediction_ev.mean()),
        "map_ev_median": float(map_rows.posterior_mean_prediction_ev.median()),
        "map_ev_positive_subjects": int((map_rows.posterior_mean_prediction_ev > 0).sum()),
        "nuts_smoke": frame[frame.method == "nuts_smoke"].to_dict("records"),
        "gate": "DESCRIPTIVE_ONLY until inversion diagnostics and held-out prediction pass",
        "not_raw_seeg_ev": True,
        "timing_seconds": {"wall": time.perf_counter() - started},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
