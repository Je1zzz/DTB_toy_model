#!/usr/bin/env python
"""Finalize one patient's MAP fit and freeze truth-free EZN predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbt.evaluation.ev import epileptogenicity_value, source_onsets
from vbt.evaluation.posterior_source import iter_stan_source_draws
from vbt.inference.reference_engine import csv_header_and_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_from_row(header: list[str], row: np.ndarray, prefix: str) -> np.ndarray:
    pattern = re.compile(re.escape(prefix) + r"\.(\d+)\.(\d+)$")
    entries = []
    for column, name in enumerate(header):
        match = pattern.fullmatch(name)
        if match:
            entries.append((int(match.group(1)), int(match.group(2)), column))
    if not entries:
        raise ValueError(f"No {prefix}.row.column values in Stan output")
    n_row = max(item[0] for item in entries)
    n_column = max(item[1] for item in entries)
    if len(entries) != n_row * n_column:
        raise ValueError(f"Incomplete {prefix} matrix in Stan output")
    entries.sort()
    return np.asarray([row[column] for _, _, column in entries]).reshape(n_row, n_column)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads((args.run_dir / "map_report.json").read_text(encoding="utf-8"))
    expected_starts = int(json.loads(args.protocol.read_text(encoding="utf-8"))["optimization"]["starts"])
    start_rows = []
    x0_by_start = []
    rank_by_start = []
    chain_hashes = {}
    for start in range(1, expected_starts + 1):
        path = args.run_dir / f"optimize-{start}.csv"
        header, rows = csv_header_and_rows(path)
        row = rows[-1]
        x0 = np.asarray([row[header.index(f"x0.{index}")] for index in range(1, 163)])
        x0_by_start.append(x0)
        rank_by_start.append(rankdata(-x0, method="average"))
        start_rows.append(
            {
                "start": start,
                "lp": float(row[header.index("lp__")]),
                "K": float(row[header.index("K")]),
                "amplitude": float(row[header.index("amp")]),
                "noise_sd": float(row[header.index("eps")]),
            }
        )
        chain_hashes[path.name] = sha256(path)
    x0_array = np.asarray(x0_by_start)
    rank_array = np.asarray(rank_by_start)
    best_start = int(report["best_start"])
    best_path = args.run_dir / f"optimize-{best_start}.csv"
    header, rows = csv_header_and_rows(best_path)
    best_row = rows[-1]
    best_x0 = x0_array[best_start - 1]

    prepared = np.load(args.run_dir / "prepared_inputs.npz")
    labels = tuple(str(value) for value in prepared["roi_names"])
    observation = np.log(prepared["Obs"])
    fitted = matrix_from_row(header, best_row, "xhat_q")
    if fitted.shape != observation.shape:
        raise ValueError(f"Fitted/observed shape mismatch: {fitted.shape}, {observation.shape}")
    fit_slice = (slice(10, None), slice(None))
    target = observation[fit_slice].ravel()
    prediction = fitted[fit_slice].ravel()
    residual = target - prediction
    denominator = np.sum((target - target.mean()) ** 2)
    fit_metrics = {
        "pearson": float(np.corrcoef(target, prediction)[0, 1]),
        "explained_variance": float(1.0 - np.sum(residual**2) / denominator),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "nrmse_by_range": float(np.sqrt(np.mean(residual**2)) / np.ptp(target)),
        "evaluated_values": int(target.size),
        "excluded_initial_model_time_bins": 10,
    }

    source = next(iter_stan_source_draws(best_path))
    onset, recruited = source_onsets(source)
    source_ev = epileptogenicity_value(source)
    pairwise_rho = [
        float(spearmanr(x0_array[i], x0_array[j]).statistic)
        for i in range(expected_starts) for j in range(i)
    ]
    frame = pd.DataFrame(
        {
            "roi_index": np.arange(len(labels)),
            "roi_name": labels,
            "map_x0": best_x0,
            "ezn_rank": rankdata(-best_x0, method="average"),
            "x0_across_start_mean": x0_array.mean(axis=0),
            "x0_across_start_sd": x0_array.std(axis=0),
            "rank_across_start_mean": rank_array.mean(axis=0),
            "rank_across_start_sd": rank_array.std(axis=0),
            "map_source_onset_index": onset,
            "map_source_recruited": recruited,
            "map_source_ev": source_ev,
        }
    ).sort_values(["ezn_rank", "roi_index"])
    output = args.run_dir / "single_patient_prediction_blind.csv"
    frame.to_csv(output, index=False)
    pd.DataFrame(start_rows).to_csv(args.run_dir / "optimization_start_summary_blind.csv", index=False)
    manifest = {
        "subject": report.get("subject", "sub-001"),
        "scope": "single patient; no population model",
        "truth_loaded": False,
        "prediction_rule": "descending MAP x0",
        "best_start": best_start,
        "best_lp": float(report["best_lp"]),
        "optimizations_attempted": expected_starts,
        "optimizations_usable": int(report["optimizations_usable"]),
        "observation_fit": fit_metrics,
        "multi_start_stability": {
            "pairwise_x0_rank_spearman_min": float(np.min(pairwise_rho)),
            "pairwise_x0_rank_spearman_median": float(np.median(pairwise_rho)),
            "x0_across_start_sd_median": float(np.median(x0_array.std(axis=0))),
            "x0_across_start_sd_max": float(np.max(x0_array.std(axis=0))),
        },
        "secondary_source_output": {
            "rule": "first x>0; no crossing is onset 200; Nature Computational Science 2025 EV",
            "recruited_rois": int(recruited.sum()),
            "status": "PASS" if recruited.any() else "FAIL_NO_SOURCE_RECRUITMENT",
        },
        "optimize_csv_sha256": chain_hashes,
        "prediction_file": output.name,
        "prediction_sha256": sha256(output),
        "protocol_sha256": sha256(args.protocol),
    }
    (args.run_dir / "single_patient_blind_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
