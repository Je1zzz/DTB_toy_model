#!/usr/bin/env python
"""Unlock EZ/PZ after all independent-patient prediction hashes are frozen."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbt.evaluation.ezn import evaluate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_evaluate(scores: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if not truth.any() or truth.all():
        return {"auroc": np.nan, "average_precision": np.nan, "prevalence": float(truth.mean()),
                "first_positive_rank": np.nan, "mean_positive_rank": np.nan, "recall_at_true_count": np.nan}
    value = evaluate(scores, truth)
    value["first_positive_rank"] = value.pop("first_ez_rank")
    value["mean_positive_rank"] = value.pop("mean_ez_rank")
    value["recall_at_true_count"] = value.pop("oracle_k_recall")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    batch = json.loads((args.run_root / "blind_batch_status.json").read_text(encoding="utf-8"))
    if batch["passed"] != len(protocol["subjects"]):
        raise RuntimeError("Not every independent-patient blind fit passed")

    rows = []
    hashes_before = {}
    for subject in protocol["subjects"]:
        run_dir = args.run_root / subject
        manifest = json.loads((run_dir / "single_patient_blind_manifest.json").read_text(encoding="utf-8"))
        prediction_path = run_dir / manifest["prediction_file"]
        current_hash = sha256(prediction_path)
        if current_hash != manifest["prediction_sha256"]:
            raise RuntimeError(f"Frozen prediction hash mismatch: {subject}")
        hashes_before[subject] = current_hash
        frame = pd.read_csv(prediction_path).sort_values("roi_index")
        parameter_path = (
            args.data_root / "derivatives" / "tvb" / subject / "ses-01" / "VEPhypothesis"
            / "parameters" / f"{subject}_epileptor_parameters_run-01.tsv"
        )
        with parameter_path.open("r", encoding="utf-8-sig", newline="") as handle:
            truth_row = next(csv.DictReader(handle, delimiter="\t"))
        ez = set(ast.literal_eval(truth_row["EZ"]))
        pz = set(ast.literal_eval(truth_row["PZ"]))
        scores = frame["map_x0"].to_numpy()
        report = json.loads((run_dir / "map_report.json").read_text(encoding="utf-8"))
        row = {
            "subject": subject,
            "engineering_pass": report["engineering_status"] == "PASS" and manifest["optimizations_usable"] == protocol["optimization"]["starts"],
            "optimizations_usable": manifest["optimizations_usable"],
            "observation_pearson": manifest["observation_fit"]["pearson"],
            "observation_explained_variance": manifest["observation_fit"]["explained_variance"],
            "observation_nrmse": manifest["observation_fit"]["nrmse_by_range"],
            "x0_rank_spearman_min": manifest["multi_start_stability"]["pairwise_x0_rank_spearman_min"],
            "x0_rank_spearman_median": manifest["multi_start_stability"]["pairwise_x0_rank_spearman_median"],
            "x0_across_start_sd_median": manifest["multi_start_stability"]["x0_across_start_sd_median"],
            "x0_across_start_sd_max": manifest["multi_start_stability"]["x0_across_start_sd_max"],
            "source_recruited_rois": manifest["secondary_source_output"]["recruited_rois"],
            "time_data_features_s": report["timing_seconds"]["data_and_features"],
            "time_optimization_s": report["timing_seconds"]["optimization"],
            "time_total_s": report["timing_seconds"]["total"],
        }
        for target_name, target in (("ez", ez), ("pz", pz), ("ez_or_pz", ez | pz)):
            truth = np.asarray([name in target for name in frame["roi_name"]], dtype=bool)
            row.update({f"{target_name}_{key}": value for key, value in safe_evaluate(scores, truth).items()})
        rows.append(row)

    subject_metrics = pd.DataFrame(rows).sort_values("subject")
    subject_metrics.to_csv(args.output / "independent_patient_subject_metrics.csv", index=False)
    metric_columns = [column for column in subject_metrics.columns if column not in ("subject", "engineering_pass")]
    summary_rows = []
    for metric in metric_columns:
        values = pd.to_numeric(subject_metrics[metric], errors="coerce").dropna()
        summary_rows.append({
            "metric": metric, "n": int(values.size), "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else np.nan,
            "median": float(values.median()), "min": float(values.min()), "max": float(values.max()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "independent_patient_mean_std_table.csv", index=False)
    display_metrics = [
        ("Observation fit", "observation_pearson", "Pearson correlation"),
        ("Observation fit", "observation_explained_variance", "Explained variance"),
        ("Observation fit", "observation_nrmse", "NRMSE by range"),
        ("Multi-start stability", "x0_rank_spearman_min", "Minimum pairwise x0 rank Spearman"),
        ("Multi-start stability", "x0_rank_spearman_median", "Median pairwise x0 rank Spearman"),
        ("Multi-start stability", "x0_across_start_sd_median", "Median ROI x0 SD"),
        ("EZ", "ez_auroc", "AUROC"),
        ("EZ", "ez_average_precision", "AUPRC"),
        ("EZ", "ez_recall_at_true_count", "Recall at true EZ count"),
        ("EZ", "ez_first_positive_rank", "First EZ rank"),
        ("PZ", "pz_auroc", "AUROC"),
        ("PZ", "pz_average_precision", "AUPRC"),
        ("PZ", "pz_recall_at_true_count", "Recall at true PZ count"),
        ("PZ", "pz_first_positive_rank", "First PZ rank"),
        ("EZ or PZ", "ez_or_pz_auroc", "AUROC"),
        ("EZ or PZ", "ez_or_pz_average_precision", "AUPRC"),
        ("EZ or PZ", "ez_or_pz_recall_at_true_count", "Recall at true positive count"),
        ("Time", "time_data_features_s", "Data and feature time (s)"),
        ("Time", "time_optimization_s", "Optimization time (s)"),
        ("Time", "time_total_s", "Total time per patient (s)"),
    ]
    lookup = summary.set_index("metric")
    markdown = [
        "# Independent single-patient fits: mean and sample SD",
        "",
        "No population model or cross-patient parameter sharing was used.",
        "",
        "| Category | Metric | n | Mean | SD | Mean ± SD |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for category, key, label in display_metrics:
        item = lookup.loc[key]
        markdown.append(
            f"| {category} | {label} | {int(item['n'])} | {item['mean']:.4f} | "
            f"{item['std']:.4f} | {item['mean']:.4f} ± {item['std']:.4f} |"
        )
    (args.output / "independent_patient_mean_std_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    hashes_after = {subject: sha256(args.run_root / subject / "single_patient_prediction_blind.csv") for subject in protocol["subjects"]}
    audit = {
        "subjects": len(rows),
        "independent_patient_fits": True,
        "population_model_used": False,
        "engineering_passed": int(subject_metrics["engineering_pass"].sum()),
        "patients_ez_auroc_ge_0_90": int((subject_metrics["ez_auroc"] >= 0.90).sum()),
        "patients_ez_auprc_ge_0_50": int((subject_metrics["ez_average_precision"] >= 0.50).sum()),
        "patients_both_ez_gates": int(((subject_metrics["ez_auroc"] >= 0.90) & (subject_metrics["ez_average_precision"] >= 0.50)).sum()),
        "batch_wall_seconds": batch["wall_seconds"],
        "prediction_hashes_unchanged": hashes_after == hashes_before,
        "inference_launched_by_unlock": False,
        "aggregation": protocol["aggregation"],
        "claim": protocol["claim_boundary"],
    }
    (args.output / "independent_patient_cohort_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
