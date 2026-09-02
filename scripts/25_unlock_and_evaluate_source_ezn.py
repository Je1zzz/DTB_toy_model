#!/usr/bin/env python
"""Evaluate frozen posterior source-EV exports after explicit truth unlock."""

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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbt.evaluation.ezn import evaluate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labels_from_parameter_file(path: Path) -> tuple[set[str], set[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    return set(ast.literal_eval(row["EZ"])), set(ast.literal_eval(row["PZ"]))


def evaluate_if_defined(scores: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if not truth.any() or truth.all():
        return {
            "auroc": float("nan"), "average_precision": float("nan"),
            "prevalence": float(truth.mean()), "first_ez_rank": float("nan"),
            "mean_ez_rank": float("nan"), "oracle_k_recall": float("nan"),
        }
    return evaluate(scores, truth)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    subject_dirs = [args.inference_root / args.subject] if args.subject else sorted(args.inference_root.glob("sub-*"))
    subject_rows = []
    frozen_hashes = {}
    annotated_frames = []
    for subject_dir in subject_dirs:
        prediction_path = subject_dir / "source_ezn_blind.csv"
        manifest = json.loads((subject_dir / "source_ezn_blind_manifest.json").read_text(encoding="utf-8"))
        current_hash = sha256(prediction_path)
        if current_hash != manifest["output_sha256"]:
            raise RuntimeError(f"Frozen prediction hash mismatch for {subject_dir.name}")
        frozen_hashes[subject_dir.name] = current_hash
        frame = pd.read_csv(prediction_path).sort_values("roi_index")
        parameter_path = (
            args.data_root / "derivatives" / "tvb" / subject_dir.name / "ses-01"
            / "VEPhypothesis" / "parameters" / f"{subject_dir.name}_epileptor_parameters_run-01.tsv"
        )
        ez, pz = labels_from_parameter_file(parameter_path)
        scores = frame["posterior_mean_ev"].to_numpy()
        row = {"subject": subject_dir.name, "inference_quality_status": manifest["inference_quality_status"]}
        for name, target in (("ez", ez), ("pz", pz), ("ez_or_pz", ez | pz)):
            truth = np.asarray([label in target for label in frame["roi_name"]], dtype=bool)
            metric = evaluate_if_defined(scores, truth)
            metric["first_positive_rank"] = metric.pop("first_ez_rank")
            metric["mean_positive_rank"] = metric.pop("mean_ez_rank")
            metric["recall_at_truth_count"] = metric.pop("oracle_k_recall")
            row.update({f"{name}_{key}": value for key, value in metric.items()})
        subject_rows.append(row)
        frame["is_ez"] = frame["roi_name"].isin(ez)
        frame["is_pz"] = frame["roi_name"].isin(pz)
        frame.insert(0, "subject", subject_dir.name)
        annotated_frames.append(frame)

    metrics = pd.DataFrame(subject_rows).sort_values("subject")
    metrics.to_csv(args.output / "source_ezn_subject_metrics.csv", index=False)
    roi_results = pd.concat(annotated_frames, ignore_index=True)
    roi_results.to_csv(args.output / "source_ezn_roi_results.csv", index=False)
    after = {subject: sha256(args.inference_root / subject / "source_ezn_blind.csv") for subject in frozen_hashes}
    audit = {
        "subjects_evaluated": len(subject_rows),
        "prediction_hashes_unchanged": after == frozen_hashes,
        "inference_launched_by_unlock": False,
        "evaluation_output_only": True,
        "scientific_status": "PASS" if subject_rows and all(row["inference_quality_status"] == "PASS" for row in subject_rows) else "FAIL_DIAGNOSTICS",
    }
    recruited_by_subject = roi_results.groupby("subject")["recruitment_probability"].max().gt(0)
    cohort = {
        **audit,
        "subjects_with_any_recruited_roi": int(recruited_by_subject.sum()),
        "subjects_without_any_recruited_roi": int((~recruited_by_subject).sum()),
        "method_validity_gate": "PASS" if recruited_by_subject.all() else "FAIL_NO_POSTERIOR_RECRUITMENT",
        "metrics": {},
        "claim": "Engineering smoke evaluation only; not valid EZN evidence when diagnostics or recruitment gates fail.",
    }
    for target in ("ez", "pz", "ez_or_pz"):
        for metric in ("auroc", "average_precision", "recall_at_truth_count", "first_positive_rank"):
            values = metrics[f"{target}_{metric}"]
            cohort["metrics"][f"macro_{target}_{metric}"] = {
                "mean": None if not values.notna().any() else float(values.mean()),
                "median": None if not values.notna().any() else float(values.median()),
                "subjects_defined": int(values.notna().sum()),
            }
    (args.output / "source_ezn_cohort_summary.json").write_text(json.dumps(cohort, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    max_recruitment = roi_results.groupby("subject")["recruitment_probability"].max()
    axes[0].bar(np.arange(len(max_recruitment)), max_recruitment.to_numpy(), color="steelblue")
    axes[0].set(title="Maximum ROI recruitment probability", xlabel="subject", ylabel="posterior probability", ylim=(0, 1.05))
    for offset, (target, label) in enumerate((("ez", "EZ"), ("pz", "PZ"), ("ez_or_pz", "EZ or PZ"))):
        axes[1].scatter(np.arange(len(metrics)) + (offset - 1) * 0.12, metrics[f"{target}_auroc"], s=20, label=label)
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set(title="Tie-aware ROI identification AUROC", xlabel="subject", ylabel="AUROC", ylim=(0, 1.05))
    axes[1].legend(frameon=False)
    figure.suptitle(f"Posterior source EV cohort: {audit['scientific_status']}")
    figure.savefig(args.output / "source_ezn_cohort_qc.png", dpi=160)
    plt.close(figure)
    (args.output / "source_ezn_evaluation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
