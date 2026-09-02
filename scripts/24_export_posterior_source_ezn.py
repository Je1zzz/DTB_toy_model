#!/usr/bin/env python
"""Truth-free export of posterior source onset, recruitment, and EV ranking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbt.evaluation.posterior_source import summarize_source_posterior


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atlas_labels(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        labels = tuple(row["region_name"] for row in csv.DictReader(handle, delimiter="\t") if row["label"] != "0")
    if not labels:
        raise ValueError(f"No non-background atlas labels in {path}")
    return labels


def chain_files(subject_dir: Path) -> list[Path]:
    return sorted(
        (path for path in subject_dir.iterdir() if re.fullmatch(r"chain-\d+\.csv", path.name)),
        key=lambda path: int(path.stem.split("-")[1]),
    )


def inference_quality(report: dict) -> str:
    if "inference_quality_status" in report:
        return str(report["inference_quality_status"])
    diagnostics = report.get("diagnostics", [])
    total_draws = sum(item.get("draws", 0) for item in diagnostics)
    valid = (
        bool(diagnostics)
        and report.get("rhat_x0_max", float("inf")) < 1.05
        and sum(item.get("divergences", 0) for item in diagnostics) == 0
        and sum(item.get("max_treedepth_hits", 0) for item in diagnostics) / max(total_draws, 1) < 0.01
        and min(item.get("bfmi", 0.0) for item in diagnostics) >= 0.3
        and sum(item.get("rejected_nan_proposals", 0) for item in diagnostics) == 0
    )
    return "PASS" if valid else "FAIL_DIAGNOSTICS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--subject")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--no-seizure-time", type=int, default=200)
    args = parser.parse_args()

    labels = atlas_labels(args.atlas)
    subject_dirs = [args.inference_root / args.subject] if args.subject else sorted(args.inference_root.glob("sub-*"))
    status = []
    for subject_dir in subject_dirs:
        chains = chain_files(subject_dir)
        if not chains:
            raise FileNotFoundError(f"No chain-N.csv files in {subject_dir}")
        summary = summarize_source_posterior(chains, args.threshold, args.no_seizure_time)
        if len(labels) != len(summary["ev_mean"]):
            raise ValueError(f"Atlas has {len(labels)} labels but source has {len(summary['ev_mean'])} regions")
        frame = pd.DataFrame(
            {
                "roi_index": np.arange(len(labels)),
                "roi_name": labels,
                "posterior_mean_onset_index": summary["onset_mean"],
                "posterior_q05_onset_index": summary["onset_q05"],
                "posterior_q50_onset_index": summary["onset_q50"],
                "posterior_q95_onset_index": summary["onset_q95"],
                "recruitment_probability": summary["recruitment_probability"],
                "posterior_mean_ev": summary["ev_mean"],
                "posterior_q05_ev": summary["ev_q05"],
                "posterior_q50_ev": summary["ev_q50"],
                "posterior_q95_ev": summary["ev_q95"],
            }
        )
        frame["ev_rank_tie_aware"] = rankdata(-frame["posterior_mean_ev"].to_numpy(), method="average")
        frame = frame.sort_values(["ev_rank_tie_aware", "roi_index"])
        output = subject_dir / "source_ezn_blind.csv"
        frame.to_csv(output, index=False)
        run_report_path = subject_dir / "blind_run_report.json"
        run_report = json.loads(run_report_path.read_text(encoding="utf-8")) if run_report_path.exists() else {}
        manifest = {
            "subject": subject_dir.name,
            "method": "Wang et al. Nature Computational Science 2025 source EV",
            "doi": "10.1038/s43588-025-00841-6",
            "source_variable": "Stan generated quantity x from the inferred two-dimensional Epileptor",
            "onset_rule": f"first time index where x > {args.threshold}; otherwise {args.no_seizure_time}",
            "ev_rule": "-log(((onset-min(onset))+1)/20), normalized to [0,1] within each draw",
            "truth_loaded": False,
            "n_regions": len(labels),
            "n_time": summary["n_time"],
            "n_draws": summary["draws"],
            "inference_quality_status": inference_quality(run_report),
            "chain_sha256": {path.name: sha256(path) for path in chains},
            "output_file": output.name,
            "output_sha256": sha256(output),
        }
        manifest_path = subject_dir / "source_ezn_blind_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        status.append({"subject": subject_dir.name, "draws": summary["draws"], "quality": manifest["inference_quality_status"]})
        print(json.dumps(status[-1]), flush=True)
    print(json.dumps({"exported": len(status), "truth_loaded": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
