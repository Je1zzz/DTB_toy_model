#!/usr/bin/env python
"""Unlock synthetic EZ/PZ only after one patient's prediction is frozen."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parameter-file", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.run_dir / "single_patient_blind_manifest.json").read_text(encoding="utf-8"))
    prediction_path = args.run_dir / manifest["prediction_file"]
    if sha256(prediction_path) != manifest["prediction_sha256"]:
        raise RuntimeError("Frozen single-patient prediction hash mismatch")
    if sha256(args.protocol) != manifest["protocol_sha256"]:
        raise RuntimeError("Frozen protocol hash mismatch")
    with args.parameter_file.open("r", encoding="utf-8-sig", newline="") as handle:
        truth_row = next(csv.DictReader(handle, delimiter="\t"))
    ez = set(ast.literal_eval(truth_row["EZ"]))
    pz = set(ast.literal_eval(truth_row["PZ"]))
    frame = pd.read_csv(prediction_path).sort_values("roi_index")
    scores = frame["map_x0"].to_numpy()
    metrics = {}
    for name, target in (("ez", ez), ("pz", pz), ("ez_or_pz", ez | pz)):
        truth = np.asarray([label in target for label in frame["roi_name"]], dtype=bool)
        value = evaluate(scores, truth)
        value["recall_at_true_count"] = value.pop("oracle_k_recall")
        value["first_positive_rank"] = value.pop("first_ez_rank")
        value["mean_positive_rank"] = value.pop("mean_ez_rank")
        metrics[name] = value
    ordered = frame.sort_values(["ezn_rank", "roi_index"])
    top_three = set(ordered.head(3)["roi_name"])
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    gates = protocol["success_gates"]
    checks = {
        "engineering_status": manifest["optimizations_usable"] >= gates["usable_optimizations_min"],
        "finite_observation_fit": all(np.isfinite(list(manifest["observation_fit"].values()))),
        "EZ_AUROC": metrics["ez"]["auroc"] >= gates["EZ_AUROC_min"],
        "EZ_AUPRC": metrics["ez"]["average_precision"] >= gates["EZ_AUPRC_min"],
        "recall_at_three": len(top_three & ez) / len(ez) >= gates["recall_at_three_min"],
        "first_EZ_rank": metrics["ez"]["first_positive_rank"] <= gates["first_EZ_rank_max"],
    }
    frame["is_EZ"] = frame["roi_name"].isin(ez)
    frame["is_PZ"] = frame["roi_name"].isin(pz)
    frame.sort_values(["ezn_rank", "roi_index"]).to_csv(args.run_dir / "single_patient_ezn_unlocked.csv", index=False)
    report = {
        "subject": manifest["subject"],
        "scope": "single patient; no population model",
        "prediction_hash_verified": True,
        "inference_launched_by_unlock": False,
        "metrics": metrics,
        "top_three_roi_names": ordered.head(3)["roi_name"].tolist(),
        "top_three_EZ_recall": len(top_three & ez) / len(ez),
        "EZ_ranks": {label: float(frame.set_index("roi_name").loc[label, "ezn_rank"]) for label in sorted(ez)},
        "PZ_ranks": {label: float(frame.set_index("roi_name").loc[label, "ezn_rank"]) for label in sorted(pz)},
        "success_gate_checks": checks,
        "overall_status": "PASS" if all(checks.values()) else "FAIL",
        "claim": protocol["claim_boundary"],
    }
    (args.run_dir / "single_patient_ezn_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
