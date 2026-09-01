#!/usr/bin/env python3
"""Audit one VEP subject without changing the source dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vbt.data.vep import VEPSubject  # noqa: E402


DEFAULT_ROOT = "/home/hmzhang/remote/public_data/VEP_Cohort_v2.0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--subject", default="sub-001")
    parser.add_argument("--hypothesis", default="VEPhypothesis")
    parser.add_argument("--output", type=Path, default=REPO / "outputs" / "phase0_audit.json")
    args = parser.parse_args()

    subject = VEPSubject.load(args.root, args.subject, hypothesis=args.hypothesis)
    recordings_by_task = Counter(recording.task for recording in subject.recordings)
    recordings_by_acquisition = Counter(recording.acquisition or "none" for recording in subject.recordings)
    summary = {
        "dataset_root": str(subject.root),
        "available_subjects": len(VEPSubject.available_subjects(args.root)),
        "subject_id": subject.subject_id,
        "regions": len(subject.region_names),
        "region_first": subject.region_names[:3],
        "region_last": subject.region_names[-3:],
        "sc_raw_shape": list(subject.connectome.raw_weights.shape),
        "sc_model_shape": list(subject.connectome.weights.shape),
        "sc_raw_finite": bool(np.isfinite(subject.connectome.raw_weights).all()),
        "sc_raw_symmetric": bool(np.allclose(subject.connectome.raw_weights, subject.connectome.raw_weights.T)),
        "sc_raw_diagonal_min": float(np.diag(subject.connectome.raw_weights).min()),
        "sc_model_diagonal_nonzero": int(np.count_nonzero(np.diag(subject.connectome.weights))),
        "gain_shape": list(subject.gain.shape),
        "gain_finite": bool(np.isfinite(subject.gain).all()),
        "electrode_rows": len(subject.electrodes),
        "recordings": len(subject.recordings),
        "recordings_by_task": dict(recordings_by_task),
        "recordings_by_acquisition": dict(recordings_by_acquisition),
        "ez_truth": list(subject.ez_truth),
        "pz_truth": list(subject.pz_truth),
        "model_parameters": str(subject.model_parameters.source_path),
        "simulator_parameters": str(subject.simulator_parameters.source_path),
        "dt": subject.simulator_parameters.dt,
        "period": subject.simulator_parameters.period,
        "coupling_factor": subject.simulator_parameters.coupling_factor,
        "source_npz_count": len(subject.source_files()),
        "source_npz_example": str(subject.source_files()[0]) if subject.source_files() else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
