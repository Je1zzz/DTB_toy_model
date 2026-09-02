#!/usr/bin/env python
"""Run one of the only two supported VBT profiles."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbt.data.vep import VEPSubject
from vbt.profiles import get_profile
from vbt.simulation.profile_pipeline import run_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=("default", "vep_25"))
    parser.add_argument("--data", default="/home/hmzhang/remote/public_data/VEP_Cohort_v2.0")
    parser.add_argument("--subject", default="sub-002")
    parser.add_argument("--duration", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    subject = VEPSubject.load(args.data, args.subject)
    result = run_forward(subject, get_profile(args.profile), args.duration, args.seed)
    output = ROOT / "outputs" / "profiles" / args.profile / subject.subject_id
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "forward.npz", **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    order = np.argsort(result["forward_ezn_score"])[::-1][:10]
    report = {
        "profile": result["profile"], "subject": subject.subject_id,
        "duration": args.duration, "recording": result["recording"],
        "parameter_files": result["parameter_files"],
        "shapes": {k: list(v.shape) for k, v in result.items() if isinstance(v, np.ndarray)},
        "top10_forward_ezn": [str(result["region_names"][index]) for index in order],
        "inference": "Use scripts/05a-05c for reference Stan EZN; forward score is descriptive, not posterior EZN.",
        "vep_25_scope": "PARCEL_INPUT_APPROXIMATION" if args.profile == "vep_25" else "COHORT_NATIVE",
        "status": "ENGINEERING_FORWARD_PASS; SCIENTIFIC_VALIDATION_NOT_ESTABLISHED",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
