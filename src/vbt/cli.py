"""Small command line surface for the general DTB baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from vbt.inference.methods import METHODS, get_method
from vbt.readiness import audit_readiness


DEFAULT_DATA = "/home/hmzhang/remote/public_data/VEP_Cohort_v2.0"


def _methods() -> int:
    print(json.dumps({name: vars(item) for name, item in METHODS.items()}, indent=2))
    return 0


def _audit_data(args) -> int:
    # VEP Cohort v2 contract: synthetic SEEG, SC and gain; no patient raw anatomy/surface.
    assets = {"synthetic_truth", "seeg", "sc", "gain"}
    report = audit_readiness(assets)
    report["data_root"] = args.data
    report["classification"] = {
        "raw_patient_measurements_missing": ["T1/T2", "DWI", "implant CT/MRI", "clinical stimulation log"],
        "patient_derived_available_at_parcel_level": ["162-region SC", "electrode coordinates", "parcel gain", "synthetic stimulation weights"],
        "patient_derived_missing": ["patient surface/vertex areas", "vertex region map", "geodesics/local connectivity", "vertex gain", "original vertex stimulation field", "high-resolution delay history"],
        "general_code_available_or_reimplementable": ["Laplace kernel", "neural mass equations", "Heun integrator", "monitor logic"],
        "historical_state_unrecoverable": ["original RNG state", "vertex initial state", "delay history", "exact TVB/runtime state"],
    }
    report["capabilities"] = {"can_parcel_forward": True, "can_surface_forward": False, "can_invert_seeg": True, "can_seeg_stimulate": "PARCEL_ONLY", "can_ti_stimulate": False}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _forward(args) -> int:
    root = Path(__file__).resolve().parents[2]
    if args.task == "spontaneous":
        command = [sys.executable, str(root / "scripts/01b_reference_spontaneous.py"), "--subject", args.subject, "--data", args.data]
    else:
        command = [sys.executable, str(root / "scripts/run_profile.py"), "--profile", args.profile, "--subject", args.subject, "--data", args.data]
    return subprocess.run(command).returncode


def _infer(args) -> int:
    method = get_method(args.inversion)
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output) if args.output else root / "outputs" / "inversion" / method.name / args.subject
    command = [sys.executable, "-m", "vbt.inference.reference_engine", "--blind-only", "--forward-profile", args.profile, "--subject", args.subject, "--output", str(output), "--opt-starts", str(args.opt_starts), "--best-inits", "2", "--opt-iter", str(args.opt_iter), "--max-parallel", str(args.max_parallel)]
    if method.name == "map":
        command += ["--map-only"]
    else:
        command += ["--chains", str(args.chains), "--warmup", str(args.warmup), "--samples", str(args.samples)]
    return subprocess.run(command).returncode


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m vbt.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("methods")
    audit_ref = sub.add_parser("audit"); audit_ref.add_argument("--profile", choices=("default", "vep_25"), default="default")
    audit = sub.add_parser("audit-data"); audit.add_argument("--data", default=DEFAULT_DATA)
    forward = sub.add_parser("forward"); forward.add_argument("--profile", choices=("default", "vep_25"), default="default"); forward.add_argument("--subject", default="sub-001"); forward.add_argument("--task", choices=("spontaneous", "stimulation"), default="spontaneous"); forward.add_argument("--data", default=DEFAULT_DATA)
    infer = sub.add_parser("infer"); infer.add_argument("--profile", choices=("default", "vep_25"), default="default", help="provenance only; inversion is shared"); infer.add_argument("--inversion", choices=tuple(METHODS), default="map"); infer.add_argument("--subject", default="sub-001"); infer.add_argument("--output"); infer.add_argument("--opt-starts", type=int, default=8); infer.add_argument("--opt-iter", type=int, default=200); infer.add_argument("--chains", type=int, default=4); infer.add_argument("--warmup", type=int, default=50); infer.add_argument("--samples", type=int, default=50); infer.add_argument("--max-parallel", type=int, default=4)
    run = sub.add_parser("run"); run.add_argument("--profile", choices=("default", "vep_25"), default="default"); run.add_argument("--inversion", choices=tuple(METHODS), default="map"); run.add_argument("--subject", default="sub-001"); run.add_argument("--task", choices=("spontaneous",), default="spontaneous"); run.add_argument("--data", default=DEFAULT_DATA); run.add_argument("--output"); run.add_argument("--opt-starts", type=int, default=8); run.add_argument("--opt-iter", type=int, default=200); run.add_argument("--chains", type=int, default=4); run.add_argument("--warmup", type=int, default=50); run.add_argument("--samples", type=int, default=50); run.add_argument("--max-parallel", type=int, default=4)
    args = parser.parse_args()
    if args.command == "methods": return _methods()
    if args.command == "audit":
        if args.profile == "default": print(json.dumps({"profile": "default", "status": "COHORT_NATIVE"}, indent=2)); return 0
        root = Path(__file__).resolve().parents[2]
        return subprocess.run([sys.executable, str(root / "scripts/09_verify_repo_equivalence.py")]).returncode
    if args.command == "audit-data": return _audit_data(args)
    if args.command == "forward": return _forward(args)
    if args.command == "run":
        get_method(args.inversion)
        code = _forward(args)
        return code if code else _infer(args)
    return _infer(args)


if __name__ == "__main__":
    try: raise SystemExit(main())
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr); raise SystemExit(2)
