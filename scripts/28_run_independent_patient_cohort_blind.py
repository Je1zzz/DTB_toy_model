#!/usr/bin/env python
"""Run identical truth-free MAP inversion independently for every VEP patient."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src/vbt/inference/reference_engine.py"
FINALIZER = ROOT / "scripts/26_finalize_single_patient_blind.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--inner-parallel", type=int, default=4)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    starts = int(protocol["optimization"]["starts"])
    iterations = int(protocol["optimization"]["maximum_iterations_per_start"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(ROOT / "src"), "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})

    def run(subject: str) -> dict:
        target = args.output_root / subject
        manifest = target / "single_patient_blind_manifest.json"
        if args.reuse and manifest.exists():
            return {"subject": subject, "status": "REUSED", "seconds": 0.0}
        if target.exists() and any(target.iterdir()):
            return {"subject": subject, "status": "FAIL_PARTIAL_OUTPUT_EXISTS", "seconds": 0.0}
        target.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(ENGINE), "--blind-only", "--map-only", "--subject", subject,
            "--output", str(target), "--opt-starts", str(starts), "--best-inits", "4",
            "--opt-iter", str(iterations), "--max-parallel", str(args.inner_parallel),
            "--seed", str(260902 + int(subject[-3:]) * 1000),
        ]
        started = time.perf_counter()
        with (target / "batch.log").open("w", encoding="utf-8") as handle:
            return_code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env).returncode
        if return_code:
            return {"subject": subject, "status": "FAIL_INVERSION", "return_code": return_code, "seconds": time.perf_counter() - started}
        finalize = subprocess.run(
            [sys.executable, str(FINALIZER), "--run-dir", str(target), "--protocol", str(args.protocol)],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env,
        )
        return {
            "subject": subject,
            "status": "PASS" if finalize.returncode == 0 else "FAIL_FINALIZE",
            "return_code": finalize.returncode,
            "seconds": time.perf_counter() - started,
        }

    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run, subject): subject for subject in protocol["subjects"]}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    report = {
        "truth_loaded": False,
        "population_model_used": False,
        "subjects": len(results),
        "passed": sum(item["status"] in ("PASS", "REUSED") for item in results),
        "wall_seconds": time.perf_counter() - started,
        "results": sorted(results, key=lambda item: item["subject"]),
    }
    (args.output_root / "blind_batch_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("subjects", "passed", "wall_seconds", "truth_loaded")}, indent=2))
    return 0 if report["passed"] == report["subjects"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
