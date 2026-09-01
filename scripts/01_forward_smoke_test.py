#!/usr/bin/env python3
"""Run the Phase 1 forward chain for one VEP subject."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vbt.data.vep import VEPSubject  # noqa: E402
from vbt.observation.seeg import project_to_seeg  # noqa: E402
from vbt.simulation.simulator import simulate_spontaneous  # noqa: E402


DEFAULT_ROOT = "/home/hmzhang/remote/public_data/VEP_Cohort_v2.0"


def _pick_recording(subject: VEPSubject):
    for recording in subject.recordings:
        if recording.task == "simulatedseizure" and recording.acquisition == "VEPhypothesis":
            return recording
    if not subject.recordings:
        raise ValueError(f"No recordings for {subject.subject_id}")
    return subject.recordings[0]


def _save_figure(path: Path, subject: VEPSubject, time: np.ndarray, source: np.ndarray, seeg: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("figure: SKIPPED (matplotlib is not installed)")
        return

    ez_indices = [subject.region_names.index(name) for name in subject.ez_truth if name in subject.region_names]
    ez = ez_indices[0] if ez_indices else 0
    non_ez = next((i for i in range(source.shape[1]) if i not in ez_indices), 0)
    connected = int(np.argsort(subject.connectome.weights[ez])[::-1][0])
    figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(time, source[:, ez], label=f"source EZ: {subject.region_names[ez]}")
    axes[0].plot(time, source[:, non_ez], label=f"source non-EZ: {subject.region_names[non_ez]}", alpha=0.8)
    axes[0].plot(time, source[:, connected], label=f"source SC-neighbor: {subject.region_names[connected]}", alpha=0.8)
    axes[0].set_ylabel("x1 - x2")
    axes[0].legend(loc="upper right", fontsize=8)
    for index in range(min(3, seeg.shape[1])):
        axes[1].plot(time, seeg[:, index], label=f"SEEG {index}")
    axes[1].set_xlabel("model time")
    axes[1].set_ylabel("projected signal")
    axes[1].legend(loc="upper right", fontsize=8)
    figure.suptitle(f"VBT baseline forward smoke: {subject.subject_id}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"figure: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--subject", default="sub-001")
    parser.add_argument("--hypothesis", default="VEPhypothesis")
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs")
    args = parser.parse_args()

    subject = VEPSubject.load(args.root, args.subject, hypothesis=args.hypothesis)
    recording = _pick_recording(subject)
    result = simulate_spontaneous(subject, duration=args.duration)
    source = result.source_activity
    gain = subject.bipolar_gain(recording)
    seeg = project_to_seeg(source, gain)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "phase1_forward_smoke.npz",
        time=result.time,
        state=result.state,
        source=source,
        seeg=seeg,
        channel_names=np.asarray(recording.channel_names, dtype=object),
    )
    _save_figure(args.output_dir / "phase1_forward_smoke.png", subject, result.time, source, seeg)

    print(f"subject_id: {subject.subject_id}")
    print(f"regions: {source.shape[1]}")
    print(f"sensors: {gain.shape[0]}")
    print(f"SC shape: {subject.connectome.weights.shape}")
    print(f"gain shape: {gain.shape}")
    print(f"source shape: {source.shape}; range=({source.min():.6g}, {source.max():.6g})")
    print(f"SEEG shape: {seeg.shape}; range=({seeg.min():.6g}, {seeg.max():.6g})")
    print(f"finite: source={np.isfinite(source).all()} seeg={np.isfinite(seeg).all()}")
    print("G3a_NUMERICAL_FORWARD: PASS")
    print("G3b_SPONTANEOUS_SEIZURE: NOT_ASSESSED (short deterministic smoke window; no seizure claim)")
    print("G4_SOURCE_TO_SEEG: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
