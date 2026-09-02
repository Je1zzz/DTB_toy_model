#!/usr/bin/env python
"""Visualize raw SEEG context and the observed-vs-predicted model target."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbt.inference.reference_engine import csv_header_and_rows


def matrix_from_row(header: list[str], row: np.ndarray, prefix: str) -> np.ndarray:
    pattern = re.compile(re.escape(prefix) + r"\.(\d+)\.(\d+)$")
    entries = []
    for column, name in enumerate(header):
        match = pattern.fullmatch(name)
        if match:
            entries.append((int(match.group(1)), int(match.group(2)), column))
    entries.sort()
    if not entries:
        raise ValueError(f"No {prefix} matrix in Stan output")
    shape = (max(item[0] for item in entries), max(item[1] for item in entries))
    if len(entries) != shape[0] * shape[1]:
        raise ValueError(f"Incomplete {prefix} matrix")
    return np.asarray([row[column] for _, _, column in entries]).reshape(shape)


def load_fit(run_dir: Path) -> dict:
    report = json.loads((run_dir / "map_report.json").read_text(encoding="utf-8"))
    prepared = np.load(run_dir / "prepared_inputs.npz")
    header, rows = csv_header_and_rows(run_dir / f"optimize-{report['best_start']}.csv")
    target = np.log(prepared["Obs"])
    prediction = matrix_from_row(header, rows[-1], "xhat_q")
    if target.shape != prediction.shape:
        raise ValueError(f"Target/prediction mismatch: {target.shape}, {prediction.shape}")
    valid_target = target[10:].ravel()
    valid_prediction = prediction[10:].ravel()
    residual = valid_target - valid_prediction
    denominator = np.sum((valid_target - valid_target.mean()) ** 2)
    temporal_target = target[10:] - target[10:].mean(axis=0, keepdims=True)
    temporal_prediction = prediction[10:] - prediction[10:].mean(axis=0, keepdims=True)
    temporal_residual = temporal_target - temporal_prediction
    return {
        "subject": run_dir.name,
        "target": target,
        "prediction": prediction,
        "raw": prepared["raw"],
        "sfreq": float(prepared["sfreq"]),
        "channel_names": tuple(str(value) for value in prepared["channel_names"]),
        "pearson": float(np.corrcoef(valid_target, valid_prediction)[0, 1]),
        "explained_variance": float(1.0 - np.sum(residual**2) / denominator),
        "nrmse": float(np.sqrt(np.mean(residual**2)) / np.ptp(valid_target)),
        "temporal_pearson_after_channel_demean": float(np.corrcoef(temporal_target.ravel(), temporal_prediction.ravel())[0, 1]),
        "temporal_explained_variance_after_channel_demean": float(
            1.0 - np.sum(temporal_residual**2) / np.sum(temporal_target**2)
        ),
        "prediction_to_observed_temporal_sd_ratio_mean": float(
            np.mean(prediction[10:].std(axis=0) / np.maximum(target[10:].std(axis=0), 1e-12))
        ),
    }


def detailed_figure(fit: dict, output: Path) -> None:
    target = fit["target"]
    prediction = fit["prediction"]
    residual = target - prediction
    top_channels = np.argsort(target[10:].var(axis=0))[-4:][::-1]
    common_low = float(min(target.min(), prediction.min()))
    common_high = float(max(target.max(), prediction.max()))
    residual_limit = float(np.quantile(np.abs(residual), 0.99))
    figure, axes = plt.subplots(3, 2, figsize=(16, 13), constrained_layout=True)

    raw = fit["raw"]
    stride = max(1, raw.shape[0] // 5000)
    raw_time = np.arange(0, raw.shape[0], stride) / fit["sfreq"]
    for offset, channel in enumerate(top_channels):
        trace = raw[::stride, channel]
        scale = np.std(trace) or 1.0
        axes[0, 0].plot(raw_time, trace / scale + 5 * offset, linewidth=0.55, label=fit["channel_names"][channel])
    axes[0, 0].set(title="Observed raw SEEG context (z-scaled and offset)", xlabel="seconds")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)

    image = axes[0, 1].imshow(target.T, aspect="auto", origin="lower", vmin=common_low, vmax=common_high, cmap="viridis")
    axes[0, 1].set(title="Observed model target: log(Obs)", xlabel="model time", ylabel="channel")
    figure.colorbar(image, ax=axes[0, 1], shrink=0.8)
    image = axes[1, 0].imshow(prediction.T, aspect="auto", origin="lower", vmin=common_low, vmax=common_high, cmap="viridis")
    axes[1, 0].set(title="Stan prediction: xhat_q", xlabel="model time", ylabel="channel")
    figure.colorbar(image, ax=axes[1, 0], shrink=0.8)
    image = axes[1, 1].imshow(residual.T, aspect="auto", origin="lower", vmin=-residual_limit, vmax=residual_limit, cmap="coolwarm")
    axes[1, 1].set(title="Residual: observed - predicted", xlabel="model time", ylabel="channel")
    figure.colorbar(image, ax=axes[1, 1], shrink=0.8)

    time = np.arange(target.shape[0])
    spacing = max(np.ptp(target), np.ptp(prediction)) * 1.1
    for offset, channel in enumerate(top_channels):
        axes[2, 0].plot(time, target[:, channel] + spacing * offset, color="black", linewidth=1.2)
        axes[2, 0].plot(time, prediction[:, channel] + spacing * offset, color="tab:orange", linewidth=1.0, alpha=0.9)
        axes[2, 0].text(time[-1] + 2, target[-1, channel] + spacing * offset, fit["channel_names"][channel], fontsize=8)
    axes[2, 0].set(
        title=("Selected channels: observed (black) vs predicted (orange)\n"
               f"channel-demeaned temporal r={fit['temporal_pearson_after_channel_demean']:.3f}, "
               f"EV={fit['temporal_explained_variance_after_channel_demean']:.3f}"),
        xlabel="model time",
    )

    x = target[10:].ravel()
    y = prediction[10:].ravel()
    axes[2, 1].scatter(x, y, s=4, alpha=0.15, color="tab:blue", rasterized=True)
    low, high = min(x.min(), y.min()), max(x.max(), y.max())
    axes[2, 1].plot([low, high], [low, high], "k--", linewidth=1)
    axes[2, 1].set(
        title=f"Predicted vs observed: r={fit['pearson']:.3f}, EV={fit['explained_variance']:.3f}, NRMSE={fit['nrmse']:.3f}",
        xlabel="observed log(Obs)", ylabel="predicted xhat_q", xlim=(low, high), ylim=(low, high),
    )
    figure.suptitle(f"{fit['subject']} observation fit (not raw-waveform prediction)", fontsize=16)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def cohort_gallery(fits: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(5, 6, figsize=(18, 15), constrained_layout=True)
    rng = np.random.default_rng(260903)
    for axis, fit in zip(axes.ravel(), fits):
        x = fit["target"][10:].ravel()
        y = fit["prediction"][10:].ravel()
        count = min(1200, x.size)
        selected = rng.choice(x.size, count, replace=False)
        low, high = min(x[selected].min(), y[selected].min()), max(x[selected].max(), y[selected].max())
        axis.scatter(x[selected], y[selected], s=3, alpha=0.18)
        axis.plot([low, high], [low, high], "k--", linewidth=0.7)
        axis.set(title=f"{fit['subject']}  r={fit['pearson']:.2f}  EV={fit['explained_variance']:.2f}", xlim=(low, high), ylim=(low, high))
        axis.tick_params(labelsize=7)
    figure.supxlabel("observed log(Obs)")
    figure.supylabel("predicted xhat_q")
    figure.suptitle("Thirty independent patient fits: predicted vs observed model target", fontsize=16)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detail-subject", default="sub-001")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fits = [load_fit(path) for path in sorted(args.run_root.glob("sub-*")) if (path / "map_report.json").exists()]
    if not fits:
        raise FileNotFoundError(f"No patient MAP outputs in {args.run_root}")
    selected = next(fit for fit in fits if fit["subject"] == args.detail_subject)
    detailed_figure(selected, args.output / f"{args.detail_subject}_observation_fit.png")
    cohort_gallery(fits, args.output / "cohort_observation_fit_gallery.png")
    metric_columns = [
        "subject", "pearson", "explained_variance", "nrmse",
        "temporal_pearson_after_channel_demean",
        "temporal_explained_variance_after_channel_demean",
        "prediction_to_observed_temporal_sd_ratio_mean",
    ]
    metrics = pd.DataFrame([{key: fit[key] for key in metric_columns} for fit in fits])
    metrics.to_csv(args.output / "observation_fit_visualization_metrics.csv", index=False)
    report = {
        "patients": len(fits), "detail_subject": args.detail_subject,
        "target": "log(Obs), derived from observed raw SEEG",
        "prediction": "Stan xhat_q = amplitude * gain * source_x + sensor_offset",
        "raw_waveform_prediction": False,
        "cohort_mean_sample_sd": {
            column: {"mean": float(metrics[column].mean()), "std": float(metrics[column].std(ddof=1))}
            for column in metric_columns if column != "subject"
        },
    }
    (args.output / "observation_fit_visualization_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
