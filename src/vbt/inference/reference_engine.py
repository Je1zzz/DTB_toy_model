#!/usr/bin/env python3
"""Exact fixed-tau VEP Stan engine, refactored from the audited reference workflow."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import configparser
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


PARAMETER_PREFIXES = ("z_init_star.", "x0_star.", "u_star.")
SCALAR_PARAMETERS = {"K_star", "amp_star", "log_eps_sq", "tau0_star"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="sub-002")
    parser.add_argument("--forward-profile", choices=("default", "vep_25"), default="default")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/home/hmzhang/remote/public_data/VEP_Cohort_v2.0/data/VirtualEpilepticCohort"
        ),
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(
            "/home/hmzhang/remote/项目/脑数字孪生/methods/VBT_baseline"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vhdr", type=Path)
    parser.add_argument("--sc", type=Path)
    parser.add_argument("--gain", type=Path)
    parser.add_argument("--electrodes", type=Path)
    parser.add_argument(
        "--binary",
        type=Path,
        # Byte-identical copy of the reference executable on an executable mount.
        default=Path("/data_hdd/hmzhang/vbt_runtime/vbt_stan_prebuilt/5367ef4afb976271ecc6297b70a95dd3d944af61/vep_mcmc"),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--map-only", action="store_true", help="stop after best L-BFGS estimate")
    parser.add_argument("--blind-only", action="store_true", help="stop after truth-free prediction artifacts")
    parser.add_argument("--n-time", type=int, default=150)
    parser.add_argument("--opt-iter", type=int, default=2000)
    parser.add_argument("--opt-starts", type=int, default=50)
    parser.add_argument("--best-inits", type=int, default=8)
    parser.add_argument("--chains", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--adapt-delta", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def read_brainvision(vhdr: Path) -> tuple[np.ndarray, float, list[str]]:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    text = vhdr.read_text(encoding="utf-8-sig")
    first_section = text.find("[Common Infos]")
    if first_section < 0:
        raise ValueError(f"{vhdr}: missing [Common Infos]")
    parser.read_string(text[first_section:], source=str(vhdr))
    common = parser["Common Infos"]
    binary = parser["Binary Infos"]
    if common["DataOrientation"].upper() != "MULTIPLEXED":
        raise ValueError("Only MULTIPLEXED BrainVision data are supported")
    if binary["BinaryFormat"].upper() != "IEEE_FLOAT_32":
        raise ValueError("Only IEEE_FLOAT_32 BrainVision data are supported")
    n_channels = int(common["NumberOfChannels"])
    sfreq = 1_000_000.0 / float(common["SamplingInterval"])
    names: list[str] = []
    resolutions: list[float] = []
    for index in range(1, n_channels + 1):
        fields = parser["Channel Infos"][f"Ch{index}"].split(",")
        names.append(fields[0].replace("\\1", ","))
        resolutions.append(float(fields[2] or 1.0))
    eeg = vhdr.with_name(common["DataFile"])
    flat = np.fromfile(eeg, dtype="<f4")
    if flat.size % n_channels:
        raise ValueError(f"{eeg}: {flat.size} values not divisible by {n_channels}")
    data = flat.reshape(-1, n_channels) * np.asarray(resolutions)[None, :]
    return data, sfreq, names


def read_connectome(path: Path) -> tuple[np.ndarray, list[str]]:
    with zipfile.ZipFile(path) as archive:
        sc = np.loadtxt(io.BytesIO(archive.read("weights.txt")))
        centres = archive.read("centres.txt").decode("utf-8").splitlines()
    labels = [line.split()[0] for line in centres if line.strip()]
    np.fill_diagonal(sc, 0.0)
    sc /= sc.max()
    return sc, labels


def bipolar_gain(
    contact_gain: np.ndarray, contact_names: list[str], channel_names: list[str]
) -> np.ndarray:
    name_to_index = {name: index for index, name in enumerate(contact_names)}
    rows = []
    missing = []
    for channel in channel_names:
        match = re.fullmatch(r"(.+?)(\d+)-(\d+)", channel)
        if not match:
            missing.append(channel)
            continue
        stem, first, second = match.groups()
        contact_a, contact_b = f"{stem}{first}", f"{stem}{second}"
        if contact_a not in name_to_index or contact_b not in name_to_index:
            missing.append(channel)
            continue
        rows.append(
            np.abs(
                contact_gain[name_to_index[contact_b]]
                - contact_gain[name_to_index[contact_a]]
            )
        )
    if missing:
        raise ValueError(f"Cannot map {len(missing)} bipolar channels: {missing[:10]}")
    return np.asarray(rows)


def log_power_features(data: np.ndarray, sfreq: float, n_time: int) -> np.ndarray:
    cleaned = data.astype(float, copy=True)
    for index in range(cleaned.shape[1]):
        values = cleaned[:, index]
        std = values.std()
        if std:
            mask = np.abs(values - values.mean()) > 2.0 * std
            values[mask] = values.mean()
    # Match vep_prepare_sim.compute_slp_sim and the reference notebook exactly:
    # third-order zero-phase filters, forward 100-sample log-power window,
    # a second outlier pass, then stride-based downsampling.
    b, a = signal.butter(3, 2.0 * 10.0 / sfreq, "highpass")
    cleaned = signal.filtfilt(b, a, cleaned, axis=0)
    window = min(100, cleaned.shape[0])
    padded = np.pad(cleaned, ((0, window), (0, 0)), mode="constant")
    features = np.empty_like(cleaned)
    for index in range(cleaned.shape[0]):
        power = np.mean(padded[index : index + window] ** 2, axis=0)
        features[index] = np.log(np.maximum(power, np.finfo(float).tiny))
    for index in range(features.shape[1]):
        values = features[:, index]
        std = values.std()
        if std:
            mask = np.abs(values - values.mean()) > 2.0 * std
            values[mask] = values.mean()
    b, a = signal.butter(3, 2.0 * 1.0 / sfreq, "lowpass")
    features = signal.filtfilt(b, a, features, axis=0)
    stride = max(1, int(features.shape[0] / n_time))
    features = features[0:-1:stride]
    return features - features.min() + 1.0


def rdump(path: Path, values: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for key, value in values.items():
            array = np.asarray(value)
            if array.ndim == 0:
                handle.write(f"{key} <- {array.item()}\n")
            elif array.ndim == 1:
                joined = ", ".join(f"{item:.17g}" for item in array)
                handle.write(f"{key} <- c({joined})\n")
            else:
                joined = ", ".join(f"{item:.17g}" for item in array.T.flat)
                dims = ", ".join(str(item) for item in array.shape)
                handle.write(f"{key} <- structure(c({joined}), .Dim = c({dims}))\n")


def csv_header_and_rows(path: Path) -> tuple[list[str], list[np.ndarray]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = (line.rstrip("\n") for line in handle if not line.startswith("#"))
        header = next(rows).split(",")
        values = [np.fromstring(line, sep=",") for line in rows if line]
    return header, values


def optimizer_parameters(optimize_csv: Path) -> tuple[float, dict[str, object]]:
    header, rows = csv_header_and_rows(optimize_csv)
    if not rows:
        raise RuntimeError("Optimize CSV has no result row")
    row = rows[-1]
    init: dict[str, object] = {}
    for prefix in PARAMETER_PREFIXES:
        indices = [i for i, name in enumerate(header) if name.startswith(prefix)]
        init[prefix[:-1]] = row[indices]
    for name in SCALAR_PARAMETERS:
        if name in header:
            init[name] = row[header.index(name)]
    return float(row[header.index("lp__")]), init


def write_chain_init(
    parameters: dict[str, object], init_path: Path, seed: int, jitter: float = 0.02
) -> None:
    rng = np.random.default_rng(seed)
    init = {
        name: np.asarray(value) + rng.normal(0.0, jitter, np.asarray(value).shape)
        for name, value in parameters.items()
    }
    init["K_star"] = max(float(init["K_star"]), -0.999)
    init["amp_star"] = max(float(init["amp_star"]), 1e-8)
    rdump(init_path, init)


def run_command(command: list[str], log: Path, env: dict[str, str]) -> int:
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
    return process.returncode


def rhat(chains: np.ndarray) -> np.ndarray:
    m, n, _ = chains.shape
    within = chains.var(axis=1, ddof=1).mean(axis=0)
    between = n * chains.mean(axis=1).var(axis=0, ddof=1)
    estimate = ((n - 1) / n * within + between / n) / within
    return np.sqrt(estimate)


def bfmi(energy: np.ndarray) -> float:
    variance = np.var(energy)
    return float(np.mean(np.diff(energy) ** 2) / variance) if variance else float("nan")


def auc(scores: np.ndarray, truth: np.ndarray) -> float:
    positives = scores[truth]
    negatives = scores[~truth]
    return float(
        (positives[:, None] > negatives[None, :]).mean()
        + 0.5 * (positives[:, None] == negatives[None, :]).mean()
    )


def load_ground_truth(path: Path, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path, sep="\t")
    text = str(frame.loc[0, "x0"]).replace("[", " ").replace("]", " ")
    x0 = np.fromstring(text, sep=" ")
    ez = list(ast.literal_eval(frame.loc[0, "EZ"]))
    if x0.size != len(labels):
        raise ValueError(f"Ground-truth x0 has {x0.size} values, expected {len(labels)}")
    return x0, ez


def main() -> int:
    wall_start = time.perf_counter()
    args = parse_args()
    subject = args.subject
    output = args.output or args.repo / "results/vep_cohort_e2e" / subject
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    subject_dir = args.dataset / subject
    struct_dir = args.dataset / "derivatives/tvb" / subject / "struct"
    vhdrs = sorted(subject_dir.glob("ses-*/ieeg/*.vhdr"))
    seizure_vhdrs = [
        path
        for path in vhdrs
        if "simulatedseizure" in path.name
        and "VEPhypothesis" in path.name
        and "run-01" in path.name
    ]
    if args.vhdr is not None:
        seizure_vhdrs=[args.vhdr]
    if len(seizure_vhdrs) != 1:
        raise ValueError(f"Expected one VEP-hypothesis run-01 seizure, found {seizure_vhdrs}")
    seizure_vhdr = seizure_vhdrs[0]
    discovered_at = time.perf_counter()

    raw, sfreq, channel_names = read_brainvision(seizure_vhdr)
    seeg_loaded_at = time.perf_counter()
    electrodes_path=args.electrodes or subject_dir / f"{subject}_electrodes.tsv"
    gain_path=args.gain or struct_dir / f"{subject}_gain.txt"
    sc_path=args.sc or struct_dir / f"{subject}_connectome.zip"
    electrodes = pd.read_csv(electrodes_path, sep="\t")
    contact_gain = np.loadtxt(gain_path)
    if contact_gain.shape[0] != len(electrodes):
        raise ValueError("Gain rows and electrode contacts differ")
    gain = bipolar_gain(contact_gain, electrodes["name"].tolist(), channel_names)
    sc, labels = read_connectome(sc_path)
    if sc.shape != (162, 162) or gain.shape != (len(channel_names), 162):
        raise ValueError(f"Unexpected SC/gain shapes: {sc.shape}, {gain.shape}")
    cerebellar = [
        labels.index("Left-Cerebellar-cortex"),
        labels.index("Right-Cerebellar-cortex"),
    ]
    gain[:, cerebellar] = gain.min()
    anatomy_loaded_at = time.perf_counter()
    obs = log_power_features(raw, sfreq, args.n_time)
    features_at = time.perf_counter()
    _, _, vh = np.linalg.svd(gain, full_matrices=True)
    eig = vh.T
    basis_at = time.perf_counter()
    if not all(np.isfinite(item).all() for item in (raw, gain, sc, obs, eig)):
        raise ValueError("Non-finite input detected")
    if not np.all(obs > 0):
        raise ValueError("Obs must be strictly positive")

    data_r = output / "stan_data.R"
    rdump(
        data_r,
        {
            "nn": sc.shape[0],
            "nt": obs.shape[0],
            "ns": obs.shape[1],
            "Obs": obs,
            "SC": sc,
            "gain": gain,
            "eig": eig,
        },
    )
    np.savez_compressed(
        output / "prepared_inputs.npz",
        raw=raw,
        sfreq=sfreq,
        channel_names=np.asarray(channel_names),
        gain=gain,
        SC=sc,
        Obs=obs,
        eig=eig,
        roi_names=np.asarray(labels),
    )
    serialized_at = time.perf_counter()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    seconds = np.arange(raw.shape[0]) / sfreq
    axes[0, 0].plot(seconds, raw[:, : min(8, raw.shape[1])])
    axes[0, 0].set(title="BrainVision seizure input (first 8 channels)", xlabel="s")
    image = axes[0, 1].imshow(obs.T, aspect="auto", origin="lower")
    axes[0, 1].set(title="Positive log-power target Obs", xlabel="model time", ylabel="channel")
    fig.colorbar(image, ax=axes[0, 1])
    image = axes[1, 0].imshow(sc, aspect="auto", origin="lower")
    axes[1, 0].set(title="Normalized structural connectivity")
    fig.colorbar(image, ax=axes[1, 0])
    image = axes[1, 1].imshow(gain, aspect="auto", origin="lower")
    axes[1, 1].set(title="Bipolar gain |G_B-G_A|", xlabel="region", ylabel="channel")
    fig.colorbar(image, ax=axes[1, 1])
    fig.savefig(output / "input_qc.png", dpi=150)
    plt.close(fig)
    qc_at = time.perf_counter()

    task_counts = {
        task: sum(task in path.name for path in vhdrs)
        for task in ("simulatedseizure", "simulatedstimulation", "simulatedinterictalspikes")
    }
    def file_sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    preflight = {
        "subject": subject,
        "selected_seizure": str(seizure_vhdr),
        "task_vhdr_counts": task_counts,
        "allowlisted_inputs": [str(seizure_vhdr),str(electrodes_path),str(gain_path),str(sc_path)],
        "gain_sha256": file_sha(gain_path),
        "sc_sha256": file_sha(sc_path),
        "eig_sha256": hashlib.sha256(np.ascontiguousarray(eig).tobytes()).hexdigest(),
        "raw_shape_time_by_channel": list(raw.shape),
        "sfreq_hz": sfreq,
        "SC_shape": list(sc.shape),
        "gain_shape": list(gain.shape),
        "Obs_shape": list(obs.shape),
        "cerebellar_gain_columns_suppressed": cerebellar,
        "strictly_positive_Obs": bool(np.all(obs > 0)),
        "finite_inputs": True,
        "stage_timing_seconds": {
            "input_discovery": discovered_at - wall_start,
            "seeg_read": seeg_loaded_at - discovered_at,
            "electrodes_gain_sc_and_bipolarization": anatomy_loaded_at - seeg_loaded_at,
            "seeg_log_power_features": features_at - anatomy_loaded_at,
            "gain_svd_basis": basis_at - features_at,
            "stan_and_npz_serialization": serialized_at - basis_at,
            "input_qc_render": qc_at - serialized_at,
            "preflight_hashes_and_manifest": time.perf_counter() - qc_at,
        },
    }
    (output / "preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    prepared_at = time.perf_counter()

    if args.prepare_only:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return 0

    binary = args.binary
    if not binary.exists():
        raise FileNotFoundError(f"Stan executable is absent: {binary}")
    env = os.environ.copy()
    legacy_tbb = Path("/data_hdd/hmzhang/vbt_ins_runtime/lib/usr/lib/x86_64-linux-gnu")
    if legacy_tbb.is_dir():
        env["LD_LIBRARY_PATH"] = str(legacy_tbb) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    started = time.time()

    optimize_jobs = []
    for start in range(1, args.opt_starts + 1):
        optimize_csv = output / f"optimize-{start}.csv"
        log_path = output / f"logs/optimize-{start}.log"
        optimize_command = [
            str(binary),
            "optimize",
            "algorithm=lbfgs",
            f"iter={args.opt_iter}",
            "save_iterations=0",
            "data",
            f"file={data_r}",
            "init=0" if start == 1 else "init=0.25",
            "random",
            f"seed={args.seed + start}",
            "output",
            f"file={optimize_csv}",
            "refresh=25",
        ]
        optimize_jobs.append((start,optimize_csv,log_path,optimize_command))

    def execute(job):
        index,csv_path,log_path,command=job
        with log_path.open("w",encoding="utf-8") as handle: code=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,env=env).returncode
        return index,csv_path,code
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        completed=list(pool.map(execute,optimize_jobs))
    optimized_at = time.perf_counter()
    optimize_results=[]
    for start, optimize_csv, return_code in completed:
        result = {"start": start, "return_code": return_code, "lp": None}
        if return_code == 0 and optimize_csv.exists():
            try:
                lp, parameters = optimizer_parameters(optimize_csv)
                result.update({"lp": lp, "parameters": parameters})
            except Exception as exc:
                result["parse_error"] = str(exc)
        optimize_results.append(result)
    valid_optimizations = [item for item in optimize_results if item["lp"] is not None]
    if not valid_optimizations:
        raise RuntimeError(f"All optimizations failed: {optimize_results}")
    best_optimizations = sorted(valid_optimizations, key=lambda item: item["lp"], reverse=True)[:args.best_inits]
    best_optimization = best_optimizations[0]

    if args.map_only:
        best_csv = output / f"optimize-{best_optimization['start']}.csv"
        header, rows = csv_header_and_rows(best_csv)
        row = rows[-1]
        x0_columns = [header.index(f"x0.{index}") for index in range(1, 163)]
        map_x0 = row[x0_columns]
        prediction = pd.DataFrame({
            "roi_index": np.arange(162), "roi_name": labels,
            "map_x0": map_x0,
        })
        prediction["rank"] = prediction["map_x0"].rank(method="first", ascending=False).astype(int)
        prediction.sort_values("rank").to_csv(output / "prediction_map.csv", index=False)
        report = {
            "method": "map", "engineering_status": "PASS",
            "forward_profile": args.forward_profile, "inverse_model": "vep_reduced_2d",
            "scientific_status": "POINT_ESTIMATE_ONLY_NOT_POSTERIOR",
            "best_start": best_optimization["start"], "best_lp": best_optimization["lp"],
            "optimizations_attempted": args.opt_starts, "optimizations_usable": len(valid_optimizations),
            "EV_status": "UNAVAILABLE_NO_POSTERIOR_SOURCE_TRAJECTORY",
            "timing_seconds": {"data_and_features": prepared_at-wall_start, "optimization": optimized_at-prepared_at, "total": time.perf_counter()-wall_start},
        }
        (output / "map_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    chain_init_files = []
    for chain in range(1, args.chains + 1):
        init_r = output / f"chain-{chain}-init.R"
        source_init=best_optimizations[min((chain-1)//2,len(best_optimizations)-1)]
        write_chain_init(source_init["parameters"], init_r, args.seed + 1000 + chain)
        chain_init_files.append(init_r)

    sample_jobs=[]
    for chain in range(1, args.chains + 1):
        csv_path = output / f"chain-{chain}.csv"
        diagnostic_path = output / f"chain-{chain}-diagnostic.csv"
        log_path=output / f"logs/chain-{chain}.log"
        command = [
            str(binary),
            f"id={chain}",
            "sample",
            "save_warmup=0",
            f"num_warmup={args.warmup}",
            f"num_samples={args.samples}",
            "adapt",
            f"delta={args.adapt_delta}",
            "algorithm=hmc",
            "engine=nuts",
            f"max_depth={args.max_depth}",
            "random",
            f"seed={args.seed + 2000 + chain}",
            "data",
            f"file={data_r}",
            f"init={chain_init_files[chain - 1]}",
            "output",
            f"file={csv_path}",
            f"diagnostic_file={diagnostic_path}",
            "refresh=5",
        ]
        sample_jobs.append((chain,csv_path,log_path,command))
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        sample_completed=list(pool.map(execute,sample_jobs))
    sampled_at = time.perf_counter()
    return_codes={str(chain):code for chain,_,code in sample_completed}
    if any(return_codes.values()):
        raise RuntimeError(f"Sampling failed: {return_codes}")

    chain_x0 = []
    diagnostics = []
    for chain in range(1, args.chains + 1):
        header, rows = csv_header_and_rows(output / f"chain-{chain}.csv")
        matrix = np.vstack(rows)
        chain_log = (output / f"logs/chain-{chain}.log").read_text(
            encoding="utf-8", errors="replace"
        )
        x0_indices = [header.index(f"x0.{index}") for index in range(1, 163)]
        chain_x0.append(matrix[:, x0_indices])
        diagnostics.append(
            {
                "chain": chain,
                "draws": int(matrix.shape[0]),
                "divergences": int(matrix[:, header.index("divergent__")].sum()),
                "max_treedepth_hits": int(
                    (matrix[:, header.index("treedepth__")] >= args.max_depth).sum()
                ),
                "mean_accept_stat": float(matrix[:, header.index("accept_stat__")].mean()),
                "bfmi": bfmi(matrix[:, header.index("energy__")]),
                "rejected_nan_proposals": chain_log.count(
                    "about to be rejected because of the following issue"
                ),
            }
        )
    chain_x0_array = np.stack(chain_x0)
    posterior_x0 = chain_x0_array.mean(axis=(0, 1))
    x0_rhat = rhat(chain_x0_array)
    flat_x0 = chain_x0_array.reshape(-1, chain_x0_array.shape[-1])
    prediction = pd.DataFrame(
        {
            "roi_index": np.arange(162),
            "roi_name": labels,
            "posterior_mean_x0": flat_x0.mean(axis=0),
            "posterior_sd_x0": flat_x0.std(axis=0, ddof=1),
            "posterior_q05_x0": np.quantile(flat_x0, 0.05, axis=0),
            "posterior_q50_x0": np.quantile(flat_x0, 0.50, axis=0),
            "posterior_q95_x0": np.quantile(flat_x0, 0.95, axis=0),
            "rhat_x0": x0_rhat,
        }
    )
    prediction["rank"] = prediction["posterior_mean_x0"].rank(
        method="first", ascending=False
    ).astype(int)
    prediction["ezn_score"] = prediction["posterior_mean_x0"]
    prediction = prediction.sort_values("rank")
    prediction.to_csv(output / "prediction_blind.csv", index=False)
    prediction.to_csv(output / "prediction.csv", index=False)
    blind_metadata = {
        "subject": subject,
        "forward_profile": args.forward_profile,
        "inverse_model": "vep_reduced_2d",
        "inversion_method": "nuts",
        "created_before_ground_truth_load": True,
        "EZN_rule": "rank by posterior mean x0; larger is more epileptogenic",
        "top_10_roi_names": prediction.head(10)["roi_name"].tolist(),
        "n_total_draws": int(flat_x0.shape[0]),
        "model_binary": str(binary),
        "model_binary_sha256": subprocess.check_output(
            ["sha256sum", str(binary)], text=True
        ).split()[0],
    }
    (output / "prediction_blind.json").write_text(
        json.dumps(blind_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.blind_only:
        quality_pass = (
            np.nanmax(x0_rhat) < 1.05
            and sum(item["divergences"] for item in diagnostics) == 0
            and sum(item["max_treedepth_hits"] for item in diagnostics) / (args.chains * args.samples) < 0.01
            and min(item["bfmi"] for item in diagnostics) >= 0.3
            and sum(item["rejected_nan_proposals"] for item in diagnostics) == 0
        )
        report={"engineering_status":"PASS","inference_quality_status":"PASS" if quality_pass else "FAIL_DIAGNOSTICS","forward_profile":args.forward_profile,"inverse_model":"vep_reduced_2d","inversion_method":"nuts","EV_status":"UNAVAILABLE_POSTERIOR_SOURCE_REPLAY_NOT_EXPORTED","subject":subject,"truth_loaded":False,"optimizations_attempted":args.opt_starts,"optimizations_usable":len(valid_optimizations),"chains":args.chains,"chain_return_codes":return_codes,"rhat_x0_max":float(np.nanmax(x0_rhat)),"diagnostics":diagnostics,"timing_seconds":{"data_and_features":prepared_at-wall_start,"optimization":optimized_at-prepared_at,"sampling":sampled_at-optimized_at,"postprocess":time.perf_counter()-sampled_at,"total":time.perf_counter()-wall_start}}
        (output/"blind_run_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report,indent=2)); return 0

    stansummary_binary = Path(
        "/data_hdd/hmzhang/vbt_runtime/cmdstan-2.21.0/bin/stansummary"
    )
    stansummary_csv = output / "stansummary.csv"
    stansummary_rc = run_command(
        [
            str(stansummary_binary),
            f"--csv_file={stansummary_csv}",
            *[str(output / f"chain-{chain}.csv") for chain in range(1, args.chains + 1)],
        ],
        output / "logs/stansummary.log",
        env,
    )
    gt_path = (
        args.dataset
        / "derivatives/tvb"
        / subject
        / "ses-01/clinicalhypothesis/parameters"
        / f"{subject}_epileptor_parameters_run-01.tsv"
    )
    gt_x0, gt_ez = load_ground_truth(gt_path, labels)
    truth = np.asarray([label in gt_ez for label in labels])
    k = len(gt_ez)
    inferred_indices = np.argsort(posterior_x0)[-k:][::-1]
    inferred_ez = [labels[index] for index in inferred_indices]
    overlap = sorted(set(inferred_ez).intersection(gt_ez))
    optimization_maximum_iterations_hit = any(
        "Maximum number of iterations hit"
        in (output / f"logs/optimize-{start}.log").read_text(
            encoding="utf-8", errors="replace"
        )
        for start in range(1, args.opt_starts + 1)
    )
    all_divergences = sum(item["divergences"] for item in diagnostics)
    all_treedepth_hits = sum(item["max_treedepth_hits"] for item in diagnostics)
    all_nan_rejections = sum(item["rejected_nan_proposals"] for item in diagnostics)
    diagnostics_pass = (
        np.nanmax(x0_rhat) < 1.05
        and all_divergences == 0
        and all_treedepth_hits / (args.chains * args.samples) < 0.01
        and min(item["bfmi"] for item in diagnostics) >= 0.3
        and all_nan_rejections == 0
    )
    summary = {
        "engineering_status": "PASS",
        "scientific_status": "PASS" if diagnostics_pass else "FAIL_DIAGNOSTICS",
        "subject": subject,
        "elapsed_seconds": time.time() - started,
        "configuration": {
            "n_time": args.n_time,
            "opt_iter": args.opt_iter,
            "opt_starts": args.opt_starts,
            "chains": args.chains,
            "warmup": args.warmup,
            "samples": args.samples,
            "max_depth": args.max_depth,
            "adapt_delta": args.adapt_delta,
            "seed": args.seed,
        },
        "optimizations": [
            {key: value for key, value in item.items() if key != "parameters"}
            for item in optimize_results
        ],
        "best_optimization_start": best_optimization["start"],
        "best_optimization_lp": best_optimization["lp"],
        "optimization_maximum_iterations_hit": optimization_maximum_iterations_hit,
        "chain_return_codes": return_codes,
        "stansummary_return_code": stansummary_rc,
        "diagnostics": diagnostics,
        "rhat_x0_max": float(np.nanmax(x0_rhat)),
        "rhat_x0_median": float(np.nanmedian(x0_rhat)),
        "ground_truth_EZ": gt_ez,
        "inferred_top_k_EZ": inferred_ez,
        "top_k_overlap": overlap,
        "top_k_recall": len(overlap) / k,
        "EZ_AUROC_from_posterior_x0": auc(posterior_x0, truth),
        "pearson_posterior_vs_ground_truth_x0": float(np.corrcoef(posterior_x0, gt_x0)[0, 1]),
        "limitations": [
            "Ground truth is synthetic simulation ground truth, not clinical truth.",
            "Stimulation and interictal modalities are inventoried but this Stan model fits seizure log-power only.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True)
    axes[0].plot(gt_x0, label="synthetic ground truth x0", alpha=0.8)
    axes[0].plot(posterior_x0, label="posterior mean x0", alpha=0.8)
    axes[0].scatter(np.flatnonzero(truth), gt_x0[truth], color="red", label="ground-truth EZ")
    axes[0].set(title="EZN parameter recovery smoke check", xlabel="VEP region index", ylabel="x0")
    axes[0].legend()
    axes[1].bar(np.arange(162), posterior_x0)
    axes[1].scatter(inferred_indices, posterior_x0[inferred_indices], color="orange", label=f"inferred top-{k}")
    axes[1].set(title="Posterior mean ranking", xlabel="VEP region index", ylabel="posterior mean x0")
    axes[1].legend()
    fig.savefig(output / "posterior_qc.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
