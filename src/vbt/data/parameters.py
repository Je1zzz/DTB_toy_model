"""Strict parsers for the single-row parameter TSVs shipped with VEP v2.0."""
from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def _row(path: str | Path) -> dict[str, str]:
    source = Path(path)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        value = next(csv.DictReader(handle, delimiter="\t"), None)
    if value is None:
        raise ValueError(f"empty parameter TSV: {source}")
    return value


def parse_array(raw: str) -> np.ndarray:
    text = str(raw).strip()
    try:
        value = ast.literal_eval(text)
        array = np.asarray(value, dtype=float).reshape(-1)
    except (ValueError, SyntaxError):
        array = np.fromstring(text.replace("[", " ").replace("]", " ").replace(",", " "), sep=" ")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"invalid numeric array: {raw!r}")
    return array


def parse_labels(raw: str) -> tuple[str, ...]:
    value = ast.literal_eval(str(raw))
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"expected label list, got {raw!r}")
    return tuple(map(str, value))


@dataclass(frozen=True)
class EpileptorParameters:
    x0: np.ndarray; ez: tuple[str, ...]; pz: tuple[str, ...]
    r: np.ndarray; iext: np.ndarray; iext2: np.ndarray
    ks: np.ndarray; kf: np.ndarray; kvf: np.ndarray
    threshold: np.ndarray | None; r2: np.ndarray | None; parameter_file: Path


@dataclass(frozen=True)
class SimulatorParameters:
    coupling_factor: float; noise_coeffs: np.ndarray; init_cond: np.ndarray
    dt: float; period: float; simulation_length: float | None; sfreq: float | None
    parameter_file: Path


@dataclass(frozen=True)
class StimulationParameters:
    weights: np.ndarray; channels: tuple[str, ...]; stimulation_length: float
    onset: float; period_samples: float; amplitude: float; pulse_width: float
    description: str; parameter_file: Path


def _scalar(row: dict[str, str], key: str) -> float:
    value = parse_array(row[key])
    if value.size != 1: raise ValueError(f"{key} must be scalar")
    return float(value[0])


def load_epileptor_parameters(path: str | Path, n_regions: int = 162) -> EpileptorParameters:
    source, row = Path(path), _row(path)
    arrays = {key: parse_array(row[key]) for key in ("x0", "Iext", "Iext2", "Ks", "Kf", "Kvf")}
    for key, value in arrays.items():
        if value.size not in (1, n_regions): raise ValueError(f"{key}: {value.size}, expected 1 or {n_regions}")
        if value.size == 1: arrays[key] = np.repeat(value, n_regions)
    optional = lambda key: parse_array(row[key]) if key in row and str(row[key]).strip() else None
    threshold, r2 = optional("threshold"), optional("r2")
    if threshold is not None and threshold.size == 1: threshold = np.repeat(threshold, n_regions)
    return EpileptorParameters(arrays["x0"], parse_labels(row["EZ"]), parse_labels(row["PZ"]), parse_array(row["r"]), arrays["Iext"], arrays["Iext2"], arrays["Ks"], arrays["Kf"], arrays["Kvf"], threshold, r2, source)


def _initial_condition(raw: str, n_regions: int = 162) -> np.ndarray:
    if "..." in str(raw):
        heads=re.findall(r"\[\s*\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",str(raw))
        if len(heads) in (6,7): return np.repeat(np.asarray(heads,float)[:,None],n_regions,axis=1)
    values=parse_array(raw)
    if values.size in (6,7): return np.repeat(values[:,None],n_regions,axis=1)
    if values.size in (6*n_regions,7*n_regions): return values.reshape(-1,n_regions)
    raise ValueError(f"cannot restore initial condition shape from {values.size} serialized values")

def load_simulator_parameters(path: str | Path, n_regions: int = 162) -> SimulatorParameters:
    source, row = Path(path), _row(path)
    return SimulatorParameters(_scalar(row,"coupling_factor"), parse_array(row["noise_coeffs"]), _initial_condition(row["init_cond"],n_regions), _scalar(row,"dt"), _scalar(row,"period"), _scalar(row,"simulation_length") if row.get("simulation_length") else None, _scalar(row,"sfreq") if row.get("sfreq") else None, source)


def load_stimulation_parameters(path: str | Path, n_regions: int = 162) -> StimulationParameters:
    source, row = Path(path), _row(path)
    weights = parse_array(row["stimulation_weights"])
    if weights.size != n_regions: raise ValueError(f"stimulation_weights: {weights.size}, expected {n_regions}")
    channels = parse_labels(row["stim_channels"]) if str(row["stim_channels"]).lstrip().startswith(("[","(")) else (str(row["stim_channels"]),)
    return StimulationParameters(weights, channels, _scalar(row,"stimulation_length"), _scalar(row,"onset"), _scalar(row,"T"), _scalar(row,"I"), _scalar(row,"pulse_width"), str(row.get("desc", "")), source)
