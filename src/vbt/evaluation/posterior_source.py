"""Streaming summaries of Stan generated-quantity source trajectories."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from vbt.evaluation.ev import epileptogenicity_value, source_onsets


_SOURCE_COLUMN = re.compile(r"x\.(\d+)\.(\d+)$")


def source_column_layout(header: Sequence[str]) -> tuple[np.ndarray, int, int]:
    """Return column indices ordered as time-major ``[time, region]``."""
    indexed = []
    for column, name in enumerate(header):
        match = _SOURCE_COLUMN.fullmatch(name)
        if match:
            indexed.append((int(match.group(2)), int(match.group(1)), column))
    if not indexed:
        raise ValueError("Stan CSV has no x.<region>.<time> columns")
    n_time = max(item[0] for item in indexed)
    n_region = max(item[1] for item in indexed)
    expected = {(time, region) for time in range(1, n_time + 1) for region in range(1, n_region + 1)}
    observed = {(time, region) for time, region, _ in indexed}
    if observed != expected or len(indexed) != len(expected):
        raise ValueError("Stan source columns do not form a complete rectangular grid")
    ordered = np.asarray([column for _, _, column in sorted(indexed)], dtype=int)
    return ordered, n_time, n_region


def iter_stan_source_draws(path: str | Path) -> Iterator[np.ndarray]:
    """Stream source draws from one CmdStan CSV without retaining nuisance columns."""
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8", errors="strict") as handle:
        header = None
        for line in handle:
            if not line.startswith("#"):
                header = line.rstrip("\n").split(",")
                break
        if header is None:
            raise ValueError(f"No header in {source_path}")
        columns, n_time, n_region = source_column_layout(header)
        for row_number, line in enumerate(handle, start=2):
            if line.startswith("#") or not line.strip():
                continue
            row = np.fromstring(line, sep=",")
            if row.size != len(header):
                raise ValueError(
                    f"{source_path}:{row_number} has {row.size} values, expected {len(header)}"
                )
            yield row[columns].reshape(n_time, n_region)


def summarize_source_posterior(
    chain_paths: Sequence[str | Path], threshold: float = 0.0, no_seizure_time: int = 200
) -> dict[str, np.ndarray | int]:
    """Summarize onset, recruitment, and EV across all posterior draws."""
    onsets = []
    recruited = []
    evs = []
    n_time = None
    for path in chain_paths:
        for source in iter_stan_source_draws(path):
            if n_time is None:
                n_time = source.shape[0]
            elif source.shape[0] != n_time:
                raise ValueError("Source time dimension differs between chains")
            onset, recruitment = source_onsets(source, threshold, no_seizure_time)
            onsets.append(onset)
            recruited.append(recruitment)
            evs.append(epileptogenicity_value(source, threshold, no_seizure_time))
    if not onsets:
        raise ValueError("No posterior source draws found")
    onset_array = np.asarray(onsets, dtype=float)
    recruitment_array = np.asarray(recruited, dtype=float)
    ev_array = np.asarray(evs, dtype=float)
    return {
        "draws": len(onsets),
        "n_time": int(n_time),
        "onset_mean": onset_array.mean(axis=0),
        "onset_q05": np.quantile(onset_array, 0.05, axis=0),
        "onset_q50": np.quantile(onset_array, 0.50, axis=0),
        "onset_q95": np.quantile(onset_array, 0.95, axis=0),
        "recruitment_probability": recruitment_array.mean(axis=0),
        "ev_mean": ev_array.mean(axis=0),
        "ev_q05": np.quantile(ev_array, 0.05, axis=0),
        "ev_q50": np.quantile(ev_array, 0.50, axis=0),
        "ev_q95": np.quantile(ev_array, 0.95, axis=0),
    }
