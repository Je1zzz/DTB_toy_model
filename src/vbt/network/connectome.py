"""Checked loading and model-ready normalization of TVB connectivity files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np


@dataclass(frozen=True)
class Connectome:
    region_labels: tuple[str, ...]
    raw_weights: np.ndarray
    weights: np.ndarray

    @property
    def cohort_weights(self) -> np.ndarray:
        """VEP generator convention: zero diagonal, divide by raw maximum."""
        return normalize_weights(self.raw_weights, log_transform=False)

    @property
    def legacy_weights(self) -> np.ndarray:
        """Historical VBT notebook convention retained for diagnostics only."""
        return self.weights


def normalize_weights(
    weights: np.ndarray,
    *,
    log_transform: bool = True,
    zero_diagonal: bool = True,
) -> np.ndarray:
    matrix = np.asarray(weights, dtype=float).copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"SC must be square, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("SC contains non-finite values")
    if np.min(matrix) < 0:
        raise ValueError("SC contains negative tract weights")
    if log_transform:
        matrix = np.log1p(matrix)
    if zero_diagonal:
        np.fill_diagonal(matrix, 0.0)
    maximum = float(np.max(matrix))
    if maximum <= 0:
        raise ValueError("SC has no positive off-diagonal weights")
    matrix /= maximum
    return matrix


def load_connectome(path: str | Path, region_labels: tuple[str, ...]) -> Connectome:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    with zipfile.ZipFile(source) as archive:
        raw = np.loadtxt(archive.open("weights.txt"), dtype=float)
    if raw.shape != (len(region_labels), len(region_labels)):
        raise ValueError(f"SC shape {raw.shape} does not match {len(region_labels)} atlas regions")
    if not np.allclose(raw, raw.T, rtol=1e-8, atol=1e-10):
        raise ValueError("VEP SC is expected to be symmetric")
    return Connectome(
        region_labels=tuple(region_labels),
        raw_weights=raw,
        weights=normalize_weights(raw),
    )


def heaviside_coupling(
    state_x1: np.ndarray,
    weights: np.ndarray,
    scale: float,
    *,
    theta: float = -1.0,
) -> np.ndarray:
    """Reference Heaviside global coupling, applied to the x1 population."""

    x1 = np.asarray(state_x1, dtype=float)
    matrix = np.asarray(weights, dtype=float)
    if x1.ndim != 1 or matrix.shape != (x1.size, x1.size):
        raise ValueError(f"x1/SC mismatch: {x1.shape}, {matrix.shape}")
    activation = np.heaviside(x1 - theta, 0.5)
    return float(scale) * (matrix @ activation)


def difference_coupling(state: np.ndarray, weights: np.ndarray, scale: float) -> np.ndarray:
    """TVB Difference coupling with ``weights[i,j]`` interpreted as j -> i."""
    values = np.asarray(state, dtype=float)
    matrix = np.asarray(weights, dtype=float)
    if values.ndim != 1 or matrix.shape != (values.size, values.size):
        raise ValueError(f"state/SC mismatch: {values.shape}, {matrix.shape}")
    return float(scale) * (matrix @ values - matrix.sum(axis=1) * values)
