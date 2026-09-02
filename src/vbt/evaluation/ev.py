"""Paper-compatible onset and epileptogenicity value from inferred sources."""

import numpy as np


def source_onsets(
    source: np.ndarray, threshold: float = 0.0, no_seizure_time: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Return first-crossing indices and recruitment flags for ``[..., time, region]``."""
    values = np.asarray(source, dtype=float)
    if values.ndim < 2:
        raise ValueError("source must end in [time, region]")
    if values.shape[-2] == 0 or values.shape[-1] == 0:
        raise ValueError("source time and region dimensions must be non-empty")
    if not np.isfinite(values).all():
        raise ValueError("source contains non-finite values")
    crossed = values > threshold
    onset = np.argmax(crossed, axis=-2)
    recruited = np.any(crossed, axis=-2)
    onset = np.where(recruited, onset, no_seizure_time)
    return onset, recruited


def epileptogenicity_value(
    source: np.ndarray, threshold: float = 0.0, no_seizure_time: int = 200
) -> np.ndarray:
    """Return per-sample EV using Wang et al., Nat Comput Sci (2025)."""
    onset, _ = source_onsets(source, threshold, no_seizure_time)
    first = np.min(onset, axis=-1, keepdims=True)
    ev = -np.log(((onset - first) + 1.0) / 20.0)
    low = ev.min(axis=-1, keepdims=True)
    span = ev.max(axis=-1, keepdims=True) - low
    return np.divide(ev - low, span, out=np.zeros_like(ev), where=span > 0)
