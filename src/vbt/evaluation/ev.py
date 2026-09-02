"""Paper-compatible epileptogenicity value from inferred source trajectories."""

import numpy as np


def epileptogenicity_value(source: np.ndarray, threshold: float = 0.0, no_seizure_time: int = 200) -> np.ndarray:
    """Return EV for ``source[..., time, region]``, normalized per sample."""
    values = np.asarray(source, dtype=float)
    if values.ndim < 2:
        raise ValueError("source must end in [time, region]")
    crossed = values > threshold
    onset = np.argmax(crossed, axis=-2)
    onset = np.where(np.any(crossed, axis=-2), onset, no_seizure_time)
    first = np.min(onset, axis=-1, keepdims=True)
    ev = -np.log(((onset - first) + 1.0) / 20.0)
    low = ev.min(axis=-1, keepdims=True)
    span = ev.max(axis=-1, keepdims=True) - low
    return np.divide(ev - low, span, out=np.zeros_like(ev), where=span > 0)
