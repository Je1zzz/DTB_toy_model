"""Small two-state Epileptor used by the context/query baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReducedEpileptor:
    """Deterministic reduced forward with patient-shared ``x0`` and ``K``."""

    x0: np.ndarray
    coupling: float
    weights: np.ndarray
    dt: float = 0.05
    tau: float = 20.0

    def __post_init__(self) -> None:
        x0 = np.asarray(self.x0, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if x0.ndim != 1 or weights.shape != (x0.size, x0.size):
            raise ValueError("x0 and weights have incompatible shapes")
        if not np.isfinite(x0).all() or not np.isfinite(weights).all():
            raise ValueError("model parameters contain non-finite values")
        if self.dt <= 0 or self.tau <= 0:
            raise ValueError("dt and tau must be positive")

    def simulate(self, initial_state: np.ndarray, n_steps: int, stimulation: np.ndarray | None = None) -> np.ndarray:
        state = np.asarray(initial_state, dtype=float).copy()
        if state.shape != (2, self.x0.size):
            raise ValueError(f"initial_state must have shape (2, {self.x0.size})")
        drive = np.zeros((n_steps, self.x0.size)) if stimulation is None else np.asarray(stimulation, dtype=float)
        if drive.shape != (n_steps, self.x0.size):
            raise ValueError(f"stimulation must have shape ({n_steps}, {self.x0.size})")
        output = np.empty((n_steps, self.x0.size), dtype=float)
        degree = self.weights.sum(axis=1)
        for step in range(n_steps):
            x, z = state
            network = self.weights @ x - degree * x
            dx = 1.0 - x**3 - 2.0 * x**2 - z + drive[step]
            dz = (4.0 * (x - self.x0) - z - self.coupling * network) / self.tau
            state += self.dt * np.stack((dx, dz))
            if not np.isfinite(state).all():
                raise FloatingPointError(f"non-finite reduced state at step {step + 1}")
            output[step] = state[0]
        return output
