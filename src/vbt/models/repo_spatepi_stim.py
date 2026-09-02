"""Equation-faithful NumPy port of ``VBT_INS_Stimulation`` SpatEpiStim.

This is the seven-state model in ``utils_model4.py`` at repository release
``0.1.0``.  It is intentionally separate from both the four-state equation
printed in Wang et al. (2025) and the VEP cohort generator model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RepoSpatEpiStim7D:
    """Original repository state order: ``u1,u2,s,q1,q2,g,m``."""

    x0: np.ndarray
    threshold: np.ndarray
    iext: float = 3.1
    iext2: float = 0.45
    y0: float = 1.0
    tau0: float = 2857.0
    tau2: float = 10.0
    tau3: float = 2857.0
    tt: float = 1.0

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=float)
        self.threshold = np.asarray(self.threshold, dtype=float)
        if self.x0.ndim != 1 or self.threshold.shape != self.x0.shape:
            raise ValueError("x0 and threshold must be equal-length vectors")
        if min(self.tau0, self.tau2, self.tau3, self.tt) <= 0:
            raise ValueError("time scales must be positive")

    @property
    def n_nodes(self) -> int:
        return self.x0.size

    def derivative(
        self,
        state: np.ndarray,
        global_coupling: np.ndarray,
        stimulus: np.ndarray,
        *,
        local_11: np.ndarray | None = None,
        local_22: np.ndarray | None = None,
        local_12: np.ndarray | None = None,
    ) -> np.ndarray:
        y = np.asarray(state, dtype=float)
        if y.shape != (7, self.n_nodes):
            raise ValueError(f"state must be (7, {self.n_nodes}), got {y.shape}")
        coupling = np.asarray(global_coupling, dtype=float)
        istim = np.asarray(stimulus, dtype=float)
        if coupling.shape != (self.n_nodes,) or istim.shape != (self.n_nodes,):
            raise ValueError("coupling and stimulus must match the node count")
        zeros = np.zeros(self.n_nodes)
        loc11 = zeros if local_11 is None else np.asarray(local_11, dtype=float)
        loc22 = zeros if local_22 is None else np.asarray(local_22, dtype=float)
        loc12 = zeros if local_12 is None else np.asarray(local_12, dtype=float)

        u1, u2, s, q1, q2, g, m = y
        f1 = np.where(u1 < 0.0, u1**3 - 3.0 * u1**2, (q1 - 0.6 * (s - 4.0) ** 2) * u1)
        s7 = np.where(s < 0.0, -0.1 * s**7, 0.0)
        q2_drive = np.where(q1 < -0.25, 0.0, 6.0 * (q1 + 0.25))
        threshold_switch = np.heaviside(m - self.threshold, 1.0)

        out = np.empty_like(y)
        out[0] = self.tt * (u2 - f1 - s + self.iext + 400.0 * istim + loc11 + coupling)
        out[1] = self.tt * (self.y0 - 5.0 * u1**2 - u2)
        out[2] = self.tt / self.tau0 * (4.0 * (u1 - self.x0 - threshold_switch) - s + s7)
        out[3] = self.tt * (-q2 + q1 - q1**3 + self.iext2 + 2.0 * g - 0.3 * (s - 3.5) + loc22)
        out[4] = self.tt / self.tau2 * (-q2 + q2_drive)
        out[5] = self.tt * (-0.01 * g + 0.003 * u1 + 0.01 * loc12)
        out[6] = self.tt / self.tau3 * (-m + 1000.0 * np.abs(istim))
        return out


def repo_source(state: np.ndarray) -> np.ndarray:
    """Notebook monitor convention: ``u1 - q1`` (raw states 0 and 3)."""

    values = np.asarray(state)
    return values[..., 0, :] - values[..., 3, :]
