"""Six-state SpatEpi equations from the reference VBT implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Epileptor6D:
    """Region-wise six-state Epileptor used for the spontaneous forward model.

    State order is ``(x1, x2, s, q1, q2, g)``. The equations mirror
    ``utils_simulate.py::SpatEpi``. ``r`` is the reciprocal slow time scale
    (the reference class stores the same quantity as ``1 / tau0``).
    """

    x0: np.ndarray
    i_ext: np.ndarray
    i_ext2: np.ndarray
    r: float
    y0: float = 1.0
    tau2: float = 10.0
    tt: float = 1.0

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=float)
        self.i_ext = np.asarray(self.i_ext, dtype=float)
        self.i_ext2 = np.asarray(self.i_ext2, dtype=float)
        if self.x0.ndim != 1:
            raise ValueError(f"x0 must be one-dimensional, got {self.x0.shape}")
        n = self.x0.size
        for name in ("i_ext", "i_ext2"):
            value = getattr(self, name)
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape {(n,)}, got {value.shape}")
        if self.r <= 0 or self.tau2 <= 0 or self.tt <= 0:
            raise ValueError("r, tau2, and tt must be positive")

    @property
    def n_regions(self) -> int:
        return self.x0.size

    def derivatives(
        self,
        state: np.ndarray,
        global_coupling: np.ndarray,
        *,
        local_11: np.ndarray | None = None,
        local_22: np.ndarray | None = None,
        local_12: np.ndarray | None = None,
    ) -> np.ndarray:
        values = np.asarray(state, dtype=float)
        if values.shape != (6, self.n_regions):
            raise ValueError(f"state must be (6, {self.n_regions}), got {values.shape}")
        coupling = np.asarray(global_coupling, dtype=float)
        if coupling.shape != (self.n_regions,):
            raise ValueError(f"global coupling must be {(self.n_regions,)}, got {coupling.shape}")
        zeros = np.zeros(self.n_regions, dtype=float)
        loc11 = zeros if local_11 is None else np.asarray(local_11, dtype=float)
        loc22 = zeros if local_22 is None else np.asarray(local_22, dtype=float)
        loc12 = zeros if local_12 is None else np.asarray(local_12, dtype=float)
        if any(value.shape != (self.n_regions,) for value in (loc11, loc22, loc12)):
            raise ValueError("local coupling arrays must match the region count")

        x1, x2, s, q1, q2, g = values
        with np.errstate(over="raise", invalid="raise"):
            f1 = np.where(x1 < 0.0, x1**3 - 3.0 * x1**2, (q1 - 0.6 * (s - 4.0) ** 2) * x1)
            ds_correction = np.where(s < 0.0, -0.1 * s**7, 0.0)
            dq2_drive = np.where(q1 < -0.25, 0.0, 6.0 * (q1 + 0.25))

            derivative = np.empty_like(values)
            derivative[0] = self.tt * (x2 - f1 - s + self.i_ext + loc11 + coupling)
            derivative[1] = self.tt * (self.y0 - 5.0 * x1**2 - x2)
            derivative[2] = self.tt * self.r * (4.0 * (x1 - self.x0) - s + ds_correction)
            derivative[3] = self.tt * (-q2 + q1 - q1**3 + self.i_ext2 + 2.0 * g - 0.3 * (s - 3.5) + loc22)
            derivative[4] = self.tt * ((-q2 + dq2_drive) / self.tau2)
            derivative[5] = self.tt * (-0.01 * g + 0.003 * x1 + 0.01 * loc12)
        return derivative
