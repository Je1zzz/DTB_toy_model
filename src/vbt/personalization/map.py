"""MAP-style fitting of patient-shared reduced Epileptor parameters."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from vbt.contracts import ContextSet, PersonalizationResult
from vbt.models.reduced_epileptor import ReducedEpileptor
from vbt.observation.seeg import project_to_seeg


def fit_context_map(
    context: ContextSet,
    weights: np.ndarray,
    gain: np.ndarray,
    *,
    initial_x0: np.ndarray | None = None,
    initial_coupling: float = 0.5,
    dt: float = 0.05,
    max_nfev: int = 300,
) -> PersonalizationResult:
    """Fit one shared ``x0`` and coupling value across all context episodes.

    Episode initial state and stimulation remain event-specific and are never
    optimized as patient attributes.
    """

    matrix = np.asarray(weights, dtype=float)
    n_regions = matrix.shape[0]
    if matrix.shape != (n_regions, n_regions):
        raise ValueError("weights must be square")
    context.validate(n_regions)
    start_x0 = np.full(n_regions, -2.2) if initial_x0 is None else np.asarray(initial_x0, dtype=float)
    if start_x0.shape != (n_regions,):
        raise ValueError(f"initial_x0 must have shape ({n_regions},)")

    def residual(theta: np.ndarray) -> np.ndarray:
        model = ReducedEpileptor(theta[:-1], float(theta[-1]), matrix, dt=dt)
        chunks: list[np.ndarray] = []
        for episode in context.episodes:
            target = episode.observation.seeg
            source = model.simulate(
                episode.condition.initial_state,
                target.shape[0],
                episode.condition.stimulation,
            )
            chunks.append((project_to_seeg(source, gain) - target).ravel())
        return np.concatenate(chunks)

    fitted = least_squares(
        residual,
        np.r_[start_x0, initial_coupling],
        bounds=(np.r_[np.full(n_regions, -3.5), 0.0], np.r_[np.full(n_regions, -1.0), 5.0]),
        max_nfev=max_nfev,
    )
    return PersonalizationResult(
        subject_id=context.subject_id,
        x0=fitted.x[:-1],
        coupling=float(fitted.x[-1]),
        method="reduced_epileptor_map",
        context_recording_ids=tuple(episode.observation.recording_id for episode in context.episodes),
        objective=float(np.mean(fitted.fun**2)),
        diagnostics={"success": bool(fitted.success), "nfev": int(fitted.nfev), "message": fitted.message},
    )
