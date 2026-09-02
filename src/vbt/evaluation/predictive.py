"""Leakage-safe query prediction for a fitted reduced twin."""

from __future__ import annotations

import numpy as np

from vbt.contracts import PersonalizationResult, QueryEpisode
from vbt.models.reduced_epileptor import ReducedEpileptor
from vbt.observation.seeg import project_to_seeg


def predict_query(
    fit: PersonalizationResult,
    query: QueryEpisode,
    weights: np.ndarray,
    gain: np.ndarray,
    n_steps: int,
    *,
    dt: float = 0.05,
) -> np.ndarray:
    """Rerun the same fitted simulator under the query's explicit condition."""

    n_regions = np.asarray(weights).shape[0]
    fit.validate(n_regions)
    query.validate(n_regions)
    if query.subject_id != fit.subject_id:
        raise ValueError("query subject does not match personalized twin")
    stimulation = query.condition.stimulation
    if stimulation is not None and stimulation.shape[0] != n_steps:
        raise ValueError("query stimulation length does not match n_steps")
    source = ReducedEpileptor(fit.x0, fit.coupling, weights, dt=dt).simulate(
        query.condition.initial_state, n_steps, stimulation
    )
    return project_to_seeg(source, gain)


def trajectory_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=float)
    observed = np.asarray(target, dtype=float)
    if predicted.shape != observed.shape or predicted.ndim != 2:
        raise ValueError("prediction and target must be matrices with identical shapes")
    residual = observed - predicted
    denominator = float(np.sum((observed - observed.mean(axis=0)) ** 2))
    ev = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else float("nan")
    scale = float(np.sqrt(np.mean(observed**2)))
    nrmse = float(np.sqrt(np.mean(residual**2)) / scale) if scale > 0 else float("nan")
    return {"explained_variance": ev, "nrmse": nrmse}


def counterfactual_response(model: ReducedEpileptor, initial_state: np.ndarray,
                            stimulation: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Return candidate-specific stimulated minus candidate-specific control SEEG."""
    drive=np.asarray(stimulation,float); steps=drive.shape[0]
    stimulated=model.simulate(initial_state,steps,drive)
    control=model.simulate(initial_state,steps,np.zeros_like(drive))
    return project_to_seeg(stimulated,gain)-project_to_seeg(control,gain)
