"""Legacy deterministic VBT SpatEpi path; not the VEP cohort generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vbt.data.vep import VEPSubject
from vbt.models.epileptor import Epileptor6D
from vbt.network.connectome import heaviside_coupling


@dataclass(frozen=True)
class SimulationResult:
    time: np.ndarray
    state: np.ndarray
    source_variable: str = "u1_minus_q1"

    @property
    def source_activity(self) -> np.ndarray:
        if self.source_variable != "u1_minus_q1":
            raise ValueError(f"Unsupported source variable: {self.source_variable}")
        # variables_of_interest=(u1,q1,s,m); the notebook subtracts monitor
        # columns 0 and 1, corresponding to raw state indices 0 and 3.
        return self.state[:, 0, :] - self.state[:, 3, :]


def simulate_spontaneous(
    subject: VEPSubject,
    *,
    duration: float = 100.0,
    dt: float | None = None,
    sample_period: float | None = None,
    coupling_factor: float | None = None,
) -> SimulationResult:
    """Run the legacy custom SpatEpi diagnostic without stimulation.

    ``duration``, ``dt`` and ``sample_period`` use the time units recorded in
    the VEP simulator parameter TSV. Noise is intentionally absent in this
    Phase 1 deterministic gate; the source TSV's noise coefficients remain
    available through ``subject.simulator_parameters``.
    """

    simulator = subject.simulator_parameters
    step = float(simulator.dt if dt is None else dt)
    output_period = float(simulator.period if sample_period is None else sample_period)
    if duration <= 0 or step <= 0 or output_period <= 0:
        raise ValueError("duration, dt, and sample_period must be positive")
    n_steps = max(1, int(round(float(duration) / step)))
    sample_every = max(1, int(round(output_period / step)))

    model = Epileptor6D(
        x0=subject.model_parameters.x0,
        i_ext=subject.model_parameters.i_ext,
        i_ext2=subject.model_parameters.i_ext2,
        r=subject.model_parameters.r,
    )
    n_regions = model.n_regions
    state = np.repeat(simulator.initial_state[:, None], n_regions, axis=1)
    scale = simulator.coupling_factor if coupling_factor is None else float(coupling_factor)
    weights = subject.connectome.weights

    def rhs(values: np.ndarray) -> np.ndarray:
        global_input = heaviside_coupling(values[0], weights, scale)
        return model.derivatives(values, global_input)

    times = [0.0]
    states = [state.copy()]
    for step_index in range(n_steps):
        try:
            first = rhs(state)
            predictor = state + step * first
            second = rhs(predictor)
        except FloatingPointError as exc:
            raise FloatingPointError(
                f"Non-finite derivative at integration step {step_index + 1}: {exc}"
            ) from exc
        state = state + 0.5 * step * (first + second)
        if not np.isfinite(state).all():
            raise FloatingPointError(f"Non-finite state at integration step {step_index + 1}")
        if (step_index + 1) % sample_every == 0 or step_index == n_steps - 1:
            times.append((step_index + 1) * step)
            states.append(state.copy())

    result = SimulationResult(time=np.asarray(times), state=np.asarray(states))
    if result.state.ndim != 3 or result.state.shape[1:] != (6, n_regions):
        raise RuntimeError(f"Unexpected simulation shape: {result.state.shape}")
    if np.allclose(result.source_activity, result.source_activity[0]):
        raise RuntimeError("Forward simulation is numerically constant")
    return result
