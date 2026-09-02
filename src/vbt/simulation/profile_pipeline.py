"""Small shared forward pipeline for ``default`` and ``vep_25``."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from vbt.data.parameters import (
    load_epileptor_parameters,
    load_simulator_parameters,
    load_stimulation_parameters,
)
from vbt.models.epileptor_stim_cohort import StimEpileptor
from vbt.models.repo_spatepi_stim import RepoSpatEpiStim7D, repo_source
from vbt.network.connectome import difference_coupling, heaviside_coupling
from vbt.observation.seeg import project_to_seeg
from vbt.profiles import Profile
from vbt.simulation.integrators import ColouredAdditiveNoise, heun_step, heun_stochastic_step
from vbt.simulation.cohort import simulate as simulate_spontaneous
from vbt.stimulation.waveform import biphasic_waveform


def _parameter_files(subject):
    base = subject.root / "derivatives" / "tvb" / subject.subject_id / "ses-02" / "VEPhypothesis" / "parameters"
    return (
        next(base.glob("*epileptor*run-01.tsv")),
        next(base.glob("*simulator*run-01.tsv")),
        next(base.glob("*stimulation*run-01.tsv")),
    )


def run_forward(subject, profile: Profile, duration: float, seed: int = 0) -> dict:
    """Run spontaneous and stimulated branches through the same SEEG operator."""

    epi_path, sim_path, stim_path = _parameter_files(subject)
    epi = load_epileptor_parameters(epi_path)
    sim = load_simulator_parameters(sim_path)
    stim = load_stimulation_parameters(stim_path)
    duration = min(float(duration), float(stim.stimulation_length))
    if profile.name == "vep_25":
        duration, dt, monitor_period = 100.0, 0.2, 10.0
        time, waveform = biphasic_waveform(duration, dt, 20.0, 20.0, 3.0, 2.0)
    else:
        dt, monitor_period = sim.dt, sim.period
        time, waveform = biphasic_waveform(
            duration, dt, stim.onset, stim.period_samples, stim.amplitude, stim.pulse_width
        )
    initial = sim.init_cond if sim.init_cond.shape[0] == 7 else np.vstack((sim.init_cond, np.zeros((1, 162))))
    weights = subject.connectome.cohort_weights

    if profile.name == "default":
        model = StimEpileptor(epi.x0, epi.threshold, epi.iext, epi.iext2, np.repeat(epi.r, 162), epi.r2, epi.ks, epi.kf, epi.kvf)
        def rhs(state, drive):
            coupling = difference_coupling(state[0], weights, sim.coupling_factor)
            return model.derivative(state, coupling, drive)
        source_fn = lambda state: state[3] - state[0]
    else:
        x0 = np.full(162, -3.5)
        threshold = np.full(162, 100.0)
        ez_indices = [subject.region_names.index(name) for name in epi.ez]
        x0[ez_indices] = -2.2
        threshold[ez_indices] = 1.8
        model = RepoSpatEpiStim7D(
            x0, threshold, iext=3.1, iext2=0.45,
            tt=profile.tt, tau0=profile.tau0, tau2=profile.tau2, tau3=profile.tau3,
        )
        initial = np.repeat(
            np.array([-1.6242601, -16.69344913, 4.1, -1.11181819, -9.56105974e-20, -0.438727802, 0.0])[:, None],
            162,
            axis=1,
        )
        weights = subject.connectome.legacy_weights
        def rhs(state, drive):
            coupling = heaviside_coupling(state[0], weights, 0.1, theta=-1.0)
            return model.derivative(state, coupling, drive)
        source_fn = repo_source

    every = max(1, round(monitor_period / dt))
    def branch(signal):
        state = initial.copy()
        noise = ColouredAdditiveNoise(sim.noise_coeffs if profile.stochastic else np.zeros(7), 1.0, seed)
        blocks, sources = [], []
        for step, scalar in enumerate(signal):
            drive = scalar * stim.weights * profile.parcel_stimulus_scale
            fn = lambda value: rhs(value, drive)
            state = (heun_stochastic_step(state, dt, fn, noise.sample(dt, 162))
                     if profile.stochastic else heun_step(state, dt, fn))
            if not np.isfinite(state).all():
                raise FloatingPointError(f"{profile.name}: non-finite state at step {step + 1}")
            blocks.append(source_fn(state))
            if (step + 1) % every == 0:
                sources.append(np.mean(blocks, axis=0)); blocks = []
        return np.asarray(sources)

    spontaneous_result = simulate_spontaneous(subject, duration=duration, noise=True, seed=seed)
    spontaneous = spontaneous_result.source_activity
    stimulated = branch(waveform)
    recording = next((item for item in subject.recordings if item.session == "ses-02"), subject.recordings[0])
    gain = subject.bipolar_gain(recording)
    spontaneous_seeg = project_to_seeg(spontaneous, gain)
    stimulated_seeg = project_to_seeg(stimulated, gain)
    score = np.ptp(stimulated, axis=0)
    return {
        "profile": asdict(profile), "time": time[every - 1::every], "spontaneous_time": spontaneous_result.time, "waveform": waveform,
        "spontaneous_source": spontaneous, "stimulated_source": stimulated,
        "spontaneous_seeg": spontaneous_seeg, "stimulated_seeg": stimulated_seeg,
        "forward_ezn_score": score, "region_names": np.asarray(subject.region_names),
        "recording": str(recording.path), "parameter_files": tuple(map(str, (epi_path, sim_path, stim_path))),
    }
