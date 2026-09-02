"""Small, method-independent contracts for the DTB pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Readiness(str, Enum):
    COHORT_SYNTHETIC = "cohort_synthetic"
    NORMATIVE_CONNECTOME_PATIENT_SEEG = "normative_connectome_patient_seeg"
    PATIENT_SPECIFIC_PARCEL = "patient_specific_parcel"
    PATIENT_SPECIFIC_SURFACE = "patient_specific_surface"


@dataclass(frozen=True)
class ObservationBundle:
    subject_id: str
    recording_id: str
    seeg: np.ndarray
    time_s: np.ndarray
    channel_names: tuple[str, ...]
    sampling_frequency_hz: float
    features: np.ndarray | None = None
    feature_time_s: np.ndarray | None = None
    provenance: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if self.seeg.ndim != 2 or self.seeg.shape[0] != self.time_s.size:
            raise ValueError("ObservationBundle shape contract failed")
        if self.seeg.shape[1] != len(self.channel_names):
            raise ValueError("SEEG channels do not match channel_names")
        if self.seeg.shape[0] == 0 or self.sampling_frequency_hz <= 0:
            raise ValueError("empty observation or invalid sampling frequency")
        if not np.isfinite(self.seeg).all():
            raise ValueError("ObservationBundle contains non-finite values")


@dataclass(frozen=True)
class TwinParameters:
    region_names: tuple[str, ...]
    sc_native: np.ndarray
    gain: np.ndarray
    forward_profile: str
    inverse_model_id: str
    readiness: Readiness
    tract_lengths_mm: np.ndarray | None = None
    electrode_xyz_mm: np.ndarray | None = None
    fixed_model_params: dict[str, object] = field(default_factory=dict)
    prior_spec: dict[str, object] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    def validate(self, n_channels: int) -> None:
        regions = len(self.region_names)
        if self.sc_native.shape != (regions, regions) or self.gain.shape != (n_channels, regions):
            raise ValueError("TwinParameters shape contract failed")


@dataclass(frozen=True)
class PosteriorResult:
    method: str
    inverse_model_id: str
    result_kind: str
    region_names: tuple[str, ...]
    x0: np.ndarray
    source_samples: np.ndarray | None
    engineering_status: str
    scientific_status: str
    diagnostics: dict[str, object]
    provenance: dict[str, str]


@dataclass(frozen=True)
class EZNResult:
    region_names: tuple[str, ...]
    ev: np.ndarray | None
    posterior_mean_x0: np.ndarray
    primary_score: str
    status: str


@dataclass(frozen=True)
class Condition:
    """An explicit intervention/initial-state condition for one episode."""

    initial_state: np.ndarray
    stimulation: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self, n_regions: int) -> None:
        state = np.asarray(self.initial_state, dtype=float)
        if state.shape != (2, n_regions) or not np.isfinite(state).all():
            raise ValueError(f"initial_state must be finite with shape (2, {n_regions})")
        if self.stimulation is not None:
            stimulation = np.asarray(self.stimulation, dtype=float)
            if stimulation.ndim != 2 or stimulation.shape[1] != n_regions:
                raise ValueError(f"stimulation must have shape (time, {n_regions})")
            if not np.isfinite(stimulation).all():
                raise ValueError("stimulation contains non-finite values")


@dataclass(frozen=True)
class Episode:
    observation: ObservationBundle
    condition: Condition

    def validate(self, n_regions: int) -> None:
        self.observation.validate()
        self.condition.validate(n_regions)


@dataclass(frozen=True)
class ContextSet:
    subject_id: str
    episodes: tuple[Episode, ...]

    def validate(self, n_regions: int) -> None:
        if not self.episodes:
            raise ValueError("ContextSet must contain at least one episode")
        for episode in self.episodes:
            episode.validate(n_regions)
            if episode.observation.subject_id != self.subject_id:
                raise ValueError("all context episodes must belong to the same subject")


@dataclass(frozen=True)
class QueryEpisode:
    subject_id: str
    recording_id: str
    condition: Condition
    target: ObservationBundle | None = None

    def validate(self, n_regions: int) -> None:
        self.condition.validate(n_regions)
        if self.target is not None:
            self.target.validate()
            if self.target.subject_id != self.subject_id:
                raise ValueError("query target subject does not match query subject")


@dataclass(frozen=True)
class PersonalizationResult:
    subject_id: str
    x0: np.ndarray
    coupling: float
    method: str
    context_recording_ids: tuple[str, ...]
    objective: float
    diagnostics: dict[str, object] = field(default_factory=dict)

    def validate(self, n_regions: int) -> None:
        values = np.asarray(self.x0, dtype=float)
        if values.shape != (n_regions,) or not np.isfinite(values).all():
            raise ValueError(f"x0 must be finite with shape ({n_regions},)")
        if not np.isfinite(self.coupling) or not np.isfinite(self.objective):
            raise ValueError("coupling and objective must be finite")
