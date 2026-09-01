"""VEP cohort loader with explicit, checked file contracts."""

from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from vbt.network.connectome import Connectome, load_connectome
from vbt.observation.seeg import bipolarize_gain


def _parse_vector(value: str) -> np.ndarray:
    """Parse the bracketed, whitespace-separated arrays used by VEP TSVs."""

    text = str(value).strip().replace("\n", " ")
    text = text.replace("[", " ").replace("]", " ").replace(",", " ")
    result = np.fromstring(text, sep=" ", dtype=float)
    if result.size == 0:
        raise ValueError(f"Could not parse numeric vector: {value!r}")
    return result


def _parse_labels(value: str) -> tuple[str, ...]:
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        parsed = []
    if isinstance(parsed, (list, tuple)):
        return tuple(str(item) for item in parsed)
    return tuple(item.strip().strip("'\"") for item in str(value).strip("[]").split(",") if item.strip())


def _read_single_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"), None)
    if row is None:
        raise ValueError(f"Empty TSV: {path}")
    return row


def _read_header(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if "=" in line and not line.startswith(";"):
                key, value = line.rstrip("\n").split("=", 1)
                values[key] = value.strip()
    return values


def _read_channels(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return tuple(row["name"] for row in csv.DictReader(handle, delimiter="\t"))


@dataclass(frozen=True)
class Recording:
    """Metadata for one BrainVision recording; binary samples are lazy."""

    path: Path
    subject_id: str
    session: str
    task: str
    acquisition: str | None
    run: int
    sampling_frequency: float
    n_channels: int
    channel_names: tuple[str, ...]


@dataclass(frozen=True)
class ModelParameters:
    """Region-wise Epileptor parameters and synthetic EZ/PZ truth."""

    x0: np.ndarray
    i_ext: np.ndarray
    i_ext2: np.ndarray
    slope: np.ndarray
    r: float
    k_s: np.ndarray
    k_f: np.ndarray
    k_vf: np.ndarray
    ez_regions: tuple[str, ...]
    pz_regions: tuple[str, ...]
    source_path: Path

    @property
    def tau0(self) -> float:
        """The six-state reference uses tau0 = 1/r."""

        if self.r <= 0:
            raise ValueError(f"r must be positive, got {self.r}")
        return 1.0 / self.r


@dataclass(frozen=True)
class SimulatorParameters:
    coupling_factor: float
    noise_coeffs: np.ndarray
    initial_state: np.ndarray
    dt: float
    period: float
    source_path: Path


def _resolve_data_root(root: str | Path) -> Path:
    candidate = Path(root).expanduser()
    nested = candidate / "data" / "VirtualEpilepticCohort"
    if (nested / "participants.tsv").exists():
        return nested
    if (candidate / "participants.tsv").exists():
        return candidate
    raise FileNotFoundError(
        f"VEP data root not found. Expected {nested} or {candidate / 'participants.tsv'}"
    )


def _find_parameter_file(base: Path, subject_id: str, session: str, hypothesis: str, kind: str) -> Path:
    preferred = base / "derivatives" / "tvb" / subject_id / session / hypothesis / "parameters"
    matches = sorted(preferred.glob(f"{subject_id}_{kind}_parameters_run-01.tsv"))
    if matches:
        return matches[0]
    fallback = sorted((base / "derivatives" / "tvb" / subject_id).glob(f"**/{subject_id}_{kind}_parameters_run-01.tsv"))
    if not fallback:
        raise FileNotFoundError(f"No {kind} parameter TSV for {subject_id}")
    return fallback[0]


def _recordings(base: Path, subject_id: str) -> tuple[Recording, ...]:
    result: list[Recording] = []
    subject_dir = base / subject_id
    for path in sorted(subject_dir.glob("ses-*/ieeg/*.vhdr")):
        header = _read_header(path)
        match = re.search(r"task-([^_]+)", path.name)
        if match is None:
            raise ValueError(f"Recording has no task entity: {path}")
        session_match = re.search(r"(ses-[^_]+)", path.name)
        run_match = re.search(r"_run-(\d+)_", path.name)
        if session_match is None or run_match is None:
            raise ValueError(f"Recording has no session/run entity: {path}")
        acquisition_match = re.search(r"acq-([^_]+)", path.name)
        channels_path = subject_dir / session_match.group(1) / "ieeg" / (
            f"{subject_id}_{session_match.group(1)}_task-{match.group(1)}_run-{int(run_match.group(1)):02d}_channels.tsv"
        )
        if not channels_path.exists():
            channels_path = next(path.parent.glob(f"{subject_id}_{session_match.group(1)}_task-{match.group(1)}*_channels.tsv"), None)
        if channels_path is None or not channels_path.exists():
            raise FileNotFoundError(f"Missing channels.tsv for {path}")
        channel_names = _read_channels(channels_path)
        result.append(
            Recording(
                path=path,
                subject_id=subject_id,
                session=session_match.group(1),
                task=match.group(1),
                acquisition=acquisition_match.group(1) if acquisition_match else None,
                run=int(run_match.group(1)),
                sampling_frequency=1_000_000.0 / float(header["SamplingInterval"]),
                n_channels=int(header["NumberOfChannels"]),
                channel_names=channel_names,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class VEPSubject:
    """All files needed by the Phase 1 forward baseline for one subject."""

    root: Path
    subject_id: str
    region_names: tuple[str, ...]
    connectome: Connectome
    gain: np.ndarray
    electrodes: tuple[dict[str, str], ...]
    model_parameters: ModelParameters
    simulator_parameters: SimulatorParameters
    recordings: tuple[Recording, ...]

    @staticmethod
    def available_subjects(root: str | Path) -> tuple[str, ...]:
        base = _resolve_data_root(root)
        subjects = [p.name for p in base.iterdir() if p.is_dir() and re.fullmatch(r"sub-\d+", p.name)]
        return tuple(sorted(subjects, key=lambda value: int(value.split("-")[1])))

    @classmethod
    def load(
        cls,
        root: str | Path,
        subject_id: str,
        *,
        session: str = "ses-01",
        hypothesis: str = "VEPhypothesis",
    ) -> "VEPSubject":
        base = _resolve_data_root(root)
        subject_id = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
        subject_dir = base / subject_id
        if not subject_dir.is_dir():
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

        atlas_path = base / "vep_atlas.tsv"
        region_names: list[str] = []
        with atlas_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["label"] != "0":
                    region_names.append(row["region_name"])

        connectome_path = base / "derivatives" / "tvb" / subject_id / "struct" / f"{subject_id}_connectome.zip"
        connectome = load_connectome(connectome_path, tuple(region_names))
        gain_path = base / "derivatives" / "tvb" / subject_id / "struct" / f"{subject_id}_gain.txt"
        gain = np.loadtxt(gain_path, dtype=float)
        if gain.ndim != 2 or gain.shape[1] != len(region_names):
            raise ValueError(f"Unexpected gain shape {gain.shape}; expected [sensors, {len(region_names)}]")
        if not np.isfinite(gain).all():
            raise ValueError(f"Non-finite gain values: {gain_path}")

        electrodes_path = subject_dir / f"{subject_id}_electrodes.tsv"
        with electrodes_path.open("r", encoding="utf-8-sig", newline="") as handle:
            electrodes = tuple(dict(row) for row in csv.DictReader(handle, delimiter="\t"))
        if len(electrodes) != gain.shape[0]:
            raise ValueError(f"Electrode rows {len(electrodes)} do not match gain rows {gain.shape[0]}")

        model_path = _find_parameter_file(base, subject_id, session, hypothesis, "epileptor")
        model_row = _read_single_row(model_path)
        model = ModelParameters(
            x0=_parse_vector(model_row["x0"]),
            i_ext=_parse_vector(model_row["Iext"]),
            i_ext2=_parse_vector(model_row["Iext2"]),
            slope=_parse_vector(model_row["slope"]),
            r=float(_parse_vector(model_row["r"])[0]),
            k_s=_parse_vector(model_row["Ks"]),
            k_f=_parse_vector(model_row["Kf"]),
            k_vf=_parse_vector(model_row["Kvf"]),
            ez_regions=_parse_labels(model_row["EZ"]),
            pz_regions=_parse_labels(model_row["PZ"]),
            source_path=model_path,
        )
        n_regions = len(region_names)
        for name in ("x0", "i_ext", "i_ext2", "slope", "k_s", "k_f", "k_vf"):
            if getattr(model, name).shape != (n_regions,):
                raise ValueError(f"{name} has shape {getattr(model, name).shape}, expected {(n_regions,)}")

        simulator_path = _find_parameter_file(base, subject_id, session, hypothesis, "simulator")
        simulator_row = _read_single_row(simulator_path)
        simulator = SimulatorParameters(
            coupling_factor=float(_parse_vector(simulator_row["coupling_factor"])[0]),
            noise_coeffs=_parse_vector(simulator_row["noise_coeffs"]),
            initial_state=_parse_vector(simulator_row["init_cond"]),
            dt=float(_parse_vector(simulator_row["dt"])[0]),
            period=float(_parse_vector(simulator_row["period"])[0]),
            source_path=simulator_path,
        )
        if simulator.initial_state.shape != (6,):
            raise ValueError(f"Expected six initial states, got {simulator.initial_state.shape}")
        if simulator.dt <= 0 or simulator.period <= 0:
            raise ValueError(f"dt and period must be positive: {simulator.dt}, {simulator.period}")

        recordings = _recordings(base, subject_id)
        return cls(
            root=base,
            subject_id=subject_id,
            region_names=tuple(region_names),
            connectome=connectome,
            gain=gain,
            electrodes=electrodes,
            model_parameters=model,
            simulator_parameters=simulator,
            recordings=recordings,
        )

    @property
    def ez_truth(self) -> tuple[str, ...]:
        return self.model_parameters.ez_regions

    @property
    def pz_truth(self) -> tuple[str, ...]:
        return self.model_parameters.pz_regions

    def bipolar_gain(self, recording: Recording | None = None) -> np.ndarray:
        """Return signed gain rows aligned to one recording's bipolar channels."""

        if recording is None:
            if not self.recordings:
                raise ValueError(f"No recordings for {self.subject_id}")
            recording = self.recordings[0]
        contact_names = tuple(row["name"] for row in self.electrodes)
        return bipolarize_gain(self.gain, contact_names, recording.channel_names)

    def source_files(self) -> tuple[Path, ...]:
        return tuple(sorted((self.root / "derivatives" / "tvb" / self.subject_id).glob("ses-*/*/*source_timeseries_run-*.npz")))
