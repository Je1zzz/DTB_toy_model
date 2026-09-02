"""The two supported VBT execution profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    model: str
    coupling: str
    stochastic: bool
    tt: float = 1.0
    tau0: float = 2857.0
    tau2: float = 10.0
    tau3: float = 2857.0
    parcel_stimulus_scale: float = 1.0
    reference: str = "VEP_Cohort_v2.0 generator"


PROFILES = {
    "default": Profile("default", "cohort_stim_7d", "difference", True),
    "vep_25": Profile(
        "vep_25",
        "repo_spatepi_stim_7d",
        "heaviside",
        False,
        tt=0.17,
        tau0=1000.0,
        tau2=10.0,
        tau3=600.0,
        parcel_stimulus_scale=0.001,
        reference="VBT_INS_Stimulation tag 0.1.0 (e8b6f597), utils_model4.py/notebooks",
    ),
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"profile must be one of {tuple(PROFILES)}, got {name!r}") from exc
