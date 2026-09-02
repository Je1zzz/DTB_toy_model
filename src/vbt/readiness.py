"""Hard scientific-claim gates from available subject assets."""

from __future__ import annotations

from vbt.contracts import Readiness


REQUIRED = {
    Readiness.COHORT_SYNTHETIC: {"synthetic_truth", "seeg", "sc", "gain"},
    Readiness.NORMATIVE_CONNECTOME_PATIENT_SEEG: {"patient_seeg", "electrodes", "normative_sc", "gain"},
    Readiness.PATIENT_SPECIFIC_PARCEL: {"patient_seeg", "electrodes", "patient_t1", "patient_dwi", "patient_sc", "gain"},
    Readiness.PATIENT_SPECIFIC_SURFACE: {"patient_seeg", "electrodes", "patient_t1", "patient_dwi", "patient_sc", "surface", "region_map", "local_connectivity", "gain", "stimulation_field"},
}


def audit_readiness(assets: set[str]) -> dict[str, object]:
    achieved = Readiness.COHORT_SYNTHETIC
    levels = {}
    for level, required in REQUIRED.items():
        missing = sorted(required - assets)
        levels[level.value] = {"pass": not missing, "missing": missing}
        if not missing:
            achieved = level
    return {"highest_readiness": achieved.value, "levels": levels}
