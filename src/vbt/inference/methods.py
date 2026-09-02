"""The honest inversion-method menu for the shared reduced-Epileptor model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InversionMethod:
    name: str
    status: str
    backend: str
    output: str
    note: str


METHODS = {
    "map": InversionMethod("map", "IMPLEMENTED", "CmdStan optimize/L-BFGS", "point estimate", "fast engineering baseline"),
    "nuts": InversionMethod("nuts", "IMPLEMENTED", "CmdStan NUTS/HMC", "posterior samples", "convergence gates are mandatory"),
    "advi": InversionMethod("advi", "NOT_IMPLEMENTED", "Stan variational", "approximate posterior", "unstable in prior experiments"),
    "enkf": InversionMethod("enkf", "NOT_IMPLEMENTED", "ensemble Kalman", "ensemble", "requires state-space observation model"),
    "sbi": InversionMethod("sbi", "NOT_IMPLEMENTED", "ABC/SNPE", "amortized posterior", "requires simulation bank and calibration"),
}


def get_method(name: str) -> InversionMethod:
    if name not in METHODS:
        raise ValueError(f"unknown inversion method {name!r}; choose from {tuple(METHODS)}")
    method = METHODS[name]
    if method.status != "IMPLEMENTED":
        raise NotImplementedError(f"{name}: NOT_IMPLEMENTED - {method.note}")
    return method
