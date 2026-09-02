"""Verify the NumPy port against the pinned upstream Python equation body."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbt.models.repo_spatepi_stim import RepoSpatEpiStim7D
from vbt.simulation.integrators import heun_step


DEFAULT_REFERENCE = ROOT.parent / "VBT_INS_Stimulation"


def load_oracle(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_numba_dfun"
    )
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "numpy": np,
        "heaviside_impl": lambda x, at_zero: 0.0 if x < 0 else (1.0 if x > 0 else at_zero),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_numba_dfun"]


def oracle_derivative(oracle, state, coupling, stimulus, model):
    output = np.empty_like(state)
    for node in range(state.shape[1]):
        oracle(
            state[:, node], np.array([coupling[node]]),
            np.array([model.x0[node]]), np.array([model.iext]), np.array([model.iext2]),
            np.array([0.0]), np.array([0.0]), np.array([0.0]),
            np.array([model.tt]), np.array([model.y0]), np.array([model.tau0]),
            np.array([model.tau2]), np.array([model.tau3]), np.array([0.01]),
            np.array([stimulus[node]]), np.array([model.threshold[node]]), output[:, node],
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "equivalence" / "repo_7d.json")
    args = parser.parse_args()
    source = args.reference / "vep_stim/core/utils_py/utils_model4.py"
    if not source.exists():
        raise FileNotFoundError(source)
    oracle = load_oracle(source)

    rng = np.random.default_rng(20250905)
    nodes = 5
    model = RepoSpatEpiStim7D(
        x0=np.linspace(-3.5, -2.2, nodes),
        threshold=np.linspace(100.0, 1.8, nodes),
        tau0=1000.0,
        tau3=600.0,
        tt=0.17,
    )
    state = np.array([-1.62, -16.69, 4.1, -1.11, 0.0, -0.44, 0.0])[:, None]
    state = np.repeat(state, nodes, axis=1) + rng.normal(0.0, 1e-3, (7, nodes))
    weights = rng.uniform(0.0, 0.2, (nodes, nodes))
    np.fill_diagonal(weights, 0.0)
    # Repository notebooks multiply mA by a sensor-to-source spatial map before
    # the model's additional 400x factor.  Use a realistic small map here so
    # the equivalence trajectory tests equations rather than intentional blow-up.
    spatial = np.linspace(1e-4, 1e-3, nodes)
    dt = 0.1

    def inputs(values, step):
        coupling = 0.1 * weights.dot(np.heaviside(values[0] + 1.0, 0.5))
        phase = step % 200
        pulse = 3.0 if 20 <= phase < 30 else (-3.0 if 30 <= phase < 40 else 0.0)
        return coupling, pulse * spatial

    coupling, stimulus = inputs(state, 0)
    derivative_error = float(np.max(np.abs(
        model.derivative(state, coupling, stimulus)
        - oracle_derivative(oracle, state, coupling, stimulus, model)
    )))

    def port_step(values, step):
        return heun_step(values, dt, lambda y: model.derivative(y, *inputs(y, step)))

    def oracle_step(values, step):
        return heun_step(values, dt, lambda y: oracle_derivative(oracle, y, *inputs(y, step), model))

    port_state = state.copy()
    oracle_state = state.copy()
    port_state = port_step(port_state, 0)
    oracle_state = oracle_step(oracle_state, 0)
    single_step_error = float(np.max(np.abs(port_state - oracle_state)))
    for step in range(1, args.steps):
        port_state = port_step(port_state, step)
        oracle_state = oracle_step(oracle_state, step)
    trajectory_finite = bool(np.isfinite(port_state).all() and np.isfinite(oracle_state).all())
    trajectory_error = float(np.max(np.abs(port_state - oracle_state))) if trajectory_finite else None

    report = {
        "reference": str(source),
        "reference_commit_required": "5367ef4afb976271ecc6297b70a95dd3d944af61",
        "states": ["u1", "u2", "s", "q1", "q2", "g", "m"],
        "steps": args.steps,
        "dt": dt,
        "derivative_max_abs_error": derivative_error,
        "heun_single_step_max_abs_error": single_step_error,
        "trajectory_finite": trajectory_finite,
        "trajectory_max_abs_error": trajectory_error,
        "equation_gate": "PASS" if derivative_error <= 1e-12 else "FAIL",
        "single_step_gate": "PASS" if single_step_error <= 1e-12 else "FAIL",
        "trajectory_gate": "PASS" if trajectory_finite and trajectory_error <= 1e-10 else "FAIL",
        "scope": "equation/integrator trajectory with explicit region-level coupling; not TVB surface/delay equivalence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all(report[key] == "PASS" for key in ("equation_gate", "single_step_gate", "trajectory_gate")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
