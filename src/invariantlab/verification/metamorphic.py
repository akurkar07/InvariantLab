"""Layer 5 metamorphic relations: rerun a candidate on a transformed input."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from invariantlab.schema import GateResult, load_task_contract
from invariantlab.verification.execution import TaskRun, run_task

if TYPE_CHECKING:
    from pathlib import Path

# Measured on the trusted oscillator package (x0=0.7, v0=-0.35, omega=1.7,
# dt=1e-3, n_steps=15000): 2.0e-15.
TIME_REVERSAL_THRESHOLD = 1e-10
# Measured on the trusted Kepler package (circular orbit, dt=0.004,
# n_steps=1080): 1.6e-14.
ROTATION_COVARIANCE_THRESHOLD = 1e-10
# Measured on the trusted heat package (nx=161, nt=1800, length=1.3): 0.0
# (power-of-two scaling keeps the FTCS ratio and grid bit-identical).
DIFFUSIVE_SCALING_THRESHOLD = 1e-10
# Measured on the trusted wave package (nx=401, nt=400, length=1.3): 0.0
# (power-of-two scaling keeps the Courant number and grid bit-identical).
WAVE_SCALING_THRESHOLD = 1e-10

ROTATION_ANGLE = 0.7


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = float(np.linalg.norm(expected))
    error = float(np.linalg.norm(actual - expected))
    return error / scale if scale > 0.0 else error


def _layer0_failure(label: str, run: TaskRun) -> str | None:
    if run.passed:
        return None
    failed = next(gate for gate in run.gates if not gate.passed)
    return f"{label} run failed Layer 0 {failed.name}: {failed.detail}"


def _rotate(state: np.ndarray, theta: float) -> np.ndarray:
    cos, sin = math.cos(theta), math.sin(theta)
    rotated = np.empty_like(state)
    rotated[..., 0] = cos * state[..., 0] - sin * state[..., 1]
    rotated[..., 1] = sin * state[..., 0] + cos * state[..., 1]
    rotated[..., 2] = cos * state[..., 2] - sin * state[..., 3]
    rotated[..., 3] = sin * state[..., 2] + cos * state[..., 3]
    return rotated


def _gate(
    name: str, residual_name: str, residual: float, threshold: float
) -> GateResult:
    passed = residual <= threshold
    return GateResult(
        name=name,
        passed=passed,
        deviation=residual,
        threshold=threshold,
        detail=f"{residual_name}={residual:.3e} {'<=' if passed else '>'} {threshold:.1e}",
    )


def _failed_gate(name: str, threshold: float, detail: str) -> GateResult:
    return GateResult(name=name, passed=False, threshold=threshold, detail=detail)


def check_metamorphic(
    task_dir: Path,
    candidate_root: Path,
    parameters: dict[str, object],
    work_dir: Path,
) -> list[GateResult]:
    """Run the task's metamorphic relation and return one Layer 5 gate."""
    task_id = load_task_contract(task_dir).id
    if task_id == "oscillator_verlet":
        name, residual_name, threshold = (
            "time_reversal",
            "time_reversal_residual",
            TIME_REVERSAL_THRESHOLD,
        )
    elif task_id == "kepler_verlet":
        name, residual_name, threshold = (
            "rotation_covariance",
            "rotation_relative_l2",
            ROTATION_COVARIANCE_THRESHOLD,
        )
    elif task_id == "heat_ftcs":
        name, residual_name, threshold = (
            "diffusive_scaling",
            "scaled_state_relative_l2",
            DIFFUSIVE_SCALING_THRESHOLD,
        )
    elif task_id == "wave_leapfrog":
        name, residual_name, threshold = (
            "wave_scaling",
            "scaled_state_relative_l2",
            WAVE_SCALING_THRESHOLD,
        )
    else:
        raise ValueError(f"no metamorphic relation for task id {task_id!r}")

    base = run_task(task_dir, candidate_root, parameters, work_dir / "base")
    failure = _layer0_failure("base", base)
    if failure is not None:
        return [_failed_gate(name, threshold, failure)]
    assert base.arrays is not None
    base_state = base.arrays["state"]

    if task_id == "oscillator_verlet":
        x_final, v_final = (float(value) for value in base_state[-1])
        transformed_parameters = {**parameters, "x0": x_final, "v0": -v_final}
    elif task_id == "kepler_verlet":
        initial = np.array(
            [parameters["rx"], parameters["ry"], parameters["vx"], parameters["vy"]],
            dtype=np.float64,
        )
        rx, ry, vx, vy = (float(value) for value in _rotate(initial, ROTATION_ANGLE))
        transformed_parameters = {**parameters, "rx": rx, "ry": ry, "vx": vx, "vy": vy}
    elif task_id == "heat_ftcs":
        transformed_parameters = {
            **parameters,
            "length": 2.0 * float(parameters["length"]),  # type: ignore[arg-type]
            "alpha": 4.0 * float(parameters["alpha"]),  # type: ignore[arg-type]
        }
    else:
        transformed_parameters = {
            **parameters,
            "length": 2.0 * float(parameters["length"]),  # type: ignore[arg-type]
            "c": 2.0 * float(parameters["c"]),  # type: ignore[arg-type]
        }

    transformed = run_task(
        task_dir, candidate_root, transformed_parameters, work_dir / "transformed"
    )
    failure = _layer0_failure("transformed", transformed)
    if failure is not None:
        return [_failed_gate(name, threshold, failure)]
    assert transformed.arrays is not None
    transformed_state = transformed.arrays["state"]

    if task_id == "oscillator_verlet":
        x_back, v_back = transformed_state[-1]
        initial = np.array([parameters["x0"], parameters["v0"]], dtype=np.float64)
        residual = _relative_l2(np.array([x_back, -v_back]), initial)
    elif task_id == "kepler_verlet":
        if transformed_state.shape != base_state.shape:
            return [_failed_gate(name, threshold, "rotated run changed the state shape")]
        residual = _relative_l2(_rotate(transformed_state, -ROTATION_ANGLE), base_state)
    else:
        if transformed_state.shape != base_state.shape:
            return [_failed_gate(name, threshold, "scaled run changed the state shape")]
        residual = _relative_l2(transformed_state, base_state)
    return [_gate(name, residual_name, residual, threshold)]
