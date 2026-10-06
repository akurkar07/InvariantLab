"""Tests for the Layer 3 physical-invariant gates."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from invariantlab.schema import load_task_contract
from invariantlab.verification.analytical import kepler_circular_orbit, kepler_elliptic_orbit
from invariantlab.verification.execution import run_task
from invariantlab.verification.invariants import check_invariants

if TYPE_CHECKING:
    from collections.abc import Callable

TASKS_ROOT = Path(__file__).resolve().parents[2] / "tasks"
ArrayMap = dict[str, np.ndarray]


def _kepler_parameters(
    orbit: tuple[float, float, float, float, float, float], mu: float, dt: float, n_steps: int
) -> dict[str, object]:
    return {
        "rx": orbit[0],
        "ry": orbit[1],
        "vx": orbit[3],
        "vy": orbit[4],
        "mu": mu,
        "dt": dt,
        "n_steps": n_steps,
    }


OSCILLATOR = {"x0": 0.7, "v0": -0.35, "omega": 1.7, "dt": 1e-3, "n_steps": 15_000}
KEPLER_CIRCULAR = _kepler_parameters(kepler_circular_orbit(0.0, 2.5, 1.7, 0.37), 2.5, 0.004, 1_080)
KEPLER_ECCENTRIC = _kepler_parameters(
    kepler_elliptic_orbit(0.0, 1.9, 2.3, 0.41, 0.63), 1.9, 0.004, 2_178
)
HEAT = {"nx": 161, "nt": 1_800, "alpha": 0.17, "length": 1.3, "t_final": 0.237}
WAVE_SLOW = {"nx": 401, "nt": 400, "c": 0.65, "length": 1.3, "t_final": 0.39}
WAVE_FAST = {"nx": 401, "nt": 500, "c": 1.15, "length": 1.7, "t_final": 0.319}
WAVE_STARTUP = {"nx": 5, "nt": 1, "c": 0.75, "length": 1.7, "t_final": 0.51}
# One full period (2 * length / c), where the trusted amplitude ratio is ~1.
WAVE_FULL_PERIOD = {"nx": 401, "nt": 1_000, "c": 0.65, "length": 1.3, "t_final": 4.0}

SCIENTIFIC_CASES = [
    pytest.param("oscillator", OSCILLATOR, id="oscillator"),
    pytest.param("kepler", KEPLER_CIRCULAR, id="kepler-circular"),
    pytest.param("kepler", KEPLER_ECCENTRIC, id="kepler-eccentric"),
    pytest.param("heat1d", HEAT, id="heat1d"),
    pytest.param("wave1d", WAVE_SLOW, id="wave1d-slow"),
    pytest.param("wave1d", WAVE_FAST, id="wave1d-fast"),
    pytest.param("wave1d", WAVE_STARTUP, id="wave1d-startup"),
    pytest.param("wave1d", WAVE_FULL_PERIOD, id="wave1d-full-period"),
]


def _trusted_arrays(task_name: str, parameters: dict[str, object], work_dir: Path) -> ArrayMap:
    task_dir = TASKS_ROOT / task_name
    run = run_task(task_dir, task_dir, parameters, work_dir)
    assert run.passed, (run.gates, run.stderr)
    assert run.arrays is not None
    return run.arrays


@pytest.mark.parametrize(("task_name", "parameters"), SCIENTIFIC_CASES)
def test_trusted_package_passes_every_invariant_gate(
    tmp_path: Path, task_name: str, parameters: dict[str, object]
) -> None:
    contract = load_task_contract(TASKS_ROOT / task_name)
    gates = check_invariants(contract, parameters, _trusted_arrays(task_name, parameters, tmp_path))

    assert gates
    assert all(gate.passed for gate in gates), gates
    for gate in gates:
        assert gate.deviation is not None and gate.threshold is not None


def _scale_oscillator_velocity(arrays: ArrayMap) -> None:
    arrays["state"][:, 1] *= 1.01


def _drift_kepler_velocity(arrays: ArrayMap) -> None:
    arrays["state"][:, 2:] *= (1.0 + 1e-4 * arrays["time"])[:, None]


def _lift_heat_boundary(arrays: ArrayMap) -> None:
    arrays["state"][0] = 1e-3


def _amplify_wave(arrays: ArrayMap) -> None:
    arrays["state"] *= 1.1


@pytest.mark.parametrize(
    ("task_name", "parameters", "breakage", "expected_failures"),
    [
        pytest.param(
            "oscillator",
            OSCILLATOR,
            _scale_oscillator_velocity,
            {"energy_relative_drift"},
            id="oscillator-velocity-x1.01",
        ),
        pytest.param(
            "kepler",
            KEPLER_CIRCULAR,
            _drift_kepler_velocity,
            {"energy_relative_drift", "angular_momentum_relative_drift"},
            id="kepler-velocity-drift",
        ),
        pytest.param(
            "heat1d", HEAT, _lift_heat_boundary, {"dirichlet_boundary"}, id="heat1d-boundary"
        ),
        pytest.param(
            "wave1d",
            WAVE_FULL_PERIOD,
            _amplify_wave,
            {"amplitude_bound"},
            id="wave1d-amplitude-x1.1",
        ),
    ],
)
def test_broken_fixture_fails_expected_gate(
    tmp_path: Path,
    task_name: str,
    parameters: dict[str, object],
    breakage: Callable[[ArrayMap], None],
    expected_failures: set[str],
) -> None:
    contract = load_task_contract(TASKS_ROOT / task_name)
    arrays = {
        name: array.copy()
        for name, array in _trusted_arrays(task_name, parameters, tmp_path).items()
    }
    breakage(arrays)

    gates = check_invariants(contract, parameters, arrays)

    assert {gate.name for gate in gates if not gate.passed} == expected_failures
    for gate in gates:
        if not gate.passed:
            assert gate.deviation is not None and gate.threshold is not None
            assert gate.name in gate.detail


@pytest.mark.parametrize(("task_name", "parameters"), SCIENTIFIC_CASES[::3])
def test_invariant_gates_are_deterministic(
    tmp_path: Path, task_name: str, parameters: dict[str, object]
) -> None:
    contract = load_task_contract(TASKS_ROOT / task_name)
    first = check_invariants(
        contract, parameters, _trusted_arrays(task_name, parameters, tmp_path / "first")
    )
    second = check_invariants(
        contract, parameters, _trusted_arrays(task_name, parameters, tmp_path / "second")
    )

    assert first == second


def test_unknown_task_id_is_rejected() -> None:
    contract = load_task_contract(TASKS_ROOT / "oscillator").model_copy(update={"id": "unknown"})

    with pytest.raises(ValueError, match="unknown"):
        check_invariants(contract, {}, {})
