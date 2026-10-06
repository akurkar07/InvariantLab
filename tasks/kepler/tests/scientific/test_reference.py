"""Scientific hidden tests exercised only through the documented NPZ boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import yaml
from conftest import TASK_ROOT, make_input

from invariantlab.verification.analytical import kepler_circular_orbit, kepler_elliptic_orbit
from invariantlab.verification.execution import run_task
from invariantlab.verification.kepler_oracle import solve_kepler_high_accuracy

if TYPE_CHECKING:
    from pathlib import Path

CIRCULAR_MU, CIRCULAR_RADIUS, CIRCULAR_PHASE = 2.5, 1.7, 0.37
CIRCULAR_DT, CIRCULAR_N_STEPS = 0.004, 1_080
ECCENTRIC_MU, ECCENTRIC_SEMI_MAJOR_AXIS, ECCENTRICITY, ECCENTRIC_PHASE = 1.9, 2.3, 0.41, 0.63
ECCENTRIC_DT, ECCENTRIC_N_STEPS = 0.004, 2_178
CONTRACT = yaml.safe_load((TASK_ROOT / "contract.yaml").read_text(encoding="utf-8"))
STATE_RELATIVE_L2 = CONTRACT["numerics"]["tolerances"]["state_relative_l2"]
ENERGY_RELATIVE_DRIFT = CONTRACT["numerics"]["tolerances"]["energy_relative_drift"]
ANGULAR_MOMENTUM_RELATIVE_DRIFT = CONTRACT["numerics"]["tolerances"][
    "angular_momentum_relative_drift"
]


def _state_from_orbit(
    values: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float]:
    return values[0], values[1], values[3], values[4]


def _invariant_drifts(state: np.ndarray, mu: float) -> tuple[float, float]:
    radius = np.hypot(state[:, 0], state[:, 1])
    energy = 0.5 * (state[:, 2] ** 2 + state[:, 3] ** 2) - mu / radius
    angular_momentum = state[:, 0] * state[:, 3] - state[:, 1] * state[:, 2]
    energy_drift = np.max(np.abs(energy - energy[0])) / abs(energy[0])
    angular_momentum_drift = np.max(np.abs(angular_momentum - angular_momentum[0])) / abs(
        angular_momentum[0]
    )
    return float(energy_drift), float(angular_momentum_drift)


def test_circular_trajectory_matches_analytical_state_and_conserves_invariants(
    tmp_path: Path,
) -> None:
    initial = _state_from_orbit(
        kepler_circular_orbit(0.0, CIRCULAR_MU, CIRCULAR_RADIUS, CIRCULAR_PHASE)
    )
    run = run_task(
        TASK_ROOT,
        TASK_ROOT,
        make_input(*initial, CIRCULAR_MU, CIRCULAR_DT, CIRCULAR_N_STEPS)["parameters"],
        tmp_path,
    )
    assert run.passed, (run.gates, run.stderr)
    assert run.arrays is not None
    time, state = run.arrays["time"], run.arrays["state"]
    assert time.shape == (CIRCULAR_N_STEPS + 1,)
    assert state.shape == (CIRCULAR_N_STEPS + 1, 4)
    expected = np.array(
        [
            _state_from_orbit(
                kepler_circular_orbit(t, CIRCULAR_MU, CIRCULAR_RADIUS, CIRCULAR_PHASE)
            )
            for t in time
        ],
        dtype=np.float64,
    )
    state_relative_l2 = np.linalg.norm(state - expected) / np.linalg.norm(expected)
    energy_drift, angular_momentum_drift = _invariant_drifts(state, CIRCULAR_MU)
    assert state_relative_l2 < STATE_RELATIVE_L2, f"state_relative_l2={state_relative_l2:.3e}"
    assert energy_drift < ENERGY_RELATIVE_DRIFT, f"energy_relative_drift={energy_drift:.3e}"
    assert angular_momentum_drift < ANGULAR_MOMENTUM_RELATIVE_DRIFT, (
        f"angular_momentum_relative_drift={angular_momentum_drift:.3e}"
    )


def test_eccentric_non_special_phase_matches_dop853_and_conserves_invariants(
    tmp_path: Path,
) -> None:
    initial = _state_from_orbit(
        kepler_elliptic_orbit(
            0.0, ECCENTRIC_MU, ECCENTRIC_SEMI_MAJOR_AXIS, ECCENTRICITY, ECCENTRIC_PHASE
        )
    )
    run = run_task(
        TASK_ROOT,
        TASK_ROOT,
        make_input(*initial, ECCENTRIC_MU, ECCENTRIC_DT, ECCENTRIC_N_STEPS)["parameters"],
        tmp_path,
    )
    assert run.passed, (run.gates, run.stderr)
    assert run.arrays is not None
    time, state = run.arrays["time"], run.arrays["state"]
    assert time.shape == (ECCENTRIC_N_STEPS + 1,)
    assert state.shape == (ECCENTRIC_N_STEPS + 1, 4)
    oracle = solve_kepler_high_accuracy(*initial, ECCENTRIC_MU, time)
    state_relative_l2 = np.linalg.norm(state - oracle) / np.linalg.norm(oracle)
    energy_drift, angular_momentum_drift = _invariant_drifts(state, ECCENTRIC_MU)
    assert state_relative_l2 < STATE_RELATIVE_L2, f"state_relative_l2={state_relative_l2:.3e}"
    assert energy_drift < ENERGY_RELATIVE_DRIFT, f"energy_relative_drift={energy_drift:.3e}"
    assert angular_momentum_drift < ANGULAR_MOMENTUM_RELATIVE_DRIFT, (
        f"angular_momentum_relative_drift={angular_momentum_drift:.3e}"
    )
