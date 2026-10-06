"""Scientific hidden tests exercised only through the documented NPZ boundary."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import yaml
from conftest import ENTRYPOINT, TASK_ROOT, make_input, write_input

from invariantlab.verification.analytical import kepler_circular_orbit, kepler_elliptic_orbit
from invariantlab.verification.kepler_oracle import solve_kepler_high_accuracy

if TYPE_CHECKING:
    from pathlib import Path

STATE_RELATIVE_L2 = 1.0e-5
ENERGY_RELATIVE_DRIFT = 1.0e-6
ANGULAR_MOMENTUM_RELATIVE_DRIFT = 1.0e-12
CIRCULAR_MU, CIRCULAR_RADIUS, CIRCULAR_PHASE = 2.5, 1.7, 0.37
CIRCULAR_DT, CIRCULAR_N_STEPS = 0.004, 1_080
ECCENTRIC_MU, ECCENTRIC_SEMI_MAJOR_AXIS, ECCENTRICITY, ECCENTRIC_PHASE = 1.9, 2.3, 0.41, 0.63
ECCENTRIC_DT, ECCENTRIC_N_STEPS = 0.004, 2_178
CONTRACT = yaml.safe_load((TASK_ROOT / "contract.yaml").read_text(encoding="utf-8"))
EXPECTED_ARCHIVE_NAMES = {array["name"] for array in CONTRACT["output"]["arrays"]}


def _state_from_orbit(
    values: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float]:
    return values[0], values[1], values[3], values[4]


def _candidate_trajectory(
    tmp_path: Path, initial: tuple[float, float, float, float], mu: float, dt: float, n_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    input_path = write_input(tmp_path / "input.json", make_input(*initial, mu, dt, n_steps))
    output_path = tmp_path / "result.npz"
    completed = subprocess.run(
        [
            sys.executable,
            str(ENTRYPOINT.relative_to(TASK_ROOT)),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=TASK_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert output_path.is_file()
    with np.load(output_path) as archive:
        assert set(archive.files) == EXPECTED_ARCHIVE_NAMES
        time, state = archive["time"].copy(), archive["state"].copy()
    assert time.shape == (n_steps + 1,)
    assert state.shape == (n_steps + 1, 4)
    assert time.dtype == np.dtype(np.float64)
    assert state.dtype == np.dtype(np.float64)
    assert np.isfinite(time).all()
    assert np.isfinite(state).all()
    return time, state


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
    time, state = _candidate_trajectory(
        tmp_path, initial, CIRCULAR_MU, CIRCULAR_DT, CIRCULAR_N_STEPS
    )
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
    time, state = _candidate_trajectory(
        tmp_path, initial, ECCENTRIC_MU, ECCENTRIC_DT, ECCENTRIC_N_STEPS
    )
    oracle = solve_kepler_high_accuracy(*initial, ECCENTRIC_MU, time)
    state_relative_l2 = np.linalg.norm(state - oracle) / np.linalg.norm(oracle)
    energy_drift, angular_momentum_drift = _invariant_drifts(state, ECCENTRIC_MU)
    assert state_relative_l2 < STATE_RELATIVE_L2, f"state_relative_l2={state_relative_l2:.3e}"
    assert energy_drift < ENERGY_RELATIVE_DRIFT, f"energy_relative_drift={energy_drift:.3e}"
    assert angular_momentum_drift < ANGULAR_MOMENTUM_RELATIVE_DRIFT, (
        f"angular_momentum_relative_drift={angular_momentum_drift:.3e}"
    )
