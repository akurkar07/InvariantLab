"""Scientific (hidden) tests for the Kepler task package.

The circular case uses the closed-form trajectory. The eccentric case is
compared to the independent DOP853 oracle rather than another Verlet update.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import solver
from conftest import make_input, write_input

from invariantlab.verification.analytical import kepler_circular_orbit, kepler_elliptic_orbit
from invariantlab.verification.kepler_oracle import solve_kepler_high_accuracy

if TYPE_CHECKING:
    from pathlib import Path

# The selected h=0.004 trajectories have O(h^2) global state error. The
# circular trajectory measures about 4.1e-6 relative L2 and the eccentric one
# about 7.6e-7, so the contract's 1e-5 state bound leaves a small method-order
# margin without admitting a first-order update.
STATE_RELATIVE_L2 = 1.0e-5

# Velocity Verlet is symplectic, so its Kepler energy error is bounded and
# oscillatory. The eccentric case's pericentre is the demanding case: its
# measured finite-horizon drift is about 3.6e-7, making 1e-6 a ~3x margin.
ENERGY_RELATIVE_DRIFT = 1.0e-6

# Central-force velocity Verlet preserves angular momentum to round-off here
# (about 1e-14 relative drift). A 1e-12 threshold allows floating-point
# accumulation but catches a non-central or incorrectly ordered update.
ANGULAR_MOMENTUM_RELATIVE_DRIFT = 1.0e-12

CIRCULAR_MU = 2.5
CIRCULAR_RADIUS = 1.7
CIRCULAR_PHASE = 0.37
CIRCULAR_DT = 0.004
CIRCULAR_N_STEPS = 1_080  # Horizon 4.32; not the period (about 8.81).

ECCENTRIC_MU = 1.9
ECCENTRIC_SEMI_MAJOR_AXIS = 2.3
ECCENTRICITY = 0.41
ECCENTRIC_PHASE = 0.63  # Non-special mean anomaly.
ECCENTRIC_DT = 0.004
ECCENTRIC_N_STEPS = 2_178  # Horizon 8.712; neither apsis nor orbital period.


def _state_from_orbit(
    values: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float]:
    return values[0], values[1], values[3], values[4]


def _candidate_trajectory(
    tmp_path: Path, initial: tuple[float, float, float, float], mu: float, dt: float, n_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    input_path = write_input(tmp_path / "input.json", make_input(*initial, mu, dt, n_steps))
    output_path = tmp_path / "result.npz"
    solver.run(input_path, output_path)
    with np.load(output_path) as archive:
        return archive["time"], archive["state"]


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
