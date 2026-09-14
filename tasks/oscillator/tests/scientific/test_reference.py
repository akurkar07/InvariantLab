"""Scientific (hidden) tests for the oscillator task package.

The candidate archive is compared against the trusted analytical solution using
deliberately non-special parameters: x0 != 0, v0 != 0, omega != 1, a horizon that
is not a whole or half period, and a timestep that does not align with any
special phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import solver
from conftest import make_input, write_input

from invariantlab.verification.analytical import oscillator_trajectory

if TYPE_CHECKING:
    from pathlib import Path

X0 = 0.7
V0 = -0.35
OMEGA = 1.7
DT = 1e-3
N_STEPS = 15_000  # horizon 15.0 = 4.06 periods of 2*pi/1.7

# Velocity Verlet is second order: the global phase error after T is about
# omega*T*(omega*dt)^2/24 ~ 3.1e-6 for these values, so the relative L2 error of
# the whole trajectory should sit near 2e-6. The contract's state_relative_l2
# (1e-5) leaves a 5x margin without admitting a first-order or off-by-one-step
# scheme (both exceed 1e-4 here).
STATE_RELATIVE_L2 = 1.0e-5

# Verlet is symplectic: energy oscillates with bounded relative amplitude
# ~(omega*dt)^2/8 ~ 3.6e-7 and does not grow secularly. The contract's
# energy_relative_drift (1e-6) bounds that oscillation with ~3x margin; a
# non-symplectic explicit scheme drifts by orders of magnitude more over 15 s.
ENERGY_RELATIVE_DRIFT = 1.0e-6


def _candidate_archive(tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    input_path = write_input(tmp_path / "input.json", make_input(X0, V0, OMEGA, DT, N_STEPS))
    output_path = tmp_path / "result.npz"
    solver.run(input_path, output_path)
    with np.load(output_path) as archive:
        return archive["time"], archive["state"]


def test_state_matches_analytical_solution(tmp_path: Path) -> None:
    time, state = _candidate_archive(tmp_path)
    exact = oscillator_trajectory(time, X0, V0, OMEGA)[:, 1:]

    relative_l2 = np.linalg.norm(state - exact) / np.linalg.norm(exact)

    assert relative_l2 < STATE_RELATIVE_L2, f"state_relative_l2={relative_l2:.3e}"


def test_energy_drift_is_bounded_over_horizon(tmp_path: Path) -> None:
    _, state = _candidate_archive(tmp_path)
    energy = 0.5 * state[:, 1] ** 2 + 0.5 * OMEGA**2 * state[:, 0] ** 2

    relative_drift = np.max(np.abs(energy - energy[0])) / energy[0]

    assert relative_drift < ENERGY_RELATIVE_DRIFT, f"energy_relative_drift={relative_drift:.3e}"
