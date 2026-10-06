"""M2 convergence evidence for the trusted ODE Velocity Verlet solvers.

Each experiment uses relative L2 state error over its entire trajectory.  Every
refinement uses the same physical horizon, with ``dt = horizon / n_steps``.
The measured values in this asymptotic regime are documented beside each case
so an update-order regression is visible in the failing assertion.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from invariantlab.verification.analytical import (
    kepler_circular_orbit,
    kepler_elliptic_orbit,
    oscillator_trajectory,
)
from invariantlab.verification.kepler_oracle import solve_kepler_high_accuracy
from invariantlab.verification.solvers import solve_kepler_verlet, solve_oscillator_verlet

SECOND_ORDER_MIN = 1.8
SECOND_ORDER_MAX = 2.2


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))


def _observed_orders(refinements: list[tuple[float, float]]) -> list[float]:
    """Return p = log(E_h / E_h_over_2) / log(2) for adjacent refinements."""
    return [
        math.log(coarse_error / fine_error) / math.log(2.0)
        for (_, coarse_error), (_, fine_error) in pairwise(refinements)
    ]


def _assert_second_order(refinements: list[tuple[float, float]]) -> None:
    orders = _observed_orders(refinements)
    details = ", ".join(
        f"dt={dt:.6g}, error={error:.6e}, order={order:.6f}"
        for (dt, error), order in zip(refinements[1:], orders, strict=True)
    )
    assert all(SECOND_ORDER_MIN <= order <= SECOND_ORDER_MAX for order in orders), details


def _state_from_orbit(
    values: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    return np.array([values[0], values[1], values[3], values[4]], dtype=np.float64)


def _forward_euler_oscillator(
    x0: float, v0: float, omega: float, dt: float, n_steps: int
) -> np.ndarray:
    """Deliberately local first-order fixture used to prove the gate bites."""
    state = np.empty((n_steps + 1, 2), dtype=np.float64)
    state[0] = (x0, v0)
    x, v = x0, v0
    for index in range(n_steps):
        x, v = x + dt * v, v - dt * omega**2 * x
        state[index + 1] = (x, v)
    return state


def test_oscillator_velocity_verlet_has_second_order_trajectory_convergence() -> None:
    # x0 != 0, v0 != 0, omega != 1, horizon 4.32 is not a special phase.
    # Measured (dt, relative L2 error):
    # (0.008, 4.1199165e-05), (0.004, 1.0301745e-05), (0.002, 2.5756875e-06).
    # Adjacent orders: 1.999726, 1.999859.
    x0, v0, omega, horizon = 0.7, -0.35, 1.7, 4.32
    step_counts = (540, 1_080, 2_160)
    assert step_counts[1] == 2 * step_counts[0]
    assert step_counts[2] == 2 * step_counts[1]
    refinements: list[tuple[float, float]] = []

    for n_steps in step_counts:
        dt = horizon / n_steps
        times = dt * np.arange(n_steps + 1, dtype=np.float64)
        trajectory = solve_oscillator_verlet(x0, v0, omega, dt, n_steps)
        np.testing.assert_allclose(trajectory[:, 0], times, rtol=0.0, atol=1.0e-12)
        exact = oscillator_trajectory(times, x0, v0, omega)[:, 1:]
        refinements.append((dt, _relative_l2(trajectory[:, 1:], exact)))

    assert all(math.isclose(dt * step_counts[index], horizon) for index, (dt, _) in enumerate(refinements))
    _assert_second_order(refinements)


def test_circular_kepler_velocity_verlet_has_second_order_trajectory_convergence() -> None:
    # mu != 1, radius != 1, nonzero phase; horizon 4.32 is not its ~8.81 period.
    # Measured (dt, relative L2 error):
    # (0.008, 1.6486987e-05), (0.004, 4.1190212e-06), (0.002, 1.0294118e-06).
    # Adjacent orders: 2.000954, 2.000481.
    mu, radius, phase, horizon = 2.5, 1.7, 0.37, 4.32
    initial = _state_from_orbit(kepler_circular_orbit(0.0, mu, radius, phase))
    step_counts = (540, 1_080, 2_160)
    assert step_counts[1] == 2 * step_counts[0]
    assert step_counts[2] == 2 * step_counts[1]
    refinements: list[tuple[float, float]] = []

    for n_steps in step_counts:
        dt = horizon / n_steps
        times = dt * np.arange(n_steps + 1, dtype=np.float64)
        trajectory = solve_kepler_verlet(*initial, mu, dt, n_steps)
        np.testing.assert_allclose(trajectory[:, 0], times, rtol=0.0, atol=1.0e-12)
        exact = np.array(
            [_state_from_orbit(kepler_circular_orbit(time, mu, radius, phase)) for time in times]
        )
        refinements.append((dt, _relative_l2(trajectory[:, 1:5], exact)))

    assert all(math.isclose(dt * step_counts[index], horizon) for index, (dt, _) in enumerate(refinements))
    _assert_second_order(refinements)


def test_eccentric_kepler_velocity_verlet_has_second_order_trajectory_convergence() -> None:
    # e > 0, phase is a non-special mean anomaly; 8.712 is neither an apsis nor
    # the ~15.9 period. DOP853 (rtol=1e-11, atol=1e-13) is the independent truth.
    # Measured (dt, relative L2 error):
    # (0.008, 3.0064326e-06), (0.004, 7.5151268e-07), (0.002, 1.8786795e-07).
    # Adjacent orders: 2.000183, 2.000078.
    mu, semi_major_axis, eccentricity, phase, horizon = 1.9, 2.3, 0.41, 0.63, 8.712
    initial = _state_from_orbit(
        kepler_elliptic_orbit(0.0, mu, semi_major_axis, eccentricity, phase)
    )
    step_counts = (1_089, 2_178, 4_356)
    assert step_counts[1] == 2 * step_counts[0]
    assert step_counts[2] == 2 * step_counts[1]
    refinements: list[tuple[float, float]] = []

    for n_steps in step_counts:
        dt = horizon / n_steps
        times = dt * np.arange(n_steps + 1, dtype=np.float64)
        trajectory = solve_kepler_verlet(*initial, mu, dt, n_steps)
        np.testing.assert_allclose(trajectory[:, 0], times, rtol=0.0, atol=1.0e-12)
        oracle = solve_kepler_high_accuracy(*initial, mu, times)
        refinements.append((dt, _relative_l2(trajectory[:, 1:5], oracle)))

    assert all(math.isclose(dt * step_counts[index], horizon) for index, (dt, _) in enumerate(refinements))
    _assert_second_order(refinements)


def test_second_order_check_rejects_local_first_order_oscillator_fixture() -> None:
    # Measured forward-Euler errors are 2.8875812e-02, 1.4301428e-02, and
    # 7.1168685e-03, yielding orders 1.013702 and 1.006845: below 1.8.
    x0, v0, omega, horizon = 0.7, -0.35, 1.7, 4.32
    step_counts = (540, 1_080, 2_160)
    assert step_counts[1] == 2 * step_counts[0]
    assert step_counts[2] == 2 * step_counts[1]
    refinements: list[tuple[float, float]] = []

    for n_steps in step_counts:
        dt = horizon / n_steps
        state = _forward_euler_oscillator(x0, v0, omega, dt, n_steps)
        exact = oscillator_trajectory(dt * np.arange(n_steps + 1), x0, v0, omega)[:, 1:]
        refinements.append((dt, _relative_l2(state, exact)))

    with pytest.raises(AssertionError):
        _assert_second_order(refinements)
