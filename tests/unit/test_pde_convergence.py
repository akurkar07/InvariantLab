"""M2 refinement evidence for the trusted heat and wave reference solvers.

Each experiment records relative L2 state error on its own grid at one fixed
physical horizon.  The heat temporal FTCS experiment uses the exact
semi-discrete fundamental-mode evolution to isolate its first-order time update
from the unavoidable fixed-grid O(dx**2) floor; all other errors use the
continuous manufactured/standing-wave analytical truth directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pytest

from invariantlab.verification.analytical import heat_trajectory, wave_standing_trajectory
from invariantlab.verification.solvers import (
    solve_heat_crank_nicolson,
    solve_heat_ftcs,
    solve_wave_leapfrog,
)

FIRST_ORDER_MIN = 0.8
FIRST_ORDER_MAX = 1.2
SECOND_ORDER_MIN = 1.8
SECOND_ORDER_MAX = 2.2


@dataclass(frozen=True)
class _Refinement:
    dx: float
    dt: float
    error: float


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))


def _uniform_grid(nx: int, length: float = 1.0) -> np.ndarray:
    """Construct the physical grid independently of a candidate solver."""
    return np.linspace(0.0, length, nx, dtype=np.float64)


def _observed_orders(refinements: list[_Refinement]) -> list[float]:
    """Return p = log(E_h / E_h_over_2) / log(2) for adjacent refinements."""
    return [
        math.log(coarse.error / fine.error) / math.log(2.0)
        for coarse, fine in pairwise(refinements)
    ]


def _assert_orders_in_band(
    refinements: list[_Refinement], lower: float, upper: float
) -> None:
    orders = _observed_orders(refinements)
    details = ", ".join(
        f"dx={refinement.dx:.6g}, dt={refinement.dt:.6g}, "
        f"error={refinement.error:.6e}, order={order:.6f}"
        for refinement, order in zip(refinements[1:], orders, strict=True)
    )
    assert all(lower <= order <= upper for order in orders), details


def _heat_semidiscrete_fundamental_mode(x: np.ndarray, alpha: float, horizon: float) -> np.ndarray:
    """Exact method-of-lines evolution of the manufactured fundamental mode."""
    dx = float(x[1] - x[0])
    eigenvalue = -4.0 * math.sin(math.pi * dx / 2.0) ** 2 / dx**2
    return np.sin(math.pi * x) * math.exp(alpha * eigenvalue * horizon)


def _legacy_fixed_c_wave(
    nx: int, nt: int, c: float, length: float, horizon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Local reproduction of the former defect: C=0.5 ignores physical inputs."""
    del c, horizon
    x = np.linspace(0.0, length, nx, dtype=np.float64)
    state = np.sin(np.pi * x / length)
    state[[0, -1]] = 0.0
    previous = state.copy()
    fixed_courant = 0.5
    previous[1:-1] += 0.5 * fixed_courant**2 * (
        state[2:] - 2.0 * state[1:-1] + state[:-2]
    )

    for _ in range(nt):
        next_state = np.zeros(nx, dtype=np.float64)
        next_state[1:-1] = (
            2.0 * state[1:-1]
            - previous[1:-1]
            + fixed_courant**2 * (state[2:] - 2.0 * state[1:-1] + state[:-2])
        )
        previous, state = state, next_state
    return x, state


def test_ftcs_has_first_order_temporal_convergence() -> None:
    # Fixed nx=161 (dx=0.00625), alpha=0.1, T=0.1. At r=0.5, 0.25, and
    # 0.125, raw PDE error mixes fixed O(dx**2) spatial error with O(dt) time
    # error. The exact semi-discrete manufactured mode below isolates FTCS time.
    # Measured (dt, temporal relative L2):
    # (1.953125e-4, 9.5131726e-6), (9.765625e-5, 4.7562920e-6),
    # (4.8828125e-5, 2.3780724e-6); orders 1.000089, 1.000045.
    alpha, horizon, nx = 0.1, 0.1, 161
    step_counts = (512, 1_024, 2_048)
    assert step_counts[1] == 2 * step_counts[0]
    assert step_counts[2] == 2 * step_counts[1]
    refinements: list[_Refinement] = []

    for nt in step_counts:
        dt = horizon / nt
        x, state = solve_heat_ftcs(nx, nt, alpha, t_final=horizon)
        expected_x = _uniform_grid(nx)
        np.testing.assert_array_equal(x, expected_x)
        dx = float(expected_x[1] - expected_x[0])
        ratio = alpha * dt / dx**2
        assert ratio <= 0.5
        expected = _heat_semidiscrete_fundamental_mode(expected_x, alpha, horizon)
        refinements.append(_Refinement(dx, dt, _relative_l2(state, expected)))

    assert all(math.isclose(refinement.dt * nt, horizon) for refinement, nt in zip(refinements, step_counts, strict=True))
    _assert_orders_in_band(refinements, FIRST_ORDER_MIN, FIRST_ORDER_MAX)


def test_ftcs_has_second_order_cfl_coupled_spatial_convergence() -> None:
    # Coupled dt = 0.4*dx**2/alpha keeps FTCS stable (r=0.4), making its
    # O(dt) temporal error O(dx**2). Measured errors: 7.1110593e-5,
    # 1.7762056e-5, 4.4395407e-6; orders 2.001266, 2.000316.
    alpha, horizon = 0.1, 0.1
    configurations = ((41, 40), (81, 160), (161, 640))
    refinements: list[_Refinement] = []

    for nx, nt in configurations:
        x, state = solve_heat_ftcs(nx, nt, alpha, t_final=horizon)
        expected_x = _uniform_grid(nx)
        np.testing.assert_array_equal(x, expected_x)
        dx = float(expected_x[1] - expected_x[0])
        dt = horizon / nt
        assert math.isclose(alpha * dt / dx**2, 0.4)
        expected = heat_trajectory(expected_x, horizon, alpha)
        refinements.append(_Refinement(dx, dt, _relative_l2(state, expected)))

    assert configurations[1][0] - 1 == 2 * (configurations[0][0] - 1)
    assert configurations[2][0] - 1 == 2 * (configurations[1][0] - 1)
    assert all(math.isclose(refinement.dt * nt, horizon) for refinement, (_, nt) in zip(refinements, configurations, strict=True))
    _assert_orders_in_band(refinements, SECOND_ORDER_MIN, SECOND_ORDER_MAX)


def test_crank_nicolson_has_second_order_temporal_convergence() -> None:
    # Fixed nx=1601 (dx=0.000625), alpha=0.1, T=0.16. The fine spatial grid
    # keeps manufactured-solution spatial error below the temporal signal.
    # Measured errors: 2.0463470e-5, 5.0769581e-6, 1.2311368e-6; orders
    # 2.011015, 2.043973.
    alpha, horizon, nx = 0.1, 0.16, 1_601
    step_counts = (4, 8, 16)
    assert step_counts[1] == 2 * step_counts[0]
    assert step_counts[2] == 2 * step_counts[1]
    refinements: list[_Refinement] = []

    for nt in step_counts:
        dt = horizon / nt
        x, state = solve_heat_crank_nicolson(nx, nt, alpha, t_final=horizon)
        expected_x = _uniform_grid(nx)
        np.testing.assert_array_equal(x, expected_x)
        expected = heat_trajectory(expected_x, horizon, alpha)
        refinements.append(
            _Refinement(float(expected_x[1] - expected_x[0]), dt, _relative_l2(state, expected))
        )

    assert all(math.isclose(refinement.dt * nt, horizon) for refinement, nt in zip(refinements, step_counts, strict=True))
    _assert_orders_in_band(refinements, SECOND_ORDER_MIN, SECOND_ORDER_MAX)


def test_crank_nicolson_has_second_order_spatial_convergence() -> None:
    # Coupled dt=0.1*dx**2 makes CN's O(dt**2) temporal error O(dx**4), below
    # its O(dx**2) spatial error. Measured errors: 5.0724726e-5, 1.2682902e-5,
    # 3.1708350e-6; orders 1.999804, 1.999950.
    alpha, horizon = 0.1, 0.1
    configurations = ((41, 1_600), (81, 6_400), (161, 25_600))
    refinements: list[_Refinement] = []

    for nx, nt in configurations:
        x, state = solve_heat_crank_nicolson(nx, nt, alpha, t_final=horizon)
        expected_x = _uniform_grid(nx)
        np.testing.assert_array_equal(x, expected_x)
        dx = float(expected_x[1] - expected_x[0])
        dt = horizon / nt
        assert math.isclose(dt, 0.1 * dx**2)
        expected = heat_trajectory(expected_x, horizon, alpha)
        refinements.append(_Refinement(dx, dt, _relative_l2(state, expected)))

    assert configurations[1][0] - 1 == 2 * (configurations[0][0] - 1)
    assert configurations[2][0] - 1 == 2 * (configurations[1][0] - 1)
    assert all(math.isclose(refinement.dt * nt, horizon) for refinement, (_, nt) in zip(refinements, configurations, strict=True))
    _assert_orders_in_band(refinements, SECOND_ORDER_MIN, SECOND_ORDER_MAX)


def test_wave_leapfrog_has_second_order_cfl_preserving_convergence() -> None:
    # c != 1, length != 1, and T=0.39 gives phase about 0.706 rad: neither a
    # zero crossing nor a phase extremum. The fixed physical CFL is C=0.3.
    # Measured errors: 3.5299546e-5, 8.8250708e-6, 2.2062792e-6; orders
    # 1.999970, 1.999992.
    c, length, horizon = 0.75, 1.3, 0.39
    configurations = ((81, 60), (161, 120), (321, 240))
    refinements: list[_Refinement] = []

    for nx, nt in configurations:
        x, state = solve_wave_leapfrog(nx, nt, c, length, horizon)
        expected_x = _uniform_grid(nx, length)
        np.testing.assert_array_equal(x, expected_x)
        dx = float(expected_x[1] - expected_x[0])
        dt = horizon / nt
        courant = c * dt / dx
        assert math.isclose(courant, 0.3)
        expected = wave_standing_trajectory(expected_x, horizon, c, length=length)
        refinements.append(_Refinement(dx, dt, _relative_l2(state, expected)))

    assert configurations[1][0] - 1 == 2 * (configurations[0][0] - 1)
    assert configurations[2][0] - 1 == 2 * (configurations[1][0] - 1)
    assert all(math.isclose(refinement.dt * nt, horizon) for refinement, (_, nt) in zip(refinements, configurations, strict=True))
    _assert_orders_in_band(refinements, SECOND_ORDER_MIN, SECOND_ORDER_MAX)


def test_second_order_check_rejects_legacy_fixed_courant_wave_fixture() -> None:
    # Fixed C=0.5 models the wrong physical speed when requested C=0.3. Its
    # errors plateau at 4.9666901e-1, 4.9672075e-1, and 4.9673368e-1, yielding
    # orders -0.000150 and -0.000038 rather than second order.
    c, length, horizon = 0.75, 1.3, 0.39
    configurations = ((81, 60), (161, 120), (321, 240))
    refinements: list[_Refinement] = []

    for nx, nt in configurations:
        x, state = _legacy_fixed_c_wave(nx, nt, c, length, horizon)
        expected_x = _uniform_grid(nx, length)
        np.testing.assert_array_equal(x, expected_x)
        dx = float(expected_x[1] - expected_x[0])
        dt = horizon / nt
        expected = wave_standing_trajectory(expected_x, horizon, c, length=length)
        refinements.append(_Refinement(dx, dt, _relative_l2(state, expected)))

    with pytest.raises(AssertionError):
        _assert_orders_in_band(refinements, SECOND_ORDER_MIN, SECOND_ORDER_MAX)
