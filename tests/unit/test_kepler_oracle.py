"""Focused tests for the independent DOP853 Kepler oracle."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from invariantlab.verification.analytical import kepler_circular_orbit, kepler_elliptic_orbit
from invariantlab.verification.kepler_oracle import ATOL, RTOL, solve_kepler_high_accuracy
from invariantlab.verification.solvers import solve_kepler_verlet

# DOP853 at rtol=1e-11 and atol=1e-13 is substantially more accurate than the
# ordinary Verlet method here. The existing elliptic analytical helper stops its
# Newton solve at 1e-12 in eccentric anomaly, which translates to a measured
# state discrepancy below 3e-10; this 4e-10 threshold allows only that oracle
# rounding, not a low-order trajectory error.
ANALYTICAL_COMPARISON_ATOL = 4.0e-10

# A deliberately perturbed component should be rejected against the high-
# accuracy state, while a finely stepped ordinary Verlet state still passes.
ORACLE_COMPARISON_ATOL = 1.0e-6


def _state_from_analytical(values: tuple[float, float, float, float, float, float]) -> np.ndarray:
    rx, ry, _, vx, vy, _ = values
    return np.array([rx, ry, vx, vy], dtype=np.float64)


def _matches_oracle(candidate: np.ndarray, oracle: np.ndarray) -> bool:
    return bool(np.allclose(candidate, oracle, rtol=0.0, atol=ORACLE_COMPARISON_ATOL))


def test_dop853_oracle_matches_nontrivial_circular_position_and_velocity() -> None:
    mu, radius, phase = 2.5, 1.7, 0.37
    times = np.linspace(0.0, 4.321, 57, dtype=np.float64)
    initial = _state_from_analytical(kepler_circular_orbit(0.0, mu, radius, phase))

    states = solve_kepler_high_accuracy(*initial, mu, times)
    expected = np.array(
        [_state_from_analytical(kepler_circular_orbit(t, mu, radius, phase)) for t in times]
    )

    np.testing.assert_allclose(states[:, :2], expected[:, :2], rtol=0.0, atol=ANALYTICAL_COMPARISON_ATOL)
    np.testing.assert_allclose(states[:, 2:], expected[:, 2:], rtol=0.0, atol=ANALYTICAL_COMPARISON_ATOL)
    assert states.dtype == np.dtype(np.float64)
    assert np.isfinite(states).all()


def test_dop853_oracle_matches_eccentric_non_special_phase_orbit() -> None:
    # e > 0, phase is a non-special mean anomaly, and 8.71 is neither an apsis
    # nor an integer period for this orbit (period is approximately 15.9).
    mu, semi_major_axis, eccentricity, phase = 1.9, 2.3, 0.41, 0.63
    times = np.linspace(0.0, 8.71, 89, dtype=np.float64)
    initial = _state_from_analytical(
        kepler_elliptic_orbit(0.0, mu, semi_major_axis, eccentricity, phase)
    )

    states = solve_kepler_high_accuracy(*initial, mu, times)
    expected = np.array(
        [
            _state_from_analytical(
                kepler_elliptic_orbit(t, mu, semi_major_axis, eccentricity, phase)
            )
            for t in times
        ]
    )

    np.testing.assert_allclose(states[:, :2], expected[:, :2], rtol=0.0, atol=ANALYTICAL_COMPARISON_ATOL)
    np.testing.assert_allclose(states[:, 2:], expected[:, 2:], rtol=0.0, atol=ANALYTICAL_COMPARISON_ATOL)


def test_dop853_oracle_is_deterministic_and_uses_required_tolerances() -> None:
    assert RTOL <= 1.0e-11
    assert ATOL <= 1.0e-13
    times = np.linspace(0.0, 2.37, 31, dtype=np.float64)

    first = solve_kepler_high_accuracy(1.2, -0.4, 0.3, 1.1, 1.7, times)
    second = solve_kepler_high_accuracy(1.2, -0.4, 0.3, 1.1, 1.7, times)

    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((1.0, 0.0, 0.0, 1.0, 0.0, [0.0, 1.0]), "mu"),
        ((0.0, 0.0, 0.0, 1.0, 1.0, [0.0, 1.0]), "radius"),
        ((1.0, 0.0, math.nan, 1.0, 1.0, [0.0, 1.0]), "vx"),
        ((1.0, 0.0, 0.0, 1.0, 1.0, [0.0]), "at least two"),
        ((1.0, 0.0, 0.0, 1.0, 1.0, [0.1, 1.0]), "begin at zero"),
        ((1.0, 0.0, 0.0, 1.0, 1.0, [0.0, 1.0, 0.5]), "strictly increasing"),
    ],
)
def test_dop853_oracle_rejects_invalid_initial_value_problems(arguments, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        solve_kepler_high_accuracy(*arguments)


def test_dop853_oracle_reports_adaptive_integration_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import invariantlab.verification.kepler_oracle as oracle_module

    monkeypatch.setattr(
        oracle_module,
        "solve_ivp",
        lambda *args, **kwargs: SimpleNamespace(success=False, message="step size stalled"),
    )

    with pytest.raises(RuntimeError, match="DOP853 Kepler integration failed: step size stalled"):
        solve_kepler_high_accuracy(1.0, 0.0, 0.0, 1.0, 1.0, [0.0, 1.0])


def test_oracle_rejects_deliberately_perturbed_verlet_final_state() -> None:
    mu, radius, dt, n_steps = 1.3, 1.4, 1.0e-4, 1_000
    initial = _state_from_analytical(kepler_circular_orbit(0.0, mu, radius, phase=0.29))
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    oracle = solve_kepler_high_accuracy(*initial, mu, times)
    verlet = solve_kepler_verlet(*initial, mu, dt, n_steps)
    verlet_final_state = verlet[-1, 1:5]

    assert _matches_oracle(verlet_final_state, oracle[-1])
    perturbed = verlet_final_state.copy()
    perturbed[0] += 1.0e-3
    assert not _matches_oracle(perturbed, oracle[-1])
