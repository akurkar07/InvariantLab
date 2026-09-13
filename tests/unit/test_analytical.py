"""Unit tests for analytical oracles and reference solvers."""

from __future__ import annotations

import math
import sys
from typing import cast

import numpy as np
import pytest

from invariantlab.verification.analytical import (
    heat_manufactured_solution,
    heat_trajectory,
    kepler_angular_momentum,
    kepler_circular_orbit,
    kepler_circular_velocity,
    kepler_eccentric_anomaly,
    kepler_orbital_energy,
    oscillator_analytic,
    oscillator_energy,
    oscillator_trajectory,
    wave_standing_solution,
    wave_standing_trajectory,
    wave_standing_velocity,
)
from invariantlab.verification.solvers import (
    solve_heat_crank_nicolson,
    solve_heat_ftcs,
    solve_kepler_verlet,
    solve_oscillator_verlet,
    solve_wave_leapfrog,
)

# -- Oscillator ---------------------------------------------------------------


def test_oscillator_circular_orbit():
    """For x0=1, v0=0, omega=1: x(t)=cos(t), v(t)=-sin(t)."""
    x, v = oscillator_analytic(0.0, 1.0, 0.0, 1.0)
    assert math.isclose(x, 1.0, abs_tol=1e-15)
    assert math.isclose(v, 0.0, abs_tol=1e-15)


def test_oscillator_energy_conservation():
    """Energy should be constant for the analytic solution."""
    x0, v0, omega = 2.0, 1.5, 3.0
    e0 = oscillator_energy(x0, v0, omega)
    for t in [0.0, 0.5, 1.0, 5.0, 10.0]:
        x, v = oscillator_analytic(t, x0, v0, omega)
        e = oscillator_energy(x, v, omega)
        assert math.isclose(e, e0, rel_tol=1e-12), f"t={t}: e={e}, e0={e0}"


def test_oscillator_period():
    """After one full period T=2pi/omega, the state should return to start."""
    x0, v0, omega = 1.5, -0.5, 2.0
    period = 2 * math.pi / omega
    x, v = oscillator_analytic(period, x0, v0, omega)
    assert math.isclose(x, x0, abs_tol=1e-12)
    assert math.isclose(v, v0, abs_tol=1e-12)


def test_oscillator_trajectory_shape():
    times = np.linspace(0, 10, 100)
    traj = oscillator_trajectory(times, 1.0, 0.0, 1.0)
    assert traj.shape == (100, 3)


def test_verlet_matches_analytic():
    """Velocity Verlet should closely match the analytical solution."""
    x0, v0, omega = 1.0, 0.0, 2.0
    dt = 0.001
    n_steps = 1000
    traj = solve_oscillator_verlet(x0, v0, omega, dt, n_steps)
    t_final = traj[-1, 0]
    x_num = traj[-1, 1]
    v_num = traj[-1, 2]
    x_exact, v_exact = oscillator_analytic(t_final, x0, v0, omega)
    assert math.isclose(x_num, x_exact, abs_tol=1e-4)
    assert math.isclose(v_num, v_exact, abs_tol=1e-4)


def test_verlet_energy_drift():
    """Velocity Verlet should conserve energy well over many periods."""
    x0, v0, omega = 1.0, 0.0, 1.0
    e0 = oscillator_energy(x0, v0, omega)
    dt = 0.001
    n_steps = 100000  # ~16 periods
    traj = solve_oscillator_verlet(x0, v0, omega, dt, n_steps)
    x_final, v_final = traj[-1, 1], traj[-1, 2]
    e_final = oscillator_energy(x_final, v_final, omega)
    drift = abs(e_final - e0) / abs(e0)
    assert drift < 1e-6, f"Energy drift {drift} too large"


# -- Kepler -------------------------------------------------------------------


def test_circular_orbit_velocity():
    """v_circ = sqrt(mu/r) gives zero orbital energy derivative."""
    mu, r = 1.0, 1.0
    v = kepler_circular_velocity(mu, r)
    assert math.isclose(v, 1.0, abs_tol=1e-15)


def test_circular_orbit_energy_conservation():
    """Circular orbit has constant negative energy: epsilon = -mu/(2r)."""
    mu, r0 = 4.0, 2.0
    v = kepler_circular_velocity(mu, r0)
    e_exact = -mu / (2 * r0)
    e_computed = kepler_orbital_energy(0.0, v, r0, 0.0, mu)
    assert math.isclose(e_computed, e_exact, rel_tol=1e-12)


def test_circular_orbit_angular_momentum():
    """h = r cross v = r v for a circular orbit."""
    mu, r0 = 4.0, 2.0
    v = kepler_circular_velocity(mu, r0)
    h = kepler_angular_momentum(r0, 0.0, 0.0, v)
    assert math.isclose(h, r0 * v, rel_tol=1e-12)


def test_circular_orbit_period():
    """After one period, circular orbit returns to the same state."""
    mu, r0 = 1.0, 1.0
    omega = math.sqrt(mu / r0 ** 3)
    period = 2 * math.pi / omega
    state_0 = kepler_circular_orbit(0.0, mu, r0)
    state_t = kepler_circular_orbit(period, mu, r0)
    for a, b in zip(state_0, state_t, strict=True):
        assert math.isclose(a, b, abs_tol=1e-10)


def test_kepler_equation_solution():
    """E - e sin(E) = M for the computed E."""
    ma, ecc = 1.0, 0.3
    ea = kepler_eccentric_anomaly(ma, ecc)
    assert math.isclose(ea - ecc * math.sin(ea), ma, abs_tol=1e-12)


def test_verlet_kepler_energy_conservation():
    """Velocity Verlet should conserve energy well for Kepler problem."""
    mu, r0 = 1.0, 1.0
    v0 = kepler_circular_velocity(mu, r0)
    dt = 0.001
    n_steps = 10000
    traj = solve_kepler_verlet(r0, 0.0, 0.0, v0, mu, dt, n_steps)
    e0 = traj[0, 5]
    e_final = traj[-1, 5]
    drift = abs(e_final - e0) / abs(e0)
    assert drift < 1e-5, f"Energy drift {drift} too large"


# -- Heat Equation ------------------------------------------------------------


def test_heat_manufactured_solution_initial():
    """At t=0: u(x,0) = sin(pi x)."""
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
        u = heat_manufactured_solution(x, 0.0, 1.0)
        assert math.isclose(u, math.sin(math.pi * x), abs_tol=1e-15)


def test_heat_manufactured_solution_decay():
    """u should decay over time."""
    x = 0.5
    u0 = heat_manufactured_solution(x, 0.0, 1.0)
    u1 = heat_manufactured_solution(x, 0.1, 1.0)
    assert u1 < u0


def test_heat_manufactured_boundary():
    """u(0,t) = u(1,t) = 0 for all t."""
    for t in [0.0, 0.1, 0.5]:
        assert math.isclose(heat_manufactured_solution(0.0, t, 1.0), 0.0, abs_tol=1e-15)
        assert math.isclose(heat_manufactured_solution(1.0, t, 1.0), 0.0, abs_tol=1e-15)


def test_ftcs_matches_manufactured():
    """FTCS should closely match the manufactured solution."""
    x, u_num = solve_heat_ftcs(101, 1000, alpha=0.1, length=1.0, t_final=0.1)
    u_exact = heat_trajectory(x, 0.1, 0.1)
    max_error = np.max(np.abs(u_num - u_exact))
    assert max_error < 1e-4, f"Max error {max_error} too large"


def test_crank_nicolson_matches_manufactured():
    """Crank-Nicolson should closely match the manufactured solution."""
    x, u_num = solve_heat_crank_nicolson(101, 500, alpha=0.1, length=1.0, t_final=0.1)
    u_exact = heat_trajectory(x, 0.1, 0.1)
    max_error = np.max(np.abs(u_num - u_exact))
    assert max_error < 1e-4, f"Max error {max_error} too large"


def test_ftcs_stability_check():
    """FTCS should raise when r > 0.5."""
    with pytest.raises(ValueError):
        solve_heat_ftcs(10, 1, alpha=1.0, length=1.0, t_final=1.0)


# -- Wave Equation ------------------------------------------------------------


def test_wave_standing_initial():
    """At t=0: u(x,0) = sin(pi x)."""
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
        u = wave_standing_solution(x, 0.0, 1.0)
        assert math.isclose(u, math.sin(math.pi * x), abs_tol=1e-15)


def test_wave_standing_velocity_initial():
    """At t=0: v(x,0) = 0 (cos(0)=1)."""
    for x in [0.0, 0.25, 0.5, 1.0]:
        v = wave_standing_velocity(x, 0.0, 1.0)
        assert math.isclose(v, 0.0, abs_tol=1e-15)


def test_wave_standing_boundary():
    """u(0,t) = u(L,t) = 0 for all t."""
    for t in [0.0, 0.1, 0.5, 1.0]:
        assert math.isclose(wave_standing_solution(0.0, t, 1.0), 0.0, abs_tol=1e-15)
        assert math.isclose(wave_standing_solution(1.0, t, 1.0), 0.0, abs_tol=1e-15)


def test_wave_period():
    """After one period T=2L/c, the wave returns to its initial state."""
    c, wave_length = 1.0, 1.0
    period = 2 * wave_length / c
    for x in [0.1, 0.3, 0.5, 0.7]:
        u0 = wave_standing_solution(x, 0.0, c, length=wave_length)
        ut = wave_standing_solution(x, period, c, length=wave_length)
        assert math.isclose(u0, ut, abs_tol=1e-10)


def test_leapfrog_matches_standing():
    """Leapfrog should closely match the standing-wave solution."""
    c, wave_length = 1.0, 1.0
    t_final = 0.5
    x, u_num = solve_wave_leapfrog(401, 2000, c, length=wave_length, t_final=t_final)
    u_exact = wave_standing_trajectory(x, t_final, c, length=wave_length)
    max_error = np.max(np.abs(u_num - u_exact))
    assert max_error < 5e-3, f"Max error {max_error} too large"


def test_leapfrog_uses_requested_final_time():
    _, short_run = solve_wave_leapfrog(101, 400, 1.0, t_final=0.2)
    _, long_run = solve_wave_leapfrog(101, 400, 1.0, t_final=0.3)

    assert np.max(np.abs(short_run - long_run)) > 0.1


def test_leapfrog_uses_requested_wave_speed():
    _, slow_wave = solve_wave_leapfrog(101, 400, 0.5, t_final=0.3)
    _, fast_wave = solve_wave_leapfrog(101, 400, 1.0, t_final=0.3)

    assert np.max(np.abs(slow_wave - fast_wave)) > 0.1


def test_leapfrog_first_step_includes_half_acceleration():
    nx, c, length, t_final = 9, 0.8, 1.0, 0.1
    x, actual = solve_wave_leapfrog(nx, 1, c, length=length, t_final=t_final)
    initial = np.sin(np.pi * x / length)
    courant = c * t_final / (length / (nx - 1))
    expected = initial.copy()
    expected[1:-1] += 0.5 * courant**2 * (initial[2:] - 2.0 * initial[1:-1] + initial[:-2])
    expected[[0, -1]] = 0.0

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_leapfrog_matches_non_unit_domain_at_nonaliased_time():
    c, length, t_final = 0.8, 1.3, 0.27
    x, actual = solve_wave_leapfrog(161, 300, c, length=length, t_final=t_final)
    expected = wave_standing_trajectory(x, t_final, c, length=length)

    assert np.max(np.abs(actual - expected)) < 1e-4


def test_leapfrog_rejects_unstable_derived_courant_number():
    with pytest.raises(ValueError, match="Courant"):
        solve_wave_leapfrog(11, 1, 2.0, length=1.0, t_final=1.0)


@pytest.mark.parametrize(("nx", "nt"), [(2, 10), (10, 0)])
def test_leapfrog_rejects_invalid_discretization(nx: int, nt: int):
    with pytest.raises(ValueError, match=r"nx|nt"):
        solve_wave_leapfrog(nx, nt, 1.0)


@pytest.mark.parametrize("parameter", ["nx", "nt"])
def test_leapfrog_rejects_counts_outside_platform_index_range(parameter: str):
    huge_count = 10**10000
    nx = huge_count if parameter == "nx" else 3
    nt = huge_count if parameter == "nt" else 1

    with pytest.raises(ValueError, match=rf"{parameter} must fit within platform index range"):
        solve_wave_leapfrog(nx, nt, 1.0)


@pytest.mark.parametrize(("nx", "nt"), [(3.5, 10), (10, 2.5), (10, True)])
def test_leapfrog_rejects_noninteger_discretization(nx: object, nt: object):
    with pytest.raises(ValueError, match="integers"):
        solve_wave_leapfrog(cast("int", nx), cast("int", nt), 1.0)


@pytest.mark.parametrize(
    ("c", "length", "t_final"),
    [
        (0.0, 1.0, 0.1),
        (math.inf, 1.0, 0.1),
        (1.0, 0.0, 0.1),
        (1.0, math.nan, 0.1),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, math.inf),
    ],
)
def test_leapfrog_rejects_invalid_physical_parameters(c: float, length: float, t_final: float):
    with pytest.raises(ValueError, match="must be positive and finite"):
        solve_wave_leapfrog(101, 400, c, length=length, t_final=t_final)


@pytest.mark.parametrize(
    ("c", "length", "t_final", "courant", "parameter"),
    [
        pytest.param(10**10000, 1.0, 0.1, None, "c", id="wave-speed"),
        pytest.param(1.0, 10**10000, 0.1, None, "length", id="length"),
        pytest.param(1.0, 1.0, 10**10000, None, "t_final", id="final-time"),
        pytest.param(1.0, 1.0, 0.1, 10**10000, "courant", id="courant"),
    ],
)
def test_leapfrog_rejects_scalars_unrepresentable_as_float(
    c: object,
    length: object,
    t_final: object,
    courant: object,
    parameter: str,
):
    with pytest.raises(ValueError, match=rf"{parameter} must be representable as a float"):
        solve_wave_leapfrog(
            3,
            1,
            cast("float", c),
            length=cast("float", length),
            t_final=cast("float", t_final),
            courant=cast("float | None", courant),
        )


def test_leapfrog_rejects_courant_that_disagrees_with_inputs():
    with pytest.raises(ValueError, match="derived Courant"):
        solve_wave_leapfrog(11, 10, 1.0, t_final=0.5, courant=0.2)


@pytest.mark.parametrize("courant", [0.0, -2e-16])
def test_leapfrog_rejects_nonpositive_courant_when_derived_value_is_tiny(courant: float):
    with pytest.raises(ValueError, match="derived Courant"):
        solve_wave_leapfrog(3, 1, 1e-16, courant=courant)


def test_leapfrog_accepts_matching_explicit_courant_number():
    nx, nt, c, length, t_final = 41, 80, 0.7, 1.2, 0.25
    derived_courant = c * (t_final / nt) / (length / (nx - 1))
    x_implicit, u_implicit = solve_wave_leapfrog(nx, nt, c, length=length, t_final=t_final)
    x_explicit, u_explicit = solve_wave_leapfrog(
        nx,
        nt,
        c,
        length=length,
        t_final=t_final,
        courant=derived_courant,
    )

    np.testing.assert_array_equal(x_explicit, x_implicit)
    np.testing.assert_array_equal(u_explicit, u_implicit)


def test_leapfrog_returns_finite_float64_output_with_exact_boundaries():
    x, displacement = solve_wave_leapfrog(65, 80, 0.7, length=1.3, t_final=0.31)

    assert x.dtype == np.dtype(np.float64)
    assert displacement.dtype == np.dtype(np.float64)
    assert np.isfinite(x).all()
    assert np.isfinite(displacement).all()
    np.testing.assert_array_equal(displacement[[0, -1]], np.zeros(2))


@pytest.mark.parametrize("length", [np.float16(1.25), np.float32(1.25)])
def test_leapfrog_returns_float64_arrays_for_numpy_scalar_length(
    length: np.float16 | np.float32,
):
    x, displacement = solve_wave_leapfrog(17, 32, 0.5, length=length, t_final=0.1)

    assert x.dtype == np.dtype(np.float64)
    assert displacement.dtype == np.dtype(np.float64)


@pytest.mark.parametrize(
    ("c", "length", "t_final"),
    [
        pytest.param(np.float16(0.7), 1.3, 0.17, id="wave-speed"),
        pytest.param(0.7, np.float16(1.3), 0.17, id="length"),
        pytest.param(0.7, 1.3, np.float16(0.17), id="final-time"),
    ],
)
def test_leapfrog_canonicalizes_numpy_scalars_before_computation(
    c: float,
    length: float,
    t_final: float,
):
    canonical_c = float(c)
    canonical_length = float(length)
    canonical_t_final = float(t_final)
    expected_x = np.linspace(0.0, canonical_length, 17, dtype=np.float64)
    oracle_x, oracle_displacement = solve_wave_leapfrog(
        17,
        64,
        canonical_c,
        length=canonical_length,
        t_final=canonical_t_final,
    )

    x, displacement = solve_wave_leapfrog(
        17,
        64,
        c,
        length=length,
        t_final=t_final,
    )

    np.testing.assert_array_equal(oracle_x, expected_x)
    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(displacement, oracle_displacement)


def test_leapfrog_avoids_phase_overflow_for_maximum_finite_length():
    x, displacement = solve_wave_leapfrog(3, 1, 1.0, length=sys.float_info.max, t_final=1.0)

    assert np.isfinite(x).all()
    assert np.isfinite(displacement).all()
    assert displacement[1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("nx", "nt", "c", "length", "t_final", "invalid_quantity"),
    [
        pytest.param(3, 1, 1.0, math.ulp(0.0), 1.0, "dx", id="zero-dx"),
        pytest.param(3, 2, 1.0, 1.0, math.ulp(0.0), "dt", id="zero-dt"),
        pytest.param(
            3,
            1,
            math.ulp(0.0),
            1.0,
            math.ulp(0.0),
            "derived Courant number",
            id="zero-derived-courant",
        ),
        pytest.param(
            3,
            1,
            np.float64(math.ulp(0.0)),
            np.float64(math.ulp(0.0)),
            np.float64(math.ulp(0.0)),
            "dx",
            id="numpy-nan-courant",
        ),
    ],
)
def test_leapfrog_rejects_unrepresentable_derived_quantities(
    nx: int,
    nt: int,
    c: float,
    length: float,
    t_final: float,
    invalid_quantity: str,
):
    with pytest.raises(ValueError, match=rf"{invalid_quantity} must be positive and finite"):
        solve_wave_leapfrog(nx, nt, c, length=length, t_final=t_final)
