"""Analytical oracles for V1 problem families.

Each oracle computes the exact solution for a given initial condition and time,
independent of any numerical solver.  These are the ground-truth reference
implementations that the verifier compares against.
"""

import math

import numpy as np

# -- Harmonic Oscillator -------------------------------------------------------


def oscillator_analytic(
    t: float,
    x0: float,
    v0: float,
    omega: float,
) -> tuple[float, float]:
    """Exact solution for the 1-D harmonic oscillator.

    x(t) = x0 cos(omega t) + (v0/omega) sin(omega t)
    v(t) = -x0 omega sin(omega t) + v0 cos(omega t)

    Returns:
        (position, velocity) at time t.
    """
    cos_wt = math.cos(omega * t)
    sin_wt = math.sin(omega * t)
    x = x0 * cos_wt + (v0 / omega) * sin_wt
    v = -x0 * omega * sin_wt + v0 * cos_wt
    return x, v


def oscillator_energy(x: float, v: float, omega: float) -> float:
    """Total energy of a 1-D harmonic oscillator: E = 0.5 v^2 + 0.5 omega^2 x^2."""
    return 0.5 * v * v + 0.5 * omega * omega * x * x


def oscillator_trajectory(
    times: np.ndarray,
    x0: float,
    v0: float,
    omega: float,
) -> np.ndarray:
    """Compute the analytical oscillator trajectory over an array of times.

    Returns an (N, 3) array with columns [t, x, v].
    """
    cos_wt = np.cos(omega * times)
    sin_wt = np.sin(omega * times)
    x = x0 * cos_wt + (v0 / omega) * sin_wt
    v = -x0 * omega * sin_wt + v0 * cos_wt
    return np.column_stack([times, x, v])


# -- Kepler Two-Body ----------------------------------------------------------


def kepler_circular_velocity(mu: float, r: float) -> float:
    """Circular orbit velocity: v = sqrt(mu/r)."""
    return math.sqrt(mu / r)


def kepler_orbital_energy(vx: float, vy: float, rx: float, ry: float, mu: float) -> float:
    """Specific orbital energy: epsilon = v^2/2 - mu/r."""
    v2 = vx * vx + vy * vy
    r = math.sqrt(rx * rx + ry * ry)
    return 0.5 * v2 - mu / r


def kepler_angular_momentum(rx: float, ry: float, vx: float, vy: float) -> float:
    """Specific angular momentum (z-component of r cross v): h = rx*vy - ry*vx."""
    return rx * vy - ry * vx


def kepler_circular_orbit(
    t: float,
    mu: float,
    r0: float,
    phase: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    """Exact solution for a circular Kepler orbit.

    Returns (rx, ry, rz, vx, vy, vz) at time t.
    """
    omega = math.sqrt(mu / (r0 ** 3))  # mean motion
    theta = omega * t + phase
    rx = r0 * math.cos(theta)
    ry = r0 * math.sin(theta)
    rz = 0.0
    v = math.sqrt(mu / r0)
    vx = -v * math.sin(theta)
    vy = v * math.cos(theta)
    vz = 0.0
    return rx, ry, rz, vx, vy, vz


def kepler_eccentric_anomaly(
    mean_anomaly: float,
    e: float,
    tol: float = 1.0e-12,
    max_iter: int = 100,
) -> float:
    """Solve Kepler's equation M = E - e sin(E) for E via Newton-Raphson.

    Args:
        mean_anomaly: Mean anomaly (rad).
        e: Eccentricity.
        tol: Convergence tolerance.
        max_iter: Maximum Newton iterations.
    Returns:
        Eccentric anomaly E.
    """
    ea = mean_anomaly if e < 0.8 else math.pi  # initial guess
    for _ in range(max_iter):
        denom = 1.0 - e * math.cos(ea)
        delta_ea = (ea - e * math.sin(ea) - mean_anomaly) / denom
        ea -= delta_ea
        if abs(delta_ea) < tol:
            break
    return ea


def kepler_elliptic_orbit(
    t: float,
    mu: float,
    a: float,
    e: float,
    phase: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    """Exact solution for an elliptic Kepler orbit using Kepler's equation.

    Args:
        t: Time since periapsis passage.
        mu: Gravitational parameter.
        a: Semi-major axis.
        e: Eccentricity (0 <= e < 1).
        phase: Initial phase offset (rad).
    Returns:
        (rx, ry, rz, vx, vy, vz).
    """
    n = math.sqrt(mu / (a ** 3))  # mean motion
    ma = n * t + phase  # mean anomaly
    ea = kepler_eccentric_anomaly(ma, e)
    cos_ea = math.cos(ea)
    sin_ea = math.sin(ea)
    r = a * (1.0 - e * cos_ea)
    # Position in orbital plane
    x_orb = a * (cos_ea - e)
    y_orb = a * math.sqrt(1.0 - e * e) * sin_ea
    # Velocity in orbital plane
    factor = math.sqrt(mu * a) / r
    vx_orb = -factor * sin_ea
    vy_orb = factor * math.sqrt(1.0 - e * e) * cos_ea
    return x_orb, y_orb, 0.0, vx_orb, vy_orb, 0.0


# -- 1-D Heat Equation --------------------------------------------------------


def heat_manufactured_solution(
    x: float,
    t: float,
    alpha: float,
) -> float:
    """Manufactured solution: u(x,t) = sin(pi x) exp(-alpha pi^2 t).

    Satisfies u_t = alpha u_xx with u(0,t) = u(1,t) = 0.
    """
    return math.sin(math.pi * x) * math.exp(-alpha * math.pi * math.pi * t)


def heat_trajectory(
    x: np.ndarray,
    t: float,
    alpha: float,
) -> np.ndarray:
    """Compute the manufactured solution at a fixed time over a spatial grid."""
    decay_float = math.exp(-alpha * math.pi * math.pi * t)
    return np.sin(np.pi * x) * decay_float


def heat_decay_rate(alpha: float) -> float:
    """Decay rate lambda = alpha pi^2 for the manufactured solution."""
    return alpha * math.pi * math.pi


# -- 1-D Wave Equation --------------------------------------------------------


def wave_standing_solution(
    x: float,
    t: float,
    c: float,
    n_mode: int = 1,
    length: float = 1.0,
) -> float:
    """Standing-wave solution: u(x,t) = sin(n pi x/L) cos(n pi c t/L).

    Satisfies u_tt = c^2 u_xx with u(0,t) = u(L,t) = 0.
    """
    k = n_mode * math.pi / length
    return math.sin(k * x) * math.cos(k * c * t)


def wave_standing_velocity(
    x: float,
    t: float,
    c: float,
    n_mode: int = 1,
    length: float = 1.0,
) -> float:
    """Time derivative of the standing-wave solution (velocity field)."""
    k = n_mode * math.pi / length
    return -k * c * math.sin(k * x) * math.sin(k * c * t)


def wave_standing_trajectory(
    x: np.ndarray,
    t: float,
    c: float,
    n_mode: int = 1,
    length: float = 1.0,
) -> np.ndarray:
    """Standing-wave displacement at a fixed time over a spatial grid."""
    k = n_mode * math.pi / length
    cos_kc = math.cos(k * c * t)
    return np.sin(k * x) * cos_kc


def wave_travelling_solution(
    x: float,
    t: float,
    c: float,
    n_mode: int = 1,
    length: float = 1.0,
) -> float:
    """Travelling-wave solution (right-moving): u(x,t) = sin(n pi (x-ct)/L)."""
    k = n_mode * math.pi / length
    return math.sin(k * (x - c * t))
