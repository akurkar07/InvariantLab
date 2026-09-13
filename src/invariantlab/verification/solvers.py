"""Reference numerical solvers for V1 problem families.

Each solver is a straightforward, well-tested numerical method chosen for
stability and simplicity.  These are the *reference* implementations: they
should pass all verification layers by construction.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from invariantlab.verification.analytical import (
    kepler_angular_momentum,
    kepler_orbital_energy,
)

# -- Harmonic Oscillator - Velocity Verlet ------------------------------------


def solve_oscillator_verlet(
    x0: float,
    v0: float,
    omega: float,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """Solve the harmonic oscillator using the velocity Verlet method.

    Args:
        x0: Initial position.
        v0: Initial velocity.
        omega: Angular frequency.
        dt: Time step.
        n_steps: Number of integration steps.
    Returns:
        (n_steps+1, 3) array with columns [t, x, v].
    """
    trajectory = np.zeros((n_steps + 1, 3))
    x, v, t = x0, v0, 0.0
    trajectory[0] = [t, x, v]

    omega2 = omega * omega

    for i in range(n_steps):
        # Half-step velocity
        a = -omega2 * x
        v_half = v + 0.5 * dt * a
        # Full-step position
        x = x + dt * v_half
        # New acceleration
        a_new = -omega2 * x
        # Full-step velocity
        v = v_half + 0.5 * dt * a_new
        t += dt
        trajectory[i + 1] = [t, x, v]

    return trajectory


# -- Kepler Two-Body - Velocity Verlet (symplectic) --------------------------


def solve_kepler_verlet(
    rx: float,
    ry: float,
    vx: float,
    vy: float,
    mu: float,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """Solve the Kepler two-body problem using velocity Verlet.

    Args:
        rx, ry: Initial position.
        vx, vy: Initial velocity.
        mu: Gravitational parameter (GM).
        dt: Time step.
        n_steps: Number of integration steps.
    Returns:
        (n_steps+1, 7) array with columns [t, rx, ry, vx, vy, energy, ang_mom].
    """
    trajectory = np.zeros((n_steps + 1, 7))
    t = 0.0
    energy = kepler_orbital_energy(vx, vy, rx, ry, mu)
    h = kepler_angular_momentum(rx, ry, vx, vy)
    trajectory[0] = [t, rx, ry, vx, vy, energy, h]

    for i in range(n_steps):
        r2 = rx * rx + ry * ry
        r = math.sqrt(r2)
        factor = -mu / (r2 * r)
        ax = factor * rx
        ay = factor * ry

        # Half-step velocity
        vx_half = vx + 0.5 * dt * ax
        vy_half = vy + 0.5 * dt * ay
        # Full-step position
        rx = rx + dt * vx_half
        ry = ry + dt * vy_half
        # New acceleration
        r2_new = rx * rx + ry * ry
        r_new = math.sqrt(r2_new)
        factor_new = -mu / (r2_new * r_new)
        ax_new = factor_new * rx
        ay_new = factor_new * ry
        # Full-step velocity
        vx = vx_half + 0.5 * dt * ax_new
        vy = vy_half + 0.5 * dt * ay_new
        t += dt

        energy = kepler_orbital_energy(vx, vy, rx, ry, mu)
        h = kepler_angular_momentum(rx, ry, vx, vy)
        trajectory[i + 1] = [t, rx, ry, vx, vy, energy, h]

    return trajectory


# -- 1-D Heat Equation - Forward Euler (FTCS) ---------------------------------


def solve_heat_ftcs(
    nx: int,
    nt: int,
    alpha: float,
    length: float = 1.0,
    t_final: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the 1-D heat equation u_t = alpha u_xx using FTCS.

    Args:
        nx: Number of spatial grid points (including boundaries).
        nt: Number of time steps.
        alpha: Thermal diffusivity.
        length: Domain length.
        t_final: Final time.
    Returns:
        (x_grid, u) -- spatial grid and final temperature distribution.
    """
    dx = length / (nx - 1)
    dt = t_final / nt
    r = alpha * dt / (dx * dx)

    if r > 0.5:
        raise ValueError(
            f"FTCS stability violated: r={r:.4f} > 0.5. "
            f"Reduce dt or increase dx."
        )

    x = np.linspace(0, length, nx)
    u = np.sin(np.pi * x)  # initial condition matching manufactured solution

    for _ in range(nt):
        u_new = u.copy()
        for i in range(1, nx - 1):
            u_new[i] = u[i] + r * (u[i + 1] - 2 * u[i] + u[i - 1])
        # Dirichlet BCs: u(0) = u(L) = 0
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u = u_new

    return x, u


def solve_heat_crank_nicolson(
    nx: int,
    nt: int,
    alpha: float,
    length: float = 1.0,
    t_final: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the 1-D heat equation u_t = alpha u_xx using Crank-Nicolson.

    Unconditionally stable.  Uses Thomas algorithm for tridiagonal solve.

    Args:
        nx: Number of spatial grid points.
        nt: Number of time steps.
        alpha: Thermal diffusivity.
        length: Domain length.
        t_final: Final time.
    Returns:
        (x_grid, u) -- spatial grid and final temperature distribution.
    """
    dx = length / (nx - 1)
    dt = t_final / nt
    r = alpha * dt / (dx * dx)

    x = np.linspace(0, length, nx)
    u = np.sin(np.pi * x)

    # Tridiagonal coefficients
    n_inner = nx - 2  # interior points
    lower = np.full(n_inner - 1, -0.5 * r)
    main = np.full(n_inner, 1.0 + r)
    upper = np.full(n_inner - 1, -0.5 * r)

    for _ in range(nt):
        # RHS: explicit part
        rhs = np.zeros(n_inner)
        for i in range(n_inner):
            idx = i + 1
            rhs[i] = (
                0.5 * r * (u[idx - 1] + u[idx + 1])
                + (1.0 - r) * u[idx]
            )
        # Apply BCs to RHS
        rhs[0] += 0.5 * r * u[0]  # u[0] = 0 so no effect
        rhs[-1] += 0.5 * r * u[-1]  # u[-1] = 0 so no effect

        # Thomas algorithm
        u_inner = thomas_solve(lower, main, upper, rhs)
        u[1:-1] = u_inner
        u[0] = 0.0
        u[-1] = 0.0

    return x, u


def thomas_solve(
    lower: np.ndarray,
    main: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal system using the Thomas algorithm.

    Args:
        lower: Sub-diagonal (length n-1).
        main: Main diagonal (length n).
        upper: Super-diagonal (length n-1).
        rhs: Right-hand side (length n).
    Returns:
        Solution vector.
    """
    n = len(main)
    c = np.zeros(n - 1)
    d = np.zeros(n)
    x = np.zeros(n)

    # Forward sweep
    c[0] = upper[0] / main[0]
    d[0] = rhs[0] / main[0]
    for i in range(1, n - 1):
        denom = main[i] - lower[i - 1] * c[i - 1]
        c[i] = upper[i] / denom
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / denom
    d[n - 1] = (rhs[n - 1] - lower[n - 2] * d[n - 2]) / (
        main[n - 1] - lower[n - 2] * c[n - 2]
    )

    # Back substitution
    x[n - 1] = d[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]

    return x


# -- 1-D Wave Equation - Leapfrog ---------------------------------------------


def solve_wave_leapfrog(
    nx: int,
    nt: int,
    c: float,
    length: float = 1.0,
    t_final: float = 0.1,
    courant: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the 1-D wave equation u_tt = c^2 u_xx using leapfrog.

    Args:
        nx: Number of spatial grid points.
        nt: Number of time steps.
        c: Wave speed.
        length: Domain length.
        t_final: Final time.
        courant: Optional consistency check for the Courant number derived from
            ``c``, ``length``, ``nx``, ``t_final``, and ``nt``.
    Returns:
        (x_grid, u) -- spatial grid and final displacement.
    Raises:
        ValueError: If grid counts are invalid or exceed the platform index
            range, physical parameters are invalid, the derived Courant number
            is unstable, or ``courant`` disagrees with the derived value.
    """
    if (
        not isinstance(nx, int)
        or isinstance(nx, bool)
        or not isinstance(nt, int)
        or isinstance(nt, bool)
    ):
        raise ValueError(f"nx and nt must be integers; got nx={nx}, nt={nt}")
    for count_name, count in (("nx", nx), ("nt", nt)):
        if count > sys.maxsize:
            raise ValueError(f"{count_name} must fit within platform index range")
    if nx < 3:
        raise ValueError(f"nx must be at least 3; got {nx}")
    if nt < 1:
        raise ValueError(f"nt must be at least 1; got {nt}")

    def as_float(name: str, value: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be representable as a float") from exc

    c = as_float("c", c)
    length = as_float("length", length)
    t_final = as_float("t_final", t_final)
    if courant is not None:
        courant = as_float("courant", courant)

    for name, value in (("c", c), ("length", length), ("t_final", t_final)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite; got {value}")

    dx = length / (nx - 1)
    dt = t_final / nt
    for name, value in (("dx", dx), ("dt", dt)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite; got {value}")

    derived_courant = c * dt / dx
    if not math.isfinite(derived_courant) or derived_courant <= 0.0:
        raise ValueError(
            f"derived Courant number must be positive and finite; got {derived_courant}"
        )
    if abs(derived_courant) > 1.0:
        raise ValueError(f"Courant number must satisfy |c*dt/dx| <= 1; got {derived_courant}")
    if courant is not None and not math.isclose(
        courant, derived_courant, rel_tol=1e-12, abs_tol=0.0
    ):
        raise ValueError(
            f"courant must match the derived Courant number {derived_courant}; got {courant}"
        )
    courant = derived_courant

    x = np.linspace(0, length, nx, dtype=np.float64)
    u = np.sin(np.pi * (x / length))  # fundamental standing-wave mode
    u[[0, -1]] = 0.0
    u_prev = u.copy()
    # Zero initial velocity gives u(-dt) = u(0) + 0.5 dt^2 u_tt(0).
    u_prev[1:-1] += 0.5 * courant**2 * (u[2:] - 2.0 * u[1:-1] + u[:-2])

    for _ in range(nt):
        u_new = np.zeros(nx)
        for i in range(1, nx - 1):
            u_new[i] = 2 * u[i] - u_prev[i] + courant**2 * (u[i + 1] - 2 * u[i] + u[i - 1])
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u_prev = u.copy()
        u = u_new

    return x, u
