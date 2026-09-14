"""Independent high-accuracy DOP853 oracle for planar Kepler trajectories.

This trusted oracle deliberately owns its gravitational right-hand side rather
than sharing the Velocity Verlet update used by the ordinary reference solver.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from collections.abc import Sequence

RTOL = 1.0e-11
ATOL = 1.0e-13


def _finite_float(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite; got {value}")
    return number


def solve_kepler_high_accuracy(
    rx: float,
    ry: float,
    vx: float,
    vy: float,
    mu: float,
    t_eval: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Integrate a planar Kepler orbit at requested times with DOP853.

    Args:
        rx, ry, vx, vy: Initial state at time zero.
        mu: Positive finite gravitational parameter.
        t_eval: At least two finite, strictly increasing times beginning at zero.

    Returns:
        An ``(len(t_eval), 4)`` float64 array with columns ``[rx, ry, vx, vy]``.

    Raises:
        ValueError: If inputs do not define a valid finite initial-value problem.
        RuntimeError: If DOP853 does not return every requested, finite state.
    """
    initial_state = np.array(
        [
            _finite_float("rx", rx),
            _finite_float("ry", ry),
            _finite_float("vx", vx),
            _finite_float("vy", vy),
        ],
        dtype=np.float64,
    )
    mu = _finite_float("mu", mu)
    if mu <= 0.0:
        raise ValueError(f"mu must be positive; got {mu}")
    if math.hypot(initial_state[0], initial_state[1]) == 0.0:
        raise ValueError("initial radius must be non-zero")

    times = np.asarray(t_eval, dtype=np.float64)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("t_eval must be a one-dimensional array with at least two times")
    if not np.isfinite(times).all():
        raise ValueError("t_eval must contain only finite times")
    if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("t_eval must begin at zero and be strictly increasing")

    def rhs(_: float, state: np.ndarray) -> np.ndarray:
        position_x, position_y, velocity_x, velocity_y = state
        radius = math.hypot(position_x, position_y)
        if radius == 0.0:
            raise RuntimeError("Kepler integration reached zero radius")
        scale = -mu / radius**3
        return np.array(
            [velocity_x, velocity_y, scale * position_x, scale * position_y], dtype=np.float64
        )

    result = solve_ivp(
        rhs,
        (0.0, float(times[-1])),
        initial_state,
        method="DOP853",
        t_eval=times,
        rtol=RTOL,
        atol=ATOL,
    )
    if not result.success:
        raise RuntimeError(f"DOP853 Kepler integration failed: {result.message}")
    if result.t.shape != times.shape or result.y.shape != (4, times.size):
        raise RuntimeError("DOP853 Kepler integration returned an incomplete trajectory")
    if not np.array_equal(result.t, times):
        raise RuntimeError("DOP853 Kepler integration did not return the requested times")

    states = np.ascontiguousarray(result.y.T, dtype=np.float64)
    if not np.isfinite(states).all():
        raise RuntimeError("DOP853 Kepler integration returned non-finite values")
    return states
