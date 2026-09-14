"""Agent-facing velocity Verlet solver for the planar Kepler problem.

Invocation (from the task root)::

    python src/solver.py --input input.json --output result.npz

The input JSON follows the shared task envelope: ``task_id``, ``parameters``
(``rx``, ``ry``, ``vx``, ``vy``, ``mu``, ``dt``, ``n_steps``), and ``numerics``
(``dtype``, ``seed``). The output archive contains exactly ``time`` with shape
``(n_steps + 1,)`` and ``state`` with shape ``(n_steps + 1, 4)`` holding
``[rx, ry, vx, vy]`` per row, both ``float64``.

This module is self-contained: it depends only on the standard library and NumPy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

TASK_ID = "kepler_verlet"
DTYPE = "float64"
SEED = 1729
REQUIRED_PARAMETERS = ("rx", "ry", "vx", "vy", "mu", "dt", "n_steps")


def _acceleration(rx: float, ry: float, mu: float) -> tuple[float, float]:
    radius = math.hypot(rx, ry)
    if radius == 0.0:
        raise ValueError("Kepler trajectory reached zero radius")
    factor = -mu / radius**3
    return factor * rx, factor * ry


def solve(
    rx: float, ry: float, vx: float, vy: float, mu: float, dt: float, n_steps: int
) -> np.ndarray:
    """Integrate the planar Kepler problem with velocity Verlet.

    Returns an ``(n_steps + 1, 4)`` float64 array of ``[rx, ry, vx, vy]`` rows,
    starting at the initial state.
    """
    state = np.empty((n_steps + 1, 4), dtype=np.float64)
    state[0] = (rx, ry, vx, vy)
    for i in range(n_steps):
        ax, ay = _acceleration(rx, ry, mu)
        vx_half = vx + 0.5 * dt * ax
        vy_half = vy + 0.5 * dt * ay
        rx = rx + dt * vx_half
        ry = ry + dt * vy_half
        ax_new, ay_new = _acceleration(rx, ry, mu)
        vx = vx_half + 0.5 * dt * ax_new
        vy = vy_half + 0.5 * dt * ay_new
        state[i + 1] = (rx, ry, vx, vy)
    return state


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"parameters.{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"parameters.{name} must be positive and finite; got {value}")
    return number


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"parameters.{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"parameters.{name} must be finite; got {value}")
    return number


def parse_input(payload: object) -> tuple[float, float, float, float, float, float, int]:
    """Validate the task envelope and return Kepler's initial-value parameters."""
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object")
    if payload.get("task_id") != TASK_ID:
        raise ValueError(f"task_id must be {TASK_ID!r}; got {payload.get('task_id')!r}")

    numerics = payload.get("numerics")
    if not isinstance(numerics, dict):
        raise ValueError("numerics must be a JSON object")
    if numerics.get("dtype") != DTYPE:
        raise ValueError(f"numerics.dtype must be {DTYPE!r}; got {numerics.get('dtype')!r}")
    if numerics.get("seed") != SEED:
        raise ValueError(f"numerics.seed must be {SEED}; got {numerics.get('seed')!r}")

    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be a JSON object")
    missing = [name for name in REQUIRED_PARAMETERS if name not in parameters]
    if missing:
        raise ValueError(f"parameters missing required fields: {missing}")
    unknown = sorted(set(parameters) - set(REQUIRED_PARAMETERS))
    if unknown:
        raise ValueError(f"parameters has unknown fields: {unknown}")

    n_steps = parameters["n_steps"]
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError(f"parameters.n_steps must be a positive integer; got {n_steps!r}")

    rx = _finite("rx", parameters["rx"])
    ry = _finite("ry", parameters["ry"])
    if math.hypot(rx, ry) == 0.0:
        raise ValueError("parameters must specify a non-zero initial radius")
    return (
        rx,
        ry,
        _finite("vx", parameters["vx"]),
        _finite("vy", parameters["vy"]),
        _positive_finite("mu", parameters["mu"]),
        _positive_finite("dt", parameters["dt"]),
        n_steps,
    )


def run(input_path: Path, output_path: Path) -> None:
    """Read ``input_path``, solve once, and write ``output_path``."""
    with input_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rx, ry, vx, vy, mu, dt, n_steps = parse_input(payload)
    state = solve(rx, ry, vx, vy, mu, dt, n_steps)
    time = dt * np.arange(n_steps + 1, dtype=np.float64)
    if not np.isfinite(time).all() or not np.isfinite(state).all():
        raise ValueError("Kepler solve produced non-finite output")
    np.savez_compressed(output_path, time=time, state=state)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        run(args.input, args.output)
    except (OSError, ValueError) as error:
        print(f"solver error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
