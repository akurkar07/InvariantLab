"""Agent-facing velocity Verlet solver for the 1-D harmonic oscillator.

Invocation (from the task root)::

    python src/solver.py --input input.json --output result.npz

The input JSON follows the shared task envelope: ``task_id``, ``parameters``
(``x0``, ``v0``, ``omega``, ``dt``, ``n_steps``), and ``numerics`` (``dtype``,
``seed``). The output archive contains exactly ``time`` with shape
``(n_steps + 1,)`` and ``state`` with shape ``(n_steps + 1, 2)`` holding
``[x, v]`` per row, both ``float64``.

This module is self-contained: it depends only on the standard library and NumPy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

TASK_ID = "oscillator_verlet"
DTYPE = "float64"
SEED = 42
REQUIRED_PARAMETERS = ("x0", "v0", "omega", "dt", "n_steps")


def solve(x0: float, v0: float, omega: float, dt: float, n_steps: int) -> np.ndarray:
    """Integrate d²x/dt² = -ω²x with velocity Verlet.

    Returns an ``(n_steps + 1, 2)`` float64 array of ``[x, v]`` rows, starting at
    the initial condition.
    """
    state = np.empty((n_steps + 1, 2), dtype=np.float64)
    state[0] = (x0, v0)
    omega2 = omega * omega
    x, v = float(x0), float(v0)
    for i in range(n_steps):
        v_half = v - 0.5 * dt * omega2 * x
        x = x + dt * v_half
        v = v_half - 0.5 * dt * omega2 * x
        state[i + 1] = (x, v)
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


def parse_input(payload: object) -> tuple[float, float, float, float, int]:
    """Validate the task envelope and return ``(x0, v0, omega, dt, n_steps)``."""
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

    return (
        _finite("x0", parameters["x0"]),
        _finite("v0", parameters["v0"]),
        _positive_finite("omega", parameters["omega"]),
        _positive_finite("dt", parameters["dt"]),
        n_steps,
    )


def run(input_path: Path, output_path: Path) -> None:
    """Read ``input_path``, solve once, and write ``output_path``."""
    with input_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    x0, v0, omega, dt, n_steps = parse_input(payload)
    state = solve(x0, v0, omega, dt, n_steps)
    time = dt * np.arange(n_steps + 1, dtype=np.float64)
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
