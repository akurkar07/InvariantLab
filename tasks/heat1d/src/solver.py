"""Agent-facing FTCS solver for the 1-D heat equation.

Invocation (from the task root)::

    python src/solver.py --input input.json --output result.npz

The input JSON follows the shared task envelope: ``task_id``, ``parameters``
(``nx``, ``nt``, ``alpha``, ``length``, ``t_final``), and ``numerics``
(``dtype``, ``seed``). The output archive contains exactly ``x`` and ``state``,
both one-dimensional ``float64`` arrays of length ``nx``. The initial condition
is ``sin(pi*x/length)`` with exact zero Dirichlet boundaries.

This module is self-contained: it depends only on the standard library and NumPy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

TASK_ID = "heat_ftcs"
DTYPE = "float64"
SEED = 42
REQUIRED_PARAMETERS = ("nx", "nt", "alpha", "length", "t_final")


def solve(
    nx: int, nt: int, alpha: float, length: float, t_final: float
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate u_t = alpha*u_xx with stable FTCS and zero Dirichlet boundaries."""
    dx = length / (nx - 1)
    dt = t_final / nt
    ratio = alpha * dt / dx**2
    if ratio > 0.5:
        raise ValueError(f"FTCS stability requires r <= 0.5; got {ratio}")

    x = np.linspace(0.0, length, nx, dtype=np.float64)
    state = np.sin(np.pi * x / length)
    state[[0, -1]] = 0.0
    for _ in range(nt):
        next_state = state.copy()
        next_state[1:-1] = state[1:-1] + ratio * (state[2:] - 2.0 * state[1:-1] + state[:-2])
        next_state[0] = 0.0
        next_state[-1] = 0.0
        state = next_state
    return x, state


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"parameters.{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"parameters.{name} must be positive and finite; got {value}")
    return number


def parse_input(payload: object) -> tuple[int, int, float, float, float]:
    """Validate the task envelope and return FTCS parameters."""
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

    nx = parameters["nx"]
    nt = parameters["nt"]
    if isinstance(nx, bool) or not isinstance(nx, int) or nx < 3:
        raise ValueError(f"parameters.nx must be an integer of at least 3; got {nx!r}")
    if isinstance(nt, bool) or not isinstance(nt, int) or nt < 1:
        raise ValueError(f"parameters.nt must be a positive integer; got {nt!r}")

    alpha = _positive_finite("alpha", parameters["alpha"])
    length = _positive_finite("length", parameters["length"])
    t_final = _positive_finite("t_final", parameters["t_final"])
    ratio = alpha * (t_final / nt) / (length / (nx - 1)) ** 2
    if ratio > 0.5:
        raise ValueError(f"FTCS stability requires r <= 0.5; got {ratio}")
    return nx, nt, alpha, length, t_final


def run(input_path: Path, output_path: Path) -> None:
    """Read ``input_path``, solve once, and write ``output_path``."""
    with input_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    x, state = solve(*parse_input(payload))
    if not np.isfinite(x).all() or not np.isfinite(state).all():
        raise ValueError("FTCS solve produced non-finite output")
    np.savez_compressed(output_path, x=x, state=state)


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
