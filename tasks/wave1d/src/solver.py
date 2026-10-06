"""Agent-facing leapfrog solver for the 1-D wave equation.

Invocation (from the task root)::

    python src/solver.py --input input.json --output result.npz

The input JSON follows the shared task envelope: ``task_id``, ``parameters``
(``nx``, ``nt``, ``c``, ``length``, ``t_final``, and optional ``courant``), and
``numerics`` (``dtype``, ``seed``). The output archive contains exactly ``x``
and ``state``, both float64 vectors of shape ``(nx,)``. ``state`` is the final
displacement after ``nt`` leapfrog steps.

This module is self-contained: it depends only on the standard library and NumPy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

TASK_ID = "wave_leapfrog"
DTYPE = "float64"
SEED = 42
REQUIRED_PARAMETERS = ("nx", "nt", "c", "length", "t_final")
OPTIONAL_PARAMETERS = ("courant",)


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


def _validate_parameters(
    nx: object,
    nt: object,
    c: object,
    length: object,
    t_final: object,
    courant: object | None = None,
) -> tuple[int, int, float, float, float, float]:
    if isinstance(nx, bool) or not isinstance(nx, int) or nx < 3:
        raise ValueError(f"parameters.nx must be an integer at least 3; got {nx!r}")
    if isinstance(nt, bool) or not isinstance(nt, int) or nt < 1:
        raise ValueError(f"parameters.nt must be a positive integer; got {nt!r}")

    wave_speed = _positive_finite("c", c)
    domain_length = _positive_finite("length", length)
    final_time = _positive_finite("t_final", t_final)
    dx = domain_length / (nx - 1)
    dt = final_time / nt
    derived_courant = wave_speed * dt / dx
    if not math.isfinite(derived_courant) or derived_courant <= 0.0:
        raise ValueError(
            f"derived Courant number must be positive and finite; got {derived_courant}"
        )
    if abs(derived_courant) > 1.0:
        raise ValueError(
            f"Courant number must satisfy |c*dt/dx| <= 1; got {derived_courant}"
        )
    if courant is not None:
        declared_courant = _finite("courant", courant)
        if not math.isclose(declared_courant, derived_courant, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(
                "parameters.courant must match the derived Courant number "
                f"{derived_courant}; got {declared_courant}"
            )
    return nx, nt, wave_speed, domain_length, final_time, derived_courant


def solve(
    nx: int,
    nt: int,
    c: float,
    length: float,
    t_final: float,
    courant: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve u_tt = c²u_xx using the corrected zero-velocity leapfrog scheme."""
    nx, nt, c, length, _t_final, courant = _validate_parameters(
        nx, nt, c, length, t_final, courant
    )
    x = np.linspace(0.0, length, nx, dtype=np.float64)
    state = np.sin(np.pi * x / length)
    state[[0, -1]] = 0.0

    # u(-dt) = u(0) + 0.5 * C² * delta2(u(0)) for zero initial velocity.
    previous = state.copy()
    previous[1:-1] += 0.5 * courant**2 * (
        state[2:] - 2.0 * state[1:-1] + state[:-2]
    )
    previous[[0, -1]] = 0.0

    for _ in range(nt):
        next_state = np.zeros(nx, dtype=np.float64)
        next_state[1:-1] = (
            2.0 * state[1:-1]
            - previous[1:-1]
            + courant**2 * (state[2:] - 2.0 * state[1:-1] + state[:-2])
        )
        next_state[[0, -1]] = 0.0
        previous, state = state, next_state

    return x, state


def parse_input(payload: object) -> tuple[int, int, float, float, float, float]:
    """Validate the shared task envelope and return the wave parameters."""
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
    allowed = set(REQUIRED_PARAMETERS) | set(OPTIONAL_PARAMETERS)
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f"parameters has unknown fields: {unknown}")

    return _validate_parameters(
        parameters["nx"],
        parameters["nt"],
        parameters["c"],
        parameters["length"],
        parameters["t_final"],
        parameters.get("courant"),
    )


def run(input_path: Path, output_path: Path) -> None:
    """Read ``input_path``, solve once, and write ``output_path``."""
    with input_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    nx, nt, c, length, t_final, courant = parse_input(payload)
    x, state = solve(nx, nt, c, length, t_final, courant)
    if not np.isfinite(x).all() or not np.isfinite(state).all():
        raise ValueError("wave solve produced non-finite output")
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
