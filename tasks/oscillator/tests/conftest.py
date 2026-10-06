"""Shared helpers for the oscillator task tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = TASK_ROOT / "src" / "solver.py"
TASK_ID = "oscillator_verlet"

if str(ENTRYPOINT.parent) not in sys.path:
    sys.path.insert(0, str(ENTRYPOINT.parent))


def make_input(x0: float, v0: float, omega: float, dt: float, n_steps: int) -> dict[str, object]:
    return {
        "task_id": TASK_ID,
        "parameters": {"x0": x0, "v0": v0, "omega": omega, "dt": dt, "n_steps": n_steps},
        "numerics": {"dtype": "float64", "seed": 42},
    }


def write_input(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
