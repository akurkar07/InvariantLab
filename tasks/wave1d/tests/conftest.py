"""Shared helpers for the wave-equation task tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = TASK_ROOT / "src" / "solver.py"
TASK_ID = "wave_leapfrog"

if str(ENTRYPOINT.parent) not in sys.path:
    sys.path.insert(0, str(ENTRYPOINT.parent))


def make_input(
    nx: int,
    nt: int,
    c: float,
    length: float,
    t_final: float,
    courant: float | None = None,
) -> dict[str, object]:
    parameters: dict[str, object] = {
        "nx": nx,
        "nt": nt,
        "c": c,
        "length": length,
        "t_final": t_final,
    }
    if courant is not None:
        parameters["courant"] = courant
    return {
        "task_id": TASK_ID,
        "parameters": parameters,
        "numerics": {"dtype": "float64", "seed": 42},
    }


def write_input(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
