"""Shared helpers for the heat task tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = TASK_ROOT / "src" / "solver.py"
TASK_ID = "heat_ftcs"

if str(ENTRYPOINT.parent) not in sys.path:
    sys.path.insert(0, str(ENTRYPOINT.parent))


def make_input(nx: int, nt: int, alpha: float, length: float, t_final: float) -> dict[str, object]:
    return {
        "task_id": TASK_ID,
        "parameters": {
            "nx": nx,
            "nt": nt,
            "alpha": alpha,
            "length": length,
            "t_final": t_final,
        },
        "numerics": {"dtype": "float64", "seed": 42},
    }


def write_input(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
