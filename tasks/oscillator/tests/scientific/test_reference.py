"""Scientific hidden tests exercised only through the documented NPZ boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import yaml
from conftest import TASK_ROOT, make_input

from invariantlab.verification.analytical import oscillator_trajectory
from invariantlab.verification.execution import run_task

if TYPE_CHECKING:
    from pathlib import Path

X0 = 0.7
V0 = -0.35
OMEGA = 1.7
DT = 1e-3
N_STEPS = 15_000
CONTRACT = yaml.safe_load((TASK_ROOT / "contract.yaml").read_text(encoding="utf-8"))
STATE_RELATIVE_L2 = CONTRACT["numerics"]["tolerances"]["state_relative_l2"]
ENERGY_RELATIVE_DRIFT = CONTRACT["numerics"]["tolerances"]["energy_relative_drift"]


def test_state_matches_analytical_solution(tmp_path: Path) -> None:
    run = run_task(
        TASK_ROOT,
        TASK_ROOT,
        make_input(X0, V0, OMEGA, DT, N_STEPS)["parameters"],
        tmp_path,
    )
    assert run.passed, (run.gates, run.stderr)
    assert run.arrays is not None
    time, state = run.arrays["time"], run.arrays["state"]
    assert time.shape == (N_STEPS + 1,)
    assert state.shape == (N_STEPS + 1, 2)
    exact = oscillator_trajectory(time, X0, V0, OMEGA)[:, 1:]
    relative_l2 = np.linalg.norm(state - exact) / np.linalg.norm(exact)
    assert relative_l2 < STATE_RELATIVE_L2, f"state_relative_l2={relative_l2:.3e}"


def test_energy_drift_is_bounded_over_horizon(tmp_path: Path) -> None:
    run = run_task(
        TASK_ROOT,
        TASK_ROOT,
        make_input(X0, V0, OMEGA, DT, N_STEPS)["parameters"],
        tmp_path,
    )
    assert run.passed, (run.gates, run.stderr)
    assert run.arrays is not None
    state = run.arrays["state"]
    assert state.shape == (N_STEPS + 1, 2)
    energy = 0.5 * state[:, 1] ** 2 + 0.5 * OMEGA**2 * state[:, 0] ** 2
    relative_drift = np.max(np.abs(energy - energy[0])) / energy[0]
    assert relative_drift < ENERGY_RELATIVE_DRIFT, f"energy_relative_drift={relative_drift:.3e}"
