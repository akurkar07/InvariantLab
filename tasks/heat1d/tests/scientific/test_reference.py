"""Scientific hidden tests exercised only through the documented NPZ boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import yaml
from conftest import TASK_ROOT, make_input

from invariantlab.verification.execution import run_task

from invariantlab.verification.analytical import heat_trajectory

if TYPE_CHECKING:
    from pathlib import Path

ALPHA, LENGTH, NX, NT, T_FINAL = 0.17, 1.3, 161, 1_800, 0.237
CONTRACT = yaml.safe_load((TASK_ROOT / "contract.yaml").read_text(encoding="utf-8"))
STATE_RELATIVE_L2 = CONTRACT["numerics"]["tolerances"]["state_relative_l2"]


EXPECTED_ARCHIVE_NAMES = {array["name"] for array in CONTRACT["output"]["arrays"]}


def _candidate_solution(tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    run = run_task(
        TASK_ROOT,
        TASK_ROOT,
        make_input(NX, NT, ALPHA, LENGTH, T_FINAL)["parameters"],
        tmp_path,
    )
    assert run.passed, (run.gates, run.stderr)
    assert run.arrays is not None
    assert set(run.arrays) == EXPECTED_ARCHIVE_NAMES
    x, state = run.arrays["x"], run.arrays["state"]
    assert x.shape == (NX,)
    assert state.shape == (NX,)
    assert x.dtype == np.dtype(np.float64)
    assert state.dtype == np.dtype(np.float64)
    assert np.isfinite(x).all()
    assert np.isfinite(state).all()
    return x, state


def test_non_special_ftcs_case_matches_manufactured_solution_and_decays(tmp_path: Path) -> None:
    x, state = _candidate_solution(tmp_path)
    expected = heat_trajectory(x, T_FINAL, ALPHA, length=LENGTH)
    expected[[0, -1]] = 0.0
    initial = heat_trajectory(x, 0.0, ALPHA, length=LENGTH)
    initial[[0, -1]] = 0.0
    state_relative_l2 = np.linalg.norm(state - expected) / np.linalg.norm(expected)
    assert state_relative_l2 < STATE_RELATIVE_L2, f"state_relative_l2={state_relative_l2:.3e}"
    np.testing.assert_array_equal(state[[0, -1]], np.zeros(2))
    assert np.linalg.norm(state) < np.linalg.norm(initial)


def test_scientific_suite_rejects_unstable_ftcs_ratio_at_process_boundary(tmp_path: Path) -> None:
    run = run_task(
        TASK_ROOT,
        TASK_ROOT,
        make_input(11, 1, 1.0, 1.0, 1.0)["parameters"],
        tmp_path,
    )
    assert not run.passed
    assert run.gates[0].name == "execution" and not run.gates[0].passed
    assert run.returncode not in (0, None)
    assert "FTCS stability" in run.stderr
    assert not (tmp_path / "result.npz").exists()
