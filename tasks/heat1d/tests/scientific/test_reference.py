"""Scientific (hidden) tests for the heat FTCS task package.

The test uses a non-unit domain and evaluates the manufactured mode locally on
the trusted-test side because the existing helper is defined specifically for
unit length.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import solver
from conftest import make_input, write_input

if TYPE_CHECKING:
    from pathlib import Path

ALPHA = 0.17
LENGTH = 1.3
NX = 161
NT = 1_800
T_FINAL = 0.237

# FTCS is first order in dt and second order in dx. For this non-special
# alpha/length/grid/time case, dx=0.008125 and dt=1.316...e-4 yield r≈0.339.
# The mode's predicted finite-grid relative error is about 7.8e-6, so the
# contract's 1e-5 limit leaves a narrow discretisation margin without accepting
# an unstable or incorrectly scaled update.
STATE_RELATIVE_L2 = 1.0e-5


def _manufactured_solution(x: np.ndarray, alpha: float, length: float, time: float) -> np.ndarray:
    """Exact sin(pi*x/L) manufactured mode at a fixed time."""
    return np.sin(np.pi * x / length) * np.exp(-alpha * (np.pi / length) ** 2 * time)


def _candidate_solution(tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    input_path = write_input(tmp_path / "input.json", make_input(NX, NT, ALPHA, LENGTH, T_FINAL))
    output_path = tmp_path / "result.npz"
    solver.run(input_path, output_path)
    with np.load(output_path) as archive:
        return archive["x"], archive["state"]


def test_non_special_ftcs_case_matches_manufactured_solution_and_decays(tmp_path: Path) -> None:
    x, state = _candidate_solution(tmp_path)
    expected = _manufactured_solution(x, ALPHA, LENGTH, T_FINAL)
    expected[[0, -1]] = 0.0
    initial = _manufactured_solution(x, ALPHA, LENGTH, 0.0)
    initial[[0, -1]] = 0.0

    state_relative_l2 = np.linalg.norm(state - expected) / np.linalg.norm(expected)

    assert state_relative_l2 < STATE_RELATIVE_L2, f"state_relative_l2={state_relative_l2:.3e}"
    np.testing.assert_array_equal(state[[0, -1]], np.zeros(2))
    assert np.linalg.norm(state) < np.linalg.norm(initial)
    assert state.dtype == np.dtype(np.float64)
    assert np.isfinite(state).all()


def test_scientific_suite_rejects_unstable_ftcs_ratio() -> None:
    with pytest.raises(ValueError, match="FTCS stability"):
        solver.solve(nx=11, nt=1, alpha=1.0, length=1.0, t_final=1.0)
