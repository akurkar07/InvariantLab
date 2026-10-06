"""Scientific hidden tests exercised only through the documented NPZ boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import yaml
from conftest import TASK_ROOT, make_input

from invariantlab.verification.analytical import wave_standing_trajectory
from invariantlab.verification.execution import run_task

if TYPE_CHECKING:
    from pathlib import Path

MODE_AMPLITUDE_ABSOLUTE_ERROR = 5.0e-6
STARTUP_RELATIVE_L2 = 5.0e-3
WAVE_CASES = (
    pytest.param(401, 400, 0.65, 1.3, 0.39, id="slow-wave-nonunit-domain"),
    pytest.param(401, 500, 1.15, 1.7, 0.319, id="fast-wave-nonunit-domain"),
)
CONTRACT = yaml.safe_load((TASK_ROOT / "contract.yaml").read_text(encoding="utf-8"))
STATE_RELATIVE_L2 = CONTRACT["numerics"]["tolerances"]["state_relative_l2"]


@pytest.mark.parametrize(("nx", "nt", "c", "length", "t_final"), WAVE_CASES)
def test_non_special_standing_wave_phase_and_amplitude_match_oracle(
    tmp_path: Path, nx: int, nt: int, c: float, length: float, t_final: float
) -> None:
    run = run_task(
        TASK_ROOT,
        TASK_ROOT,
        make_input(nx, nt, c, length, t_final)["parameters"],
        tmp_path,
    )
    assert run.passed, (run.gates, run.stderr)
    assert run.arrays is not None
    x, state = run.arrays["x"], run.arrays["state"]
    assert x.shape == (nx,)
    assert state.shape == (nx,)
    expected = wave_standing_trajectory(x, t_final, c, length=length)
    phase = np.pi * c * t_final / length
    mode = np.sin(np.pi * x / length)
    assert abs(np.cos(phase)) > 0.1
    assert abs(abs(np.cos(phase)) - 1.0) > 0.1
    state_relative_l2 = np.linalg.norm(state - expected) / np.linalg.norm(expected)
    amplitude_error = abs(np.dot(state, mode) / np.dot(mode, mode) - np.cos(phase))
    assert state_relative_l2 < STATE_RELATIVE_L2, f"state_relative_l2={state_relative_l2:.3e}"
    assert amplitude_error < MODE_AMPLITUDE_ABSOLUTE_ERROR, (
        f"mode_amplitude_error={amplitude_error:.3e}"
    )


def test_zero_velocity_startup_matches_non_special_standing_wave_oracle(tmp_path: Path) -> None:
    nx, nt, c, length, t_final = 5, 1, 0.75, 1.7, 0.51
    run = run_task(
        TASK_ROOT,
        TASK_ROOT,
        make_input(nx, nt, c, length, t_final)["parameters"],
        tmp_path,
    )
    assert run.passed, (run.gates, run.stderr)
    assert run.arrays is not None
    x, state = run.arrays["x"], run.arrays["state"]
    assert x.shape == (nx,)
    assert state.shape == (nx,)
    expected = wave_standing_trajectory(x, t_final, c, length=length)
    phase = np.pi * c * t_final / length
    startup_relative_l2 = np.linalg.norm(state - expected) / np.linalg.norm(expected)
    assert abs(np.cos(phase)) > 0.1
    assert abs(abs(np.cos(phase)) - 1.0) > 0.1
    assert startup_relative_l2 < STARTUP_RELATIVE_L2, (
        f"startup_relative_l2={startup_relative_l2:.3e}"
    )


def test_different_wave_speeds_produce_distinct_correct_phases(tmp_path: Path) -> None:
    slow, fast = WAVE_CASES[0].values, WAVE_CASES[1].values
    slow_run = run_task(
        TASK_ROOT, TASK_ROOT, make_input(*slow)["parameters"], tmp_path / "slow"
    )
    assert slow_run.passed, (slow_run.gates, slow_run.stderr)
    assert slow_run.arrays is not None
    slow_x, slow_state = slow_run.arrays["x"], slow_run.arrays["state"]
    assert slow_x.shape == (slow[0],)
    assert slow_state.shape == (slow[0],)
    fast_run = run_task(
        TASK_ROOT, TASK_ROOT, make_input(*fast)["parameters"], tmp_path / "fast"
    )
    assert fast_run.passed, (fast_run.gates, fast_run.stderr)
    assert fast_run.arrays is not None
    fast_x, fast_state = fast_run.arrays["x"], fast_run.arrays["state"]
    assert fast_x.shape == (fast[0],)
    assert fast_state.shape == (fast[0],)
    slow_expected = wave_standing_trajectory(slow_x, slow[4], slow[2], length=slow[3])
    fast_expected = wave_standing_trajectory(fast_x, fast[4], fast[2], length=fast[3])
    assert not np.isclose(
        np.cos(np.pi * slow[2] * slow[4] / slow[3]), np.cos(np.pi * fast[2] * fast[4] / fast[3])
    )
    np.testing.assert_allclose(
        slow_state, slow_expected, rtol=0.0, atol=MODE_AMPLITUDE_ABSOLUTE_ERROR
    )
    np.testing.assert_allclose(
        fast_state, fast_expected, rtol=0.0, atol=MODE_AMPLITUDE_ABSOLUTE_ERROR
    )


def test_scientific_suite_rejects_unstable_cfl_at_process_boundary(tmp_path: Path) -> None:
    run = run_task(
        TASK_ROOT,
        TASK_ROOT,
        make_input(11, 1, 2.0, 1.0, 1.0)["parameters"],
        tmp_path,
    )
    assert not run.passed
    assert run.gates[0].name == "execution" and not run.gates[0].passed
    assert run.returncode not in (0, None)
    assert "Courant" in run.stderr
    assert not (tmp_path / "result.npz").exists()
