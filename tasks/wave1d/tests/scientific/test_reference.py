"""Scientific hidden tests exercised only through the documented NPZ boundary."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
import yaml
from conftest import ENTRYPOINT, TASK_ROOT, make_input, write_input

from invariantlab.verification.analytical import wave_standing_trajectory

if TYPE_CHECKING:
    from pathlib import Path

STATE_RELATIVE_L2 = 5.0e-6
MODE_AMPLITUDE_ABSOLUTE_ERROR = 5.0e-6
STARTUP_RELATIVE_L2 = 5.0e-3
WAVE_CASES = (
    pytest.param(401, 400, 0.65, 1.3, 0.39, id="slow-wave-nonunit-domain"),
    pytest.param(401, 500, 1.15, 1.7, 0.319, id="fast-wave-nonunit-domain"),
)
CONTRACT = yaml.safe_load((TASK_ROOT / "contract.yaml").read_text(encoding="utf-8"))
EXPECTED_ARCHIVE_NAMES = {array["name"] for array in CONTRACT["output"]["arrays"]}


def _invoke(input_path: Path, output_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ENTRYPOINT.relative_to(TASK_ROOT)),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=TASK_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _candidate_state(
    tmp_path: Path, nx: int, nt: int, c: float, length: float, t_final: float
) -> tuple[np.ndarray, np.ndarray]:
    input_path = write_input(tmp_path / "input.json", make_input(nx, nt, c, length, t_final))
    output_path = tmp_path / "result.npz"
    completed = _invoke(input_path, output_path)
    assert completed.returncode == 0, completed.stderr
    assert output_path.is_file()
    with np.load(output_path) as archive:
        assert set(archive.files) == EXPECTED_ARCHIVE_NAMES
        x, state = archive["x"].copy(), archive["state"].copy()
    assert x.shape == (nx,)
    assert state.shape == (nx,)
    assert x.dtype == np.dtype(np.float64)
    assert state.dtype == np.dtype(np.float64)
    assert np.isfinite(x).all()
    assert np.isfinite(state).all()
    return x, state


@pytest.mark.parametrize(("nx", "nt", "c", "length", "t_final"), WAVE_CASES)
def test_non_special_standing_wave_phase_and_amplitude_match_oracle(
    tmp_path: Path, nx: int, nt: int, c: float, length: float, t_final: float
) -> None:
    x, state = _candidate_state(tmp_path, nx, nt, c, length, t_final)
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
    x, state = _candidate_state(tmp_path, nx, nt, c, length, t_final)
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
    slow_x, slow_state = _candidate_state(tmp_path / "slow", *slow)
    fast_x, fast_state = _candidate_state(tmp_path / "fast", *fast)
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
    input_path = write_input(tmp_path / "input.json", make_input(11, 1, 2.0, 1.0, 1.0))
    output_path = tmp_path / "result.npz"
    completed = _invoke(input_path, output_path)
    assert completed.returncode != 0
    assert "Courant" in completed.stderr
    assert not output_path.exists()
