"""Scientific hidden tests exercised only through the documented NPZ boundary."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import yaml
from conftest import ENTRYPOINT, TASK_ROOT, make_input, write_input

if TYPE_CHECKING:
    from pathlib import Path

ALPHA, LENGTH, NX, NT, T_FINAL = 0.17, 1.3, 161, 1_800, 0.237
STATE_RELATIVE_L2 = 1.0e-5
CONTRACT = yaml.safe_load((TASK_ROOT / "contract.yaml").read_text(encoding="utf-8"))
EXPECTED_ARCHIVE_NAMES = {array["name"] for array in CONTRACT["output"]["arrays"]}


def _manufactured_solution(x: np.ndarray, alpha: float, length: float, time: float) -> np.ndarray:
    return np.sin(np.pi * x / length) * np.exp(-alpha * (np.pi / length) ** 2 * time)


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


def _candidate_solution(tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    input_path = write_input(tmp_path / "input.json", make_input(NX, NT, ALPHA, LENGTH, T_FINAL))
    output_path = tmp_path / "result.npz"
    completed = _invoke(input_path, output_path)
    assert completed.returncode == 0, completed.stderr
    assert output_path.is_file()
    with np.load(output_path) as archive:
        assert set(archive.files) == EXPECTED_ARCHIVE_NAMES
        x, state = archive["x"].copy(), archive["state"].copy()
    assert x.shape == (NX,)
    assert state.shape == (NX,)
    assert x.dtype == np.dtype(np.float64)
    assert state.dtype == np.dtype(np.float64)
    assert np.isfinite(x).all()
    assert np.isfinite(state).all()
    return x, state


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


def test_scientific_suite_rejects_unstable_ftcs_ratio_at_process_boundary(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.json", make_input(11, 1, 1.0, 1.0, 1.0))
    output_path = tmp_path / "result.npz"
    completed = _invoke(input_path, output_path)
    assert completed.returncode != 0
    assert "FTCS stability" in completed.stderr
    assert not output_path.exists()
