"""Scientific hidden tests exercised only through the documented NPZ boundary."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import yaml
from conftest import ENTRYPOINT, TASK_ROOT, make_input, write_input

from invariantlab.verification.analytical import oscillator_trajectory

if TYPE_CHECKING:
    from pathlib import Path

X0 = 0.7
V0 = -0.35
OMEGA = 1.7
DT = 1e-3
N_STEPS = 15_000
STATE_RELATIVE_L2 = 1.0e-5
ENERGY_RELATIVE_DRIFT = 1.0e-6
CONTRACT = yaml.safe_load((TASK_ROOT / "contract.yaml").read_text(encoding="utf-8"))
EXPECTED_ARCHIVE_NAMES = {array["name"] for array in CONTRACT["output"]["arrays"]}


def _candidate_archive(tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    input_path = write_input(tmp_path / "input.json", make_input(X0, V0, OMEGA, DT, N_STEPS))
    output_path = tmp_path / "result.npz"
    completed = subprocess.run(
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
    assert completed.returncode == 0, completed.stderr
    assert output_path.is_file()
    with np.load(output_path) as archive:
        assert set(archive.files) == EXPECTED_ARCHIVE_NAMES
        time, state = archive["time"].copy(), archive["state"].copy()
    assert time.shape == (N_STEPS + 1,)
    assert state.shape == (N_STEPS + 1, 2)
    assert time.dtype == np.dtype(np.float64)
    assert state.dtype == np.dtype(np.float64)
    assert np.isfinite(time).all()
    assert np.isfinite(state).all()
    return time, state


def test_state_matches_analytical_solution(tmp_path: Path) -> None:
    time, state = _candidate_archive(tmp_path)
    exact = oscillator_trajectory(time, X0, V0, OMEGA)[:, 1:]
    relative_l2 = np.linalg.norm(state - exact) / np.linalg.norm(exact)
    assert relative_l2 < STATE_RELATIVE_L2, f"state_relative_l2={relative_l2:.3e}"


def test_energy_drift_is_bounded_over_horizon(tmp_path: Path) -> None:
    _, state = _candidate_archive(tmp_path)
    energy = 0.5 * state[:, 1] ** 2 + 0.5 * OMEGA**2 * state[:, 0] ** 2
    relative_drift = np.max(np.abs(energy - energy[0])) / energy[0]
    assert relative_drift < ENERGY_RELATIVE_DRIFT, f"energy_relative_drift={relative_drift:.3e}"
