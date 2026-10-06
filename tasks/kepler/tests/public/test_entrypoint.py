"""End-to-end tests of the documented Kepler task invocation protocol."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
from conftest import ENTRYPOINT, TASK_ROOT, make_input, write_input

if TYPE_CHECKING:
    from pathlib import Path

N_STEPS = 120


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


def test_entrypoint_runs_from_task_root_and_writes_contract_archive(tmp_path: Path) -> None:
    input_path = write_input(
        tmp_path / "input.json", make_input(1.5, 0.2, -0.1, 0.9, 2.0, 0.02, N_STEPS)
    )
    output_path = tmp_path / "result.npz"

    completed = _invoke(input_path, output_path)

    assert completed.returncode == 0, completed.stderr
    assert output_path.is_file()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["input.json", "result.npz"]
    with np.load(output_path) as archive:
        assert sorted(archive.files) == ["state", "time"]
        assert archive["time"].shape == (N_STEPS + 1,)
        assert archive["state"].shape == (N_STEPS + 1, 4)
        assert archive["time"].dtype == np.dtype(np.float64)
        assert archive["state"].dtype == np.dtype(np.float64)
        assert np.isfinite(archive["time"]).all()
        assert np.isfinite(archive["state"]).all()


def test_entrypoint_exits_non_zero_on_invalid_input(tmp_path: Path) -> None:
    payload = make_input(1.5, 0.2, -0.1, 0.9, 2.0, 0.02, N_STEPS)
    payload["parameters"]["mu"] = 0.0
    input_path = write_input(tmp_path / "input.json", payload)
    output_path = tmp_path / "result.npz"

    completed = _invoke(input_path, output_path)

    assert completed.returncode != 0
    assert "mu" in completed.stderr
    assert not output_path.exists()
