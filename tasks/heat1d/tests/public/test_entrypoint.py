"""End-to-end tests of the documented heat-task invocation protocol."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
from conftest import ENTRYPOINT, TASK_ROOT, make_input, write_input

if TYPE_CHECKING:
    from pathlib import Path

NX = 47


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
    input_path = write_input(tmp_path / "input.json", make_input(NX, 120, 0.1, 1.2, 0.05))
    output_path = tmp_path / "result.npz"

    completed = _invoke(input_path, output_path)

    assert completed.returncode == 0, completed.stderr
    assert output_path.is_file()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["input.json", "result.npz"]
    with np.load(output_path) as archive:
        assert sorted(archive.files) == ["state", "x"]
        assert archive["x"].shape == (NX,)
        assert archive["state"].shape == (NX,)
        assert archive["x"].dtype == np.dtype(np.float64)
        assert archive["state"].dtype == np.dtype(np.float64)
        assert np.isfinite(archive["x"]).all()
        assert np.isfinite(archive["state"]).all()
        np.testing.assert_array_equal(archive["state"][[0, -1]], np.zeros(2))


def test_entrypoint_exits_non_zero_on_unstable_input(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.json", make_input(11, 1, 1.0, 1.0, 1.0))
    output_path = tmp_path / "result.npz"

    completed = _invoke(input_path, output_path)

    assert completed.returncode != 0
    assert "FTCS stability" in completed.stderr
    assert not output_path.exists()
