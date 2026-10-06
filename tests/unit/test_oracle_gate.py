"""Tests for Layer 2 trusted-oracle comparison."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from invariantlab.schema import TaskContract, load_task_contract
from invariantlab.verification.analytical import kepler_elliptic_orbit
from invariantlab.verification.execution import run_task
from invariantlab.verification.oracles import GATE_NAME, check_oracle

if TYPE_CHECKING:
    import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = REPO_ROOT / "tasks"
TRUSTED_TASKS = ("oscillator", "kepler", "heat1d", "wave1d")


def _parameters(task_name: str) -> dict[str, object]:
    if task_name == "oscillator":
        return {"x0": 0.7, "v0": -0.35, "omega": 1.7, "dt": 1e-3, "n_steps": 15_000}
    if task_name == "kepler":
        orbit = kepler_elliptic_orbit(0.0, 1.9, 2.3, 0.41, 0.63)
        return {
            "rx": orbit[0],
            "ry": orbit[1],
            "vx": orbit[3],
            "vy": orbit[4],
            "mu": 1.9,
            "dt": 0.004,
            "n_steps": 2_178,
        }
    if task_name == "heat1d":
        return {"nx": 161, "nt": 1_800, "alpha": 0.17, "length": 1.3, "t_final": 0.237}
    if task_name == "wave1d":
        return {"nx": 401, "nt": 400, "c": 0.65, "length": 1.3, "t_final": 0.39}
    raise AssertionError(f"unknown trusted task: {task_name}")


def _run_task(task_name: str, tmp_path: Path) -> tuple[TaskContract, dict[str, np.ndarray]]:
    task_dir = TASKS_ROOT / task_name
    run = run_task(task_dir, task_dir, _parameters(task_name), tmp_path)
    assert run.passed, (run.gates, run.stderr)
    assert run.arrays is not None
    return load_task_contract(task_dir), run.arrays


@pytest.mark.parametrize("task_name", TRUSTED_TASKS)
def test_trusted_task_passes_oracle_gate(tmp_path: Path, task_name: str) -> None:
    contract, arrays = _run_task(task_name, tmp_path)

    gate = check_oracle(contract, _parameters(task_name), arrays)[0]

    assert gate.name == GATE_NAME
    assert gate.passed
    assert gate.threshold == contract.numerics.tolerances[GATE_NAME] == 1e-5
    assert gate.deviation is not None
    assert 0 <= gate.deviation < gate.threshold


@pytest.mark.parametrize("task_name", TRUSTED_TASKS)
def test_wrong_state_fails_oracle_gate(tmp_path: Path, task_name: str) -> None:
    contract, arrays = _run_task(task_name, tmp_path)
    wrong_arrays = {**arrays, "state": arrays["state"] * (1 + 1e-3)}

    gate = check_oracle(contract, _parameters(task_name), wrong_arrays)[0]

    assert not gate.passed
    assert gate.deviation == pytest.approx(1e-3, rel=1e-2)
    assert gate.deviation > gate.threshold


def test_shape_mismatch_fails_oracle_gate(tmp_path: Path) -> None:
    contract, arrays = _run_task("heat1d", tmp_path)
    wrong_arrays = {**arrays, "state": arrays["state"][:-1]}

    gate = check_oracle(contract, _parameters("heat1d"), wrong_arrays)[0]

    assert not gate.passed
    assert gate.threshold == contract.numerics.tolerances[GATE_NAME]
    assert gate.deviation is None
    assert "does not match oracle shape" in gate.detail


def test_kepler_time_not_starting_at_zero_fails_oracle_gate(tmp_path: Path) -> None:
    contract, arrays = _run_task("kepler", tmp_path)
    invalid_arrays = {**arrays, "time": arrays["time"] + 1.0}

    gate = check_oracle(contract, _parameters("kepler"), invalid_arrays)[0]

    assert not gate.passed
    assert gate.threshold == contract.numerics.tolerances[GATE_NAME]
    assert gate.deviation is None
    assert "t_eval must begin at zero" in gate.detail


def test_unknown_contract_id_raises_value_error(tmp_path: Path) -> None:
    contract, arrays = _run_task("oscillator", tmp_path)

    with pytest.raises(ValueError, match="no oracle for task contract 'nope'"):
        check_oracle(contract.model_copy(update={"id": "nope"}), _parameters("oscillator"), arrays)
