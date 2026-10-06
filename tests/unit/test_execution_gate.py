"""Tests for Layer 0 task execution and archive validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from invariantlab.schema import load_task_contract
from invariantlab.verification.analytical import kepler_circular_orbit
from invariantlab.verification.execution import GATE_NAMES, TaskRun, run_task

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = REPO_ROOT / "tasks"
TRUSTED_TASKS = ("oscillator", "kepler", "heat1d", "wave1d")


def _parameters(task_name: str) -> dict[str, object]:
    if task_name == "oscillator":
        return {"x0": 0.7, "v0": -0.35, "omega": 1.7, "dt": 1e-3, "n_steps": 15_000}
    if task_name == "kepler":
        orbit = kepler_circular_orbit(0.0, 2.5, 1.7, 0.37)
        return {
            "rx": orbit[0],
            "ry": orbit[1],
            "vx": orbit[3],
            "vy": orbit[4],
            "mu": 2.5,
            "dt": 0.004,
            "n_steps": 1_080,
        }
    if task_name == "heat1d":
        return {"nx": 161, "nt": 1_800, "alpha": 0.17, "length": 1.3, "t_final": 0.237}
    if task_name == "wave1d":
        return {"nx": 401, "nt": 400, "c": 0.65, "length": 1.3, "t_final": 0.39}
    raise AssertionError(f"unknown trusted task: {task_name}")


@pytest.mark.parametrize("task_name", TRUSTED_TASKS)
def test_trusted_task_passes_all_execution_gates(tmp_path: Path, task_name: str) -> None:
    task_dir = TASKS_ROOT / task_name
    run = run_task(task_dir, task_dir, _parameters(task_name), tmp_path)

    contract = load_task_contract(task_dir)
    assert run.passed
    assert [gate.name for gate in run.gates] == list(GATE_NAMES)
    assert run.returncode == 0
    assert run.arrays is not None
    assert set(run.arrays) == {spec.name for spec in contract.output.arrays}


def test_writes_contract_input_payload(tmp_path: Path) -> None:
    task_dir = TASKS_ROOT / "oscillator"
    parameters = _parameters("oscillator")

    run_task(task_dir, task_dir, parameters, tmp_path)

    contract = load_task_contract(task_dir)
    assert json.loads((tmp_path / "input.json").read_text(encoding="utf-8")) == {
        "task_id": contract.id,
        "parameters": parameters,
        "numerics": {"dtype": contract.numerics.dtype, "seed": contract.numerics.seed},
    }


def _candidate_root(tmp_path: Path, body: str) -> Path:
    candidate_root = tmp_path / "candidate"
    solver = candidate_root / "src" / "solver.py"
    solver.parent.mkdir(parents=True)
    solver.write_text(
        "import argparse\n"
        "import numpy as np\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--input')\n"
        "parser.add_argument('--output')\n"
        "args = parser.parse_args()\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return candidate_root


def _assert_failed_gate(run: TaskRun, name: str, detail: str) -> None:
    assert not run.passed
    failed_index = next(index for index, gate in enumerate(run.gates) if gate.name == name)
    assert all(gate.passed for gate in run.gates[:failed_index])
    assert not run.gates[failed_index].passed
    assert run.gates[failed_index].detail == detail
    for gate in run.gates[failed_index + 1 :]:
        assert gate.detail == f"skipped: {name} failed"
    assert run.arrays is None


def test_nonzero_exit_fails_execution_gate(tmp_path: Path) -> None:
    task_dir = TASKS_ROOT / "heat1d"
    parameters = {"nx": 11, "nt": 1, "alpha": 1.0, "length": 1.0, "t_final": 1.0}

    run = run_task(task_dir, task_dir, parameters, tmp_path)

    _assert_failed_gate(run, "execution", f"exit code {run.returncode}")
    assert run.returncode not in (0, None)
    assert "FTCS stability" in run.stderr


@pytest.mark.parametrize(
    ("body", "timeout_seconds", "preexisting_output", "gate", "detail"),
    [
        ("while True: pass", 1, False, "execution", "timed out after 1 s"),
        ("pass", None, False, "output_present", "result.npz was not written"),
        ("pass", None, True, "output_present", "result.npz was not written"),
        (
            "np.savez(args.output, time=np.array([0.0]))",
            None,
            False,
            "archive_schema",
            "archive keys mismatch (missing: ['state']; unexpected: [])",
        ),
        (
            "np.savez(args.output, time=np.array([0.0]), "
            "state=np.array([[0.0, 0.0]], dtype=np.float32))",
            None,
            False,
            "archive_schema",
            "array 'state' has dtype float32; expected float64",
        ),
        (
            "np.savez(args.output, time=np.array([0.0]), state=np.array([0.0, 0.0]))",
            None,
            False,
            "archive_schema",
            "array 'state' has rank 1; expected 2",
        ),
        (
            "np.savez(args.output, time=np.array([0.0]), "
            "state=np.array([[np.nan, 0.0]]))",
            None,
            False,
            "finite",
            "non-finite values in arrays: ['state']",
        ),
        (
            "np.savez(args.output, time=np.array([0.0]), "
            "state=np.array([0.0, None], dtype=object))",
            None,
            False,
            "archive_schema",
            "array 'state' cannot be loaded without pickle (object arrays are rejected)",
        ),
    ],
)
def test_failures_close_later_gates_and_discard_arrays(
    tmp_path: Path,
    body: str,
    timeout_seconds: float | None,
    preexisting_output: bool,
    gate: str,
    detail: str,
) -> None:
    task_dir = TASKS_ROOT / "oscillator"
    candidate_root = _candidate_root(tmp_path, body)
    work_dir = tmp_path / "work"
    if preexisting_output:
        work_dir.mkdir()
        (work_dir / "result.npz").write_bytes(b"stale archive")

    run = run_task(
        task_dir,
        candidate_root,
        {},
        work_dir,
        timeout_seconds=timeout_seconds,
    )

    _assert_failed_gate(run, gate, detail)
    if gate == "execution":
        assert run.returncode is None
    elif gate == "output_present":
        assert run.returncode == 0
        assert not (work_dir / "result.npz").exists()


def test_candidate_contract_cannot_change_trusted_archive_schema(tmp_path: Path) -> None:
    task_dir = TASKS_ROOT / "oscillator"
    candidate_root = _candidate_root(
        tmp_path, "np.savez(args.output, time=np.array([0.0]))"
    )
    (candidate_root / "contract.yaml").write_text(
        "id: untrusted\n"
        "family: oscillator\n"
        "output:\n"
        "  path: result.npz\n"
        "  arrays:\n"
        "    - name: time\n"
        "      shape: [null]\n"
        "      dtype: float64\n",
        encoding="utf-8",
    )

    run = run_task(task_dir, candidate_root, {}, tmp_path / "work")

    _assert_failed_gate(
        run,
        "archive_schema",
        "archive keys mismatch (missing: ['state']; unexpected: [])",
    )


def test_gate_results_are_deterministic_for_pass_and_failure(tmp_path: Path) -> None:
    task_dir = TASKS_ROOT / "heat1d"
    parameters = _parameters("heat1d")
    first = run_task(task_dir, task_dir, parameters, tmp_path / "heat-one")
    second = run_task(task_dir, task_dir, parameters, tmp_path / "heat-two")
    assert first.gates == second.gates

    missing_state_candidate = _candidate_root(
        tmp_path, "np.savez(args.output, time=np.array([0.0]))"
    )
    failure_one = run_task(
        TASKS_ROOT / "oscillator",
        missing_state_candidate,
        {},
        tmp_path / "failure-one",
    )
    failure_two = run_task(
        TASKS_ROOT / "oscillator",
        missing_state_candidate,
        {},
        tmp_path / "failure-two",
    )
    assert failure_one.gates == failure_two.gates
