"""Tests for the Layer 6 held-out robustness gate."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from invariantlab.schema import load_task_contract
from invariantlab.tasks.workspace import build_agent_workspace
from invariantlab.verification import robustness
from invariantlab.verification.execution import run_task
from invariantlab.verification.oracles import check_oracle
from invariantlab.verification.robustness import check_robustness

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = REPO_ROOT / "tasks"
TASK_CASES = {
    "oscillator": (robustness.OSCILLATOR_VALID_CASES, robustness.OSCILLATOR_REJECTION_CASES),
    "kepler": (robustness.KEPLER_VALID_CASES, robustness.KEPLER_REJECTION_CASES),
    "heat1d": (robustness.HEAT_VALID_CASES, robustness.HEAT_REJECTION_CASES),
    "wave1d": (robustness.WAVE_VALID_CASES, robustness.WAVE_REJECTION_CASES),
}
# Defining physical parameter of every parameter set used in tasks/<t>/tests.
TASK_TEST_PHYSICAL_PARAMETERS = {
    "oscillator": ("omega", {0.0, 1.7, 2.0}),
    "kepler": ("mu", {0.0, 1.9, 2.0, 2.5, 4.0}),
    "heat1d": ("alpha", {0.1, 0.17, 1.0}),
    "wave1d": ("c", {0.65, 0.7, 0.75, 1.15, 2.0}),
}
OSCILLATOR_SCIENTIFIC_PARAMETERS: dict[str, object] = {
    "x0": 0.7,
    "v0": -0.35,
    "omega": 1.7,
    "dt": 1e-3,
    "n_steps": 15_000,
}
FTCS_STABILITY_CHECK = (
    "    if ratio > 0.5:\n"
    '        raise ValueError(f"FTCS stability requires r <= 0.5; got {ratio}")\n'
)


def _candidate_with_replacement(
    tmp_path: Path, task_name: str, old: str, new: str, count: int = 1
) -> Path:
    source = (TASKS_ROOT / task_name / "src" / "solver.py").read_text(encoding="utf-8")
    assert source.count(old) == count
    candidate_solver = tmp_path / "candidate" / "src" / "solver.py"
    candidate_solver.parent.mkdir(parents=True)
    candidate_solver.write_text(source.replace(old, new), encoding="utf-8")
    return candidate_solver.parents[1]


def _gates_by_case(gates: list) -> dict[str, list]:
    by_case: dict[str, list] = {}
    for gate in gates:
        case_id = gate.name.removeprefix("robustness[").split("]/", 1)[0]
        by_case.setdefault(case_id, []).append(gate)
    return by_case


@pytest.mark.parametrize("task_name", list(TASK_CASES))
def test_case_table_shape_and_held_out_parameters(task_name: str) -> None:
    valid_cases, rejection_cases = TASK_CASES[task_name]
    assert 3 <= len(valid_cases) <= 5
    assert 1 <= len(rejection_cases) <= 2
    case_ids = [case_id for case_id, _ in valid_cases + rejection_cases]
    assert len(case_ids) == len(set(case_ids))

    name, used_values = TASK_TEST_PHYSICAL_PARAMETERS[task_name]
    for case_id, parameters in valid_cases:
        assert parameters[name] not in used_values, case_id
    for case_id, parameters in valid_cases:
        if task_name == "heat1d":
            ratio = (
                parameters["alpha"]
                * (parameters["t_final"] / parameters["nt"])
                / (parameters["length"] / (parameters["nx"] - 1)) ** 2
            )
            assert ratio <= 0.5, case_id
        if task_name == "wave1d":
            courant = (
                parameters["c"]
                * (parameters["t_final"] / parameters["nt"])
                / (parameters["length"] / (parameters["nx"] - 1))
            )
            assert courant <= 1.0, case_id


def test_trusted_packages_pass_every_case_quickly_and_deterministically(tmp_path: Path) -> None:
    start = time.perf_counter()
    first = {
        task_name: check_robustness(
            TASKS_ROOT / task_name, TASKS_ROOT / task_name, tmp_path / "first" / task_name
        )
        for task_name in TASK_CASES
    }
    elapsed = time.perf_counter() - start
    assert elapsed < 60.0

    for task_name, gates in first.items():
        valid_cases, rejection_cases = TASK_CASES[task_name]
        failed = [(gate.name, gate.detail) for gate in gates if not gate.passed]
        assert not failed, failed
        by_case = _gates_by_case(gates)
        assert list(by_case) == [case_id for case_id, _ in valid_cases + rejection_cases]
        for case_id, _ in valid_cases:
            names = [gate.name.split("]/", 1)[1] for gate in by_case[case_id]]
            assert names[:5] == [
                "execution",
                "output_present",
                "archive_schema",
                "finite",
                "state_relative_l2",
            ]
            assert len(names) > 5
        for case_id, _ in rejection_cases:
            assert [gate.name for gate in by_case[case_id]] == [f"robustness[{case_id}]/rejected"]

    for task_name, gates in first.items():
        second = check_robustness(
            TASKS_ROOT / task_name, TASKS_ROOT / task_name, tmp_path / "second" / task_name
        )
        assert [gate.model_dump() for gate in second] == [gate.model_dump() for gate in gates]


@pytest.mark.parametrize("task_name", list(TASK_CASES))
def test_agent_workspace_never_exposes_case_table(task_name: str, tmp_path: Path) -> None:
    workspace = build_agent_workspace(TASKS_ROOT / task_name, tmp_path / "agent")
    files = [path for path in workspace.rglob("*") if path.is_file()]
    assert files
    assert not any(path.name == "robustness.py" for path in files)
    valid_cases, rejection_cases = TASK_CASES[task_name]
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "robustness" not in text, path
        for case_id, _ in valid_cases + rejection_cases:
            assert case_id not in text, (path, case_id)


def test_oscillator_tuned_to_scientific_omega_fails_a_valid_case(tmp_path: Path) -> None:
    task_dir = TASKS_ROOT / "oscillator"
    candidate_root = _candidate_with_replacement(
        tmp_path, "oscillator", "omega2 = omega * omega", "omega2 = 1.7 * 1.7"
    )
    public_run = run_task(
        task_dir, candidate_root, OSCILLATOR_SCIENTIFIC_PARAMETERS, tmp_path / "scientific"
    )
    assert public_run.passed and public_run.arrays is not None
    contract = load_task_contract(task_dir)
    assert check_oracle(contract, OSCILLATOR_SCIENTIFIC_PARAMETERS, public_run.arrays)[0].passed

    gates = check_robustness(task_dir, candidate_root, tmp_path / "work")

    failed = [gate.name for gate in gates if not gate.passed]
    assert "robustness[oscillator-slow-long]/state_relative_l2" in failed
    assert all(not name.endswith("/rejected") for name in failed)


def test_heat_accepting_unstable_ratio_fails_rejection_gate(tmp_path: Path) -> None:
    task_dir = TASKS_ROOT / "heat1d"
    candidate_root = _candidate_with_replacement(
        tmp_path, "heat1d", FTCS_STABILITY_CHECK, "", count=2
    )

    gates = check_robustness(task_dir, candidate_root, tmp_path / "work")

    by_name = {gate.name: gate for gate in gates}
    unstable = by_name["robustness[heat-unstable-ratio]/rejected"]
    assert not unstable.passed
    assert unstable.detail == "exit code 0; archive written: True"
    assert by_name["robustness[heat-negative-alpha]/rejected"].passed
    assert all(gate.passed for gate in gates if not gate.name.endswith("/rejected"))


def test_valid_case_layer0_failure_skips_oracle_and_invariants(tmp_path: Path) -> None:
    candidate_solver = tmp_path / "candidate" / "src" / "solver.py"
    candidate_solver.parent.mkdir(parents=True)
    candidate_solver.write_text("raise SystemExit(1)\n", encoding="utf-8")

    gates = check_robustness(TASKS_ROOT / "wave1d", candidate_solver.parents[1], tmp_path / "w")

    by_case = _gates_by_case(gates)
    for case_id, _ in robustness.WAVE_VALID_CASES:
        assert [gate.passed for gate in by_case[case_id]] == [False] * 4
        assert by_case[case_id][0].name == f"robustness[{case_id}]/execution"
    for case_id, _ in robustness.WAVE_REJECTION_CASES:
        assert by_case[case_id][0].passed


def test_unknown_task_id_has_no_robustness_cases(tmp_path: Path) -> None:
    task_dir = tmp_path / "unknown-task"
    task_dir.mkdir()
    contract = (TASKS_ROOT / "heat1d" / "contract.yaml").read_text(encoding="utf-8")
    assert contract.count("id: heat_ftcs") == 1
    (task_dir / "contract.yaml").write_text(
        contract.replace("id: heat_ftcs", "id: unknown"), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="no robustness cases for task id 'unknown'"):
        check_robustness(task_dir, task_dir, tmp_path / "work")
