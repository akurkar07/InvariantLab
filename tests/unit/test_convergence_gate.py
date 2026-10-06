"""Tests for the Layer 4 convergence gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from invariantlab.verification.convergence import (
    ORDER_MAX,
    ORDER_MIN,
    check_convergence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = REPO_ROOT / "tasks"
TRUSTED_TASKS = ("oscillator", "kepler", "heat1d", "wave1d")


@pytest.mark.parametrize("task_name", TRUSTED_TASKS)
def test_trusted_task_passes_convergence_gate(tmp_path: Path, task_name: str) -> None:
    task_dir = TASKS_ROOT / task_name

    gates, table = check_convergence(task_dir, task_dir, tmp_path)

    assert [gate.name for gate in gates] == ["observed_order[0]", "observed_order[1]"]
    for gate in gates:
        assert gate.passed, gate.detail
        assert gate.deviation is not None and ORDER_MIN <= gate.deviation <= ORDER_MAX

    assert len(table) == 3
    expected_keys = {"level", "dt", "error"}
    if task_name in ("heat1d", "wave1d"):
        expected_keys.add("dx")
    assert "order" not in table[0]
    assert all(
        set(row) == expected_keys | ({"order"} if index else set())
        for index, row in enumerate(table)
    )
    for index in (1, 2):
        assert table[index]["order"] == pytest.approx(gates[index - 1].deviation)

    if task_name in ("oscillator", "kepler"):
        assert table[0]["dt"] == pytest.approx(2.0 * table[1]["dt"])
        assert table[1]["dt"] == pytest.approx(2.0 * table[2]["dt"])
    elif task_name == "heat1d":
        assert table[0]["dx"] == pytest.approx(2.0 * table[1]["dx"])
        assert table[1]["dx"] == pytest.approx(2.0 * table[2]["dx"])
        assert table[0]["dt"] == pytest.approx(4.0 * table[1]["dt"])
        assert table[1]["dt"] == pytest.approx(4.0 * table[2]["dt"])
    else:
        assert table[0]["dx"] == pytest.approx(2.0 * table[1]["dx"])
        assert table[1]["dx"] == pytest.approx(2.0 * table[2]["dx"])
        assert table[0]["dt"] == pytest.approx(2.0 * table[1]["dt"])
        assert table[1]["dt"] == pytest.approx(2.0 * table[2]["dt"])


@pytest.mark.parametrize("task_name", TRUSTED_TASKS)
def test_convergence_gate_is_deterministic(tmp_path: Path, task_name: str) -> None:
    task_dir = TASKS_ROOT / task_name

    first_gates, first_table = check_convergence(task_dir, task_dir, tmp_path / "first")
    second_gates, second_table = check_convergence(task_dir, task_dir, tmp_path / "second")

    assert [gate.model_dump() for gate in first_gates] == [
        gate.model_dump() for gate in second_gates
    ]
    assert first_table == second_table


def _candidate_with_replacement(tmp_path: Path, task_name: str, old: str, new: str) -> Path:
    source_path = TASKS_ROOT / task_name / "src" / "solver.py"
    source = source_path.read_text(encoding="utf-8")
    assert source.count(old) == 1

    candidate_root = tmp_path / "candidate"
    candidate_solver = candidate_root / "src" / "solver.py"
    candidate_solver.parent.mkdir(parents=True)
    candidate_solver.write_text(source.replace(old, new), encoding="utf-8")
    return candidate_root


def test_first_order_oscillator_fails_convergence_gate(tmp_path: Path) -> None:
    task_dir = TASKS_ROOT / "oscillator"
    candidate_root = _candidate_with_replacement(
        tmp_path,
        "oscillator",
        "        v_half = v - 0.5 * dt * omega2 * x\n"
        "        x = x + dt * v_half\n"
        "        v = v_half - 0.5 * dt * omega2 * x",
        "        x, v = x + dt * v, v - dt * omega2 * x",
    )

    gates, _ = check_convergence(task_dir, candidate_root, tmp_path / "work")

    assert all(not gate.passed for gate in gates)
    for gate in gates:
        assert gate.deviation is not None
        # Forward Euler is first order: measured orders 1.013702 and 1.006845.
        assert 0.8 <= gate.deviation <= 1.2


def test_fixed_nt_wave_fails_convergence_gate(tmp_path: Path) -> None:
    # The loop is pinned at the coarse nt=60 while the declared nt refines, so
    # finer levels stop a fraction of the way to t_final: measured errors
    # 3.53e-05, 2.34e-01, 2.95e-01 give orders -12.693372 and -0.333481.
    task_dir = TASKS_ROOT / "wave1d"
    candidate_root = _candidate_with_replacement(
        tmp_path, "wave1d", "    for _ in range(nt):", "    for _ in range(60):"
    )

    gates, table = check_convergence(task_dir, candidate_root, tmp_path / "work")

    assert all(not gate.passed for gate in gates)
    assert all("outside band [1.8, 2.2]" in gate.detail for gate in gates)
    assert gates[0].deviation == pytest.approx(-12.693372, abs=1e-4)
    assert gates[1].deviation == pytest.approx(-0.333481, abs=1e-4)
    assert len(table) == 3


def test_layer0_failure_is_reported(tmp_path: Path) -> None:
    task_dir = TASKS_ROOT / "heat1d"
    candidate_solver = tmp_path / "candidate" / "src" / "solver.py"
    candidate_solver.parent.mkdir(parents=True)
    candidate_solver.write_text("raise SystemExit(1)\n", encoding="utf-8")

    gates, table = check_convergence(task_dir, candidate_solver.parents[1], tmp_path / "work")

    assert len(gates) == 2
    assert all(not gate.passed for gate in gates)
    assert all("Layer 0" in gate.detail for gate in gates)
    assert table == []


def test_unknown_task_id_has_no_convergence_plan(tmp_path: Path) -> None:
    task_dir = tmp_path / "unknown-task"
    task_dir.mkdir()
    contract = (TASKS_ROOT / "heat1d" / "contract.yaml").read_text(encoding="utf-8")
    assert contract.count("id: heat_ftcs") == 1
    (task_dir / "contract.yaml").write_text(
        contract.replace("id: heat_ftcs", "id: unknown"), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="no convergence plan for task id 'unknown'"):
        check_convergence(task_dir, task_dir, tmp_path / "work")
