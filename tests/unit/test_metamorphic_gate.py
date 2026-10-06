"""Tests for Layer 5 metamorphic relations."""

from __future__ import annotations

from pathlib import Path

import pytest

from invariantlab.verification.analytical import kepler_circular_orbit
from invariantlab.verification.metamorphic import (
    DIFFUSIVE_SCALING_THRESHOLD,
    ROTATION_COVARIANCE_THRESHOLD,
    TIME_REVERSAL_THRESHOLD,
    WAVE_SCALING_THRESHOLD,
    check_metamorphic,
)

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


@pytest.mark.parametrize(
    ("task_name", "gate_name", "threshold"),
    [
        ("oscillator", "time_reversal", TIME_REVERSAL_THRESHOLD),
        ("kepler", "rotation_covariance", ROTATION_COVARIANCE_THRESHOLD),
        ("heat1d", "diffusive_scaling", DIFFUSIVE_SCALING_THRESHOLD),
        ("wave1d", "wave_scaling", WAVE_SCALING_THRESHOLD),
    ],
)
def test_trusted_task_passes_metamorphic_gate(
    tmp_path: Path, task_name: str, gate_name: str, threshold: float
) -> None:
    task_dir = TASKS_ROOT / task_name

    gates = check_metamorphic(task_dir, task_dir, _parameters(task_name), tmp_path)

    assert len(gates) == 1
    gate = gates[0]
    assert gate.passed
    assert gate.name == gate_name
    assert gate.deviation is not None and gate.deviation <= threshold
    assert gate.threshold == threshold


@pytest.mark.parametrize("task_name", TRUSTED_TASKS)
def test_metamorphic_gate_is_deterministic(tmp_path: Path, task_name: str) -> None:
    task_dir = TASKS_ROOT / task_name
    parameters = _parameters(task_name)

    first = check_metamorphic(task_dir, task_dir, parameters, tmp_path / "first")
    second = check_metamorphic(task_dir, task_dir, parameters, tmp_path / "second")

    assert [gate.model_dump() for gate in first] == [gate.model_dump() for gate in second]


def _candidate_with_replacement(tmp_path: Path, task_name: str, old: str, new: str) -> Path:
    source_path = TASKS_ROOT / task_name / "src" / "solver.py"
    source = source_path.read_text(encoding="utf-8")
    assert source.count(old) == 1

    candidate_root = tmp_path / "candidate"
    candidate_solver = candidate_root / "src" / "solver.py"
    candidate_solver.parent.mkdir(parents=True)
    candidate_solver.write_text(source.replace(old, new), encoding="utf-8")
    return candidate_root


@pytest.mark.parametrize(
    ("task_name", "old", "new"),
    [
        (
            "oscillator",
            "        v_half = v - 0.5 * dt * omega2 * x\n"
            "        x = x + dt * v_half\n"
            "        v = v_half - 0.5 * dt * omega2 * x",
            "        v = v - dt * omega2 * x\n        x = x + dt * v",
        ),
        (
            "heat1d",
            "state = np.sin(np.pi * x / length)",
            "state = np.sin(np.pi * x)",
        ),
        (
            "wave1d",
            "state = np.sin(np.pi * x / length)",
            "state = np.sin(np.pi * x)",
        ),
        (
            "kepler",
            "    return factor * rx, factor * ry",
            "    return factor * rx + 0.05, factor * ry",
        ),
    ],
)
def test_broken_relation_fails_with_nonzero_residual(
    tmp_path: Path, task_name: str, old: str, new: str
) -> None:
    task_dir = TASKS_ROOT / task_name
    candidate_root = _candidate_with_replacement(tmp_path, task_name, old, new)

    gates = check_metamorphic(task_dir, candidate_root, _parameters(task_name), tmp_path / "work")

    assert len(gates) == 1
    gate = gates[0]
    assert not gate.passed
    assert gate.deviation is not None and gate.threshold is not None
    assert gate.deviation > gate.threshold


def test_base_layer0_failure_is_reported(tmp_path: Path) -> None:
    task_dir = TASKS_ROOT / "heat1d"
    candidate_solver = tmp_path / "candidate" / "src" / "solver.py"
    candidate_solver.parent.mkdir(parents=True)
    candidate_solver.write_text("raise SystemExit(1)\n", encoding="utf-8")

    gates = check_metamorphic(
        task_dir,
        candidate_solver.parents[1],
        _parameters("heat1d"),
        tmp_path / "work",
    )

    assert len(gates) == 1
    assert not gates[0].passed
    assert gates[0].detail.startswith("base run failed Layer 0 execution")


def test_transformed_layer0_failure_is_reported(tmp_path: Path) -> None:
    task_dir = TASKS_ROOT / "heat1d"
    candidate_root = _candidate_with_replacement(
        tmp_path,
        "heat1d",
        '    """Integrate u_t = alpha*u_xx with stable FTCS and zero Dirichlet boundaries."""',
        '    """Integrate u_t = alpha*u_xx with stable FTCS and zero Dirichlet boundaries."""\n'
        "    if length > 2.0:\n"
        '        raise ValueError("boom")',
    )

    gates = check_metamorphic(task_dir, candidate_root, _parameters("heat1d"), tmp_path / "work")

    assert len(gates) == 1
    assert not gates[0].passed
    assert gates[0].detail.startswith("transformed run failed Layer 0 execution")


def test_unknown_task_id_has_no_metamorphic_relation(tmp_path: Path) -> None:
    task_dir = tmp_path / "unknown-task"
    task_dir.mkdir()
    contract = (TASKS_ROOT / "heat1d" / "contract.yaml").read_text(encoding="utf-8")
    assert contract.count("id: heat_ftcs") == 1
    (task_dir / "contract.yaml").write_text(
        contract.replace("id: heat_ftcs", "id: unknown"), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="no metamorphic relation for task id 'unknown'"):
        check_metamorphic(task_dir, task_dir, {}, tmp_path / "work")
