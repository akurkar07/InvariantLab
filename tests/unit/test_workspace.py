from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from invariantlab.tasks.workspace import build_agent_workspace, build_evaluation_workspace
from invariantlab.verification.analytical import kepler_circular_orbit
from invariantlab.verification.execution import run_task

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_NAMES = ("oscillator", "kepler", "heat1d", "wave1d")


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


def _run_pytest(cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    return result


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_agent_workspace_contains_only_public_task_files(task_name: str, tmp_path: Path) -> None:
    task_root = REPO_ROOT / "tasks" / task_name
    agent_workspace = build_agent_workspace(task_root, tmp_path / "agent")

    assert not (agent_workspace / "tests" / "scientific").exists()
    for relative_path in (
        "contract.yaml",
        "specification.md",
        "src/solver.py",
        "tests/conftest.py",
        "tests/public",
    ):
        assert (agent_workspace / relative_path).exists()
    for path in agent_workspace.rglob("*"):
        if path.is_file():
            assert "invariantlab.verification" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_agent_workspace_public_tests_pass(task_name: str, tmp_path: Path) -> None:
    agent_workspace = build_agent_workspace(REPO_ROOT / "tasks" / task_name, tmp_path / "agent")

    _run_pytest(agent_workspace)


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_evaluation_workspace_uses_trusted_tests_and_candidate_source(
    task_name: str, tmp_path: Path
) -> None:
    task_root = REPO_ROOT / "tasks" / task_name
    agent_workspace = build_agent_workspace(task_root, tmp_path / "agent")

    contract_path = agent_workspace / "contract.yaml"
    contract_data = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    for tolerance_name in contract_data["numerics"]["tolerances"]:
        contract_data["numerics"]["tolerances"][tolerance_name] = 1.0
    contract_path.write_text(yaml.safe_dump(contract_data), encoding="utf-8")
    with (agent_workspace / "tests" / "conftest.py").open("a", encoding="utf-8") as handle:
        handle.write("\n\ndef pytest_collection_modifyitems(items):\n    items.clear()\n")
    (agent_workspace / "tests" / "public" / "test_solver.py").write_text(
        "def test_trivial_pass():\n    assert True\n", encoding="utf-8"
    )
    with (agent_workspace / "src" / "solver.py").open("a", encoding="utf-8") as handle:
        handle.write("\n# Candidate workspace change.\n")

    evaluation_workspace = build_evaluation_workspace(
        task_root, agent_workspace, tmp_path / "evaluation"
    )
    for relative_path in (
        "contract.yaml",
        "tests/conftest.py",
        "tests/public/test_solver.py",
        "tests/scientific/test_reference.py",
    ):
        assert (evaluation_workspace / relative_path).read_bytes() == (
            task_root / relative_path
        ).read_bytes()
    assert (evaluation_workspace / "src" / "solver.py").read_bytes() == (
        agent_workspace / "src" / "solver.py"
    ).read_bytes()


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_untampered_evaluation_workspace_tests_pass(task_name: str, tmp_path: Path) -> None:
    task_root = REPO_ROOT / "tasks" / task_name
    agent_workspace = build_agent_workspace(task_root, tmp_path / "agent")
    evaluation_workspace = build_evaluation_workspace(
        task_root, agent_workspace, tmp_path / "evaluation"
    )

    _run_pytest(evaluation_workspace)


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_evaluation_workspace_runs_under_execution_gate(task_name: str, tmp_path: Path) -> None:
    task_root = REPO_ROOT / "tasks" / task_name
    agent_workspace = build_agent_workspace(task_root, tmp_path / "agent")
    evaluation_workspace = build_evaluation_workspace(
        task_root, agent_workspace, tmp_path / "evaluation"
    )

    run = run_task(
        evaluation_workspace,
        evaluation_workspace,
        _parameters(task_name),
        tmp_path / "work",
    )

    assert run.passed, (run.gates, run.stderr)
    assert run.returncode == 0
    assert run.arrays is not None


@pytest.mark.parametrize("builder", ["agent", "evaluation"])
def test_builders_reject_nonempty_destinations(builder: str, tmp_path: Path) -> None:
    task_root = REPO_ROOT / "tasks" / "oscillator"
    candidate_workspace = build_agent_workspace(task_root, tmp_path / "candidate")
    dest = tmp_path / "nonempty"
    dest.mkdir()
    (dest / "existing").touch()

    with pytest.raises(ValueError, match="destination must be empty"):
        if builder == "agent":
            build_agent_workspace(task_root, dest)
        else:
            build_evaluation_workspace(task_root, candidate_workspace, dest)


@pytest.mark.parametrize("builder", ["agent", "evaluation"])
def test_builders_reject_destinations_inside_task_root(builder: str, tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    shutil.copytree(REPO_ROOT / "tasks" / "oscillator", task_root)
    candidate_workspace = build_agent_workspace(task_root, tmp_path / "candidate")
    dest = task_root / "workspace"

    with pytest.raises(ValueError, match="outside source roots"):
        if builder == "agent":
            build_agent_workspace(task_root, dest)
        else:
            build_evaluation_workspace(task_root, candidate_workspace, dest)


def test_evaluation_builder_rejects_destination_inside_candidate(tmp_path: Path) -> None:
    task_root = REPO_ROOT / "tasks" / "oscillator"
    candidate_workspace = build_agent_workspace(task_root, tmp_path / "candidate")

    with pytest.raises(ValueError, match="outside source roots"):
        build_evaluation_workspace(task_root, candidate_workspace, candidate_workspace / "eval")


def test_agent_builder_rejects_symlink_inside_task_source(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    shutil.copytree(REPO_ROOT / "tasks" / "oscillator", task_root)
    outside_file = tmp_path / "outside.py"
    outside_file.write_text("pass\n", encoding="utf-8")
    try:
        os.symlink(outside_file, task_root / "src" / "linked.py")
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(ValueError, match="Symlinks are not allowed"):
        build_agent_workspace(task_root, tmp_path / "agent")


def test_evaluation_builder_rejects_symlink_inside_candidate_source(tmp_path: Path) -> None:
    task_root = REPO_ROOT / "tasks" / "oscillator"
    candidate_workspace = build_agent_workspace(task_root, tmp_path / "candidate")
    outside_file = tmp_path / "outside.py"
    outside_file.write_text("pass\n", encoding="utf-8")
    try:
        os.symlink(outside_file, candidate_workspace / "src" / "linked.py")
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(ValueError, match="Symlinks are not allowed"):
        build_evaluation_workspace(task_root, candidate_workspace, tmp_path / "evaluation")
