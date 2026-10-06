"""Tests for package-task repair runs without a Docker daemon."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from invariantlab.cli import app
from invariantlab.config import ExperimentConfig
from invariantlab.experiments import repair
from invariantlab.mutations import discover_mutants
from invariantlab.schema import load_task_contract
from invariantlab.tasks.workspace import build_agent_workspace

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = REPO_ROOT / "tasks"
PACKAGE_CONFIG = REPO_ROOT / "configs/experiments/repair-package-replay.yaml"
RUNNER = CliRunner()


def _write_experiment(
    tmp_path: Path,
    *,
    task: Path,
    mutation: str | Path,
    conditions: tuple[str, ...] = ("weak",),
) -> Path:
    config_path = tmp_path / "experiment.yaml"
    lines = [
        "name: package-repair-test",
        f"model: {(REPO_ROOT / 'configs/models/replay-repair-package.yaml').as_posix()}",
        "runner: repair",
        f"task: {task.as_posix()}",
        f"mutation: {Path(mutation).as_posix()}",
        "conditions:",
        *(f"  - {condition}" for condition in conditions),
        "n_attempts: 1",
        "seed: 1729",
        "randomize_order: false",
        "container_image: invariantlab/package-candidate:py3.12",
    ]
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def _dry_run(config_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(REPO_ROOT)
    return RUNNER.invoke(
        app,
        ["run", "--experiment", str(config_path), "--dry-run"],
    )


def test_package_and_legacy_configs_report_their_grading_paths(monkeypatch):
    package = _dry_run(PACKAGE_CONFIG, monkeypatch)
    legacy = _dry_run(
        REPO_ROOT / "configs/experiments/update-order-feedback-replication.yaml",
        monkeypatch,
    )

    assert package.exit_code == 0, package.output
    assert "package path" in package.output
    assert legacy.exit_code == 0, legacy.output
    assert "legacy_study path" in legacy.output


def test_dry_run_reports_unknown_task(tmp_path, monkeypatch):
    unknown_task = tmp_path / "unknown-task"
    unknown_task.mkdir()
    result = _dry_run(
        _write_experiment(
            tmp_path,
            task=unknown_task,
            mutation="missing-mutant",
        ),
        monkeypatch,
    )

    assert result.exit_code == 1
    assert "unknown-task" in result.output


def test_dry_run_reports_unknown_mutant(tmp_path, monkeypatch):
    result = _dry_run(
        _write_experiment(
            tmp_path,
            task=TASKS_ROOT / "oscillator",
            mutation="unknown-mutant",
        ),
        monkeypatch,
    )

    assert result.exit_code == 1
    assert "unknown-mutant" in result.output


def test_dry_run_rejects_metrics_for_package_mutants(tmp_path, monkeypatch):
    result = _dry_run(
        _write_experiment(
            tmp_path,
            task=TASKS_ROOT / "oscillator",
            mutation=TASKS_ROOT / "oscillator/mutations/non-conservative-damping",
            conditions=("metrics",),
        ),
        monkeypatch,
    )

    assert result.exit_code == 1
    assert "only the weak and placebo conditions" in result.output


@pytest.mark.parametrize(
    "mutant",
    [mutant for mutant in discover_mutants(TASKS_ROOT) if mutant.definition.interface == "package"],
    ids=lambda mutant: f"{mutant.task_dir.name}-{mutant.definition.id}",
)
def test_package_mutant_workspaces_and_prompts_exclude_reference_and_hidden_tests(
    tmp_path: Path,
    mutant,
) -> None:
    experiment = ExperimentConfig(
        name="package-prompt-test",
        model=str(REPO_ROOT / "configs/models/replay-repair-package.yaml"),
        task=str(mutant.task_dir),
        mutation=str(mutant.mutation_dir),
        conditions=["weak"],
    )
    assets = repair._resolve_assets(experiment)
    workspace = repair.build_repair_workspace(
        mutant.task_dir,
        mutant.contract,
        mutant.source_path,
        tmp_path / "workspace",
    )
    prompt = repair._render_prompt(assets, "")
    reference = (mutant.task_dir / mutant.contract.entrypoint).read_text(encoding="utf-8")

    assert (workspace / mutant.contract.entrypoint).read_text(encoding="utf-8") == (
        mutant.source_path.read_text(encoding="utf-8")
    )
    assert not any(
        path.is_file()
        and ("tests" in path.relative_to(workspace).parts)
        and ("scientific" in path.relative_to(workspace).parts)
        for path in workspace.rglob("*")
    )
    assert assets.mutation_source in prompt
    assert reference not in prompt
    for hidden_test in (mutant.task_dir / "tests/scientific").rglob("*"):
        if hidden_test.is_file():
            assert hidden_test.name not in prompt
            assert hidden_test.read_text(encoding="utf-8") not in prompt


@pytest.mark.parametrize("task_name", ("oscillator", "kepler", "heat1d", "wave1d"))
def test_package_prompt_contains_contract_specification_and_solver(
    tmp_path: Path,
    task_name: str,
) -> None:
    task_dir = TASKS_ROOT / task_name
    contract = load_task_contract(task_dir)
    workspace = build_agent_workspace(task_dir, tmp_path / "agent")
    specification = (workspace / "specification.md").read_text(encoding="utf-8")
    source = (workspace / contract.entrypoint).read_text(encoding="utf-8")

    prompt = repair.render_package_prompt(specification, source)

    assert specification in prompt
    assert source in prompt
    assert all(
        placeholder not in prompt
        for placeholder in ("{specification}", "{source}", "{condition_context}")
    )


PACKAGE_REPLAY_CASES = [
    ("oscillator", "non-conservative-damping"),
    ("wave1d", "sign-error-startup"),
]


# Kepler and heat1d package mutants are tracked by issues #90 and #92.
@pytest.mark.parametrize(("task_name", "mutation_id"), PACKAGE_REPLAY_CASES)
def test_package_replay_run_uses_docker_executor_and_verifies_all_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_name: str,
    mutation_id: str,
) -> None:
    calls: list[list[str]] = []
    container_results: list[tuple[list[str], subprocess.CompletedProcess[str]]] = []

    def fake_run_container(
        image: str,
        mounts: list[str],
        args: list[str],
        *,
        timeout: float = repair.CANDIDATE_TIMEOUT_SECONDS,
        workdir: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del image
        host_mounts: dict[str, Path] = {}
        for mount in mounts:
            host, target, _mode = mount.rsplit(":", 2)
            host_path = Path(host)
            host_mounts[target] = host_path
            for path in host_path.rglob("*"):
                relative = path.relative_to(host_path).parts
                assert not (path.is_file() and "tests" in relative and "scientific" in relative)

        rewritten_args = list(args)
        for target, host_path in sorted(
            host_mounts.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            rewritten: list[str] = []
            for argument in rewritten_args:
                if argument == target:
                    rewritten.append(str(host_path))
                elif argument.startswith(f"{target}/"):
                    relative = argument[len(target) + 1 :].replace("/", "\\")
                    rewritten.append(str(host_path / relative))
                elif argument.endswith(f"={target}"):
                    rewritten.append(f"{argument[: -len(target)]}{host_path}")
                else:
                    rewritten.append(argument)
            rewritten_args = rewritten
        assert workdir is not None
        calls.append(list(args))
        completed = subprocess.run(
            [sys.executable, *rewritten_args],
            cwd=host_mounts[workdir],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        container_results.append((args, completed))
        return completed

    monkeypatch.setattr(repair, "_resolve_image_digest", lambda image: "local@sha256:" + "a" * 64)
    monkeypatch.setattr(repair, "_git_state", lambda: ("0" * 40, False))
    monkeypatch.setattr(repair, "_run_container", fake_run_container)

    if task_name == "oscillator":
        config_path = PACKAGE_CONFIG
    else:
        config_path = tmp_path / "wave1d-experiment.yaml"
        config_path.write_text(
            PACKAGE_CONFIG.read_text(encoding="utf-8")
            .replace("task: tasks/oscillator", f"task: tasks/{task_name}")
            .replace(
                "mutation: tasks/oscillator/mutations/non-conservative-damping",
                f"mutation: tasks/{task_name}/mutations/{mutation_id}",
            ),
            encoding="utf-8",
        )

    output = repair.run_repair_experiment(config_path, tmp_path / f"run-{task_name}")

    status = json.loads((output / "run-status.json").read_text(encoding="utf-8"))
    record = json.loads((output / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    baseline = record["baseline"]
    repaired = record["repaired"]

    assert status["status"] == "complete"
    assert len((output / "events.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert baseline["public_passed"] is True
    assert baseline["scientific_passed"] is False
    assert any(not gate["passed"] for gate in baseline["gates"])
    assert repaired["public_passed"] is True
    assert repaired["scientific_passed"] is True, [
        {
            "args": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        for args, completed in container_results
    ]
    assert {gate["layer"] for gate in repaired["gates"]} == {
        "L0",
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "L6",
    }
    assert all({"id", "passed", "detail"} <= gate.keys() for gate in repaired["gates"])
    assert any(args and args[0] == "/work/src/solver.py" for args in calls)
    assert any("-m" in args and "pytest" in args for args in calls)
