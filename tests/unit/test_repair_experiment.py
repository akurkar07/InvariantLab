"""Unit tests for the generic repair experiment architecture."""

import json
import subprocess
from pathlib import Path

from invariantlab.config import ExperimentConfig
from invariantlab.experiments import repair
from invariantlab.experiments.repair import (
    _audit_records,
    _build_schedule,
    _condition_context,
    _resolve_assets,
    run_repair_experiment,
)
from invariantlab.schema import (
    load_mutation_definition,
    load_task_definition,
)


def test_loads_config_driven_task_and_mutation():
    task = load_task_definition("tasks/oscillator")
    mutation = load_mutation_definition(
        "tasks/oscillator/mutations/update-order"
    )

    assert task.id == "oscillator_verlet"
    assert task.verifier == "verifier.py"
    assert mutation.id == "update-order"
    assert mutation.task_id == task.id


def test_task_feedback_metadata_drives_condition_context():
    task = load_task_definition("tasks/oscillator")
    baseline = {
        "metrics": {
            "max_state_relative_error": 0.021,
            "max_energy_relative_drift": 0.046,
        }
    }

    metrics = _condition_context("metrics", baseline, task)
    interpreted = _condition_context("interpreted", baseline, task)

    assert "max state relative error: 0.021" in metrics
    assert "max energy relative drift: 0.046" in metrics
    assert "acceptance threshold" not in metrics
    assert "acceptance threshold" in interpreted


def test_repair_assets_are_resolved_from_experiment_config():
    experiment = ExperimentConfig(
        name="generic-repair-test",
        task_suite="configs/task-suites/v1-smoke.yaml",
        model="configs/models/replay-first-model.yaml",
        runner="repair",
        task="tasks/oscillator",
        mutation="tasks/oscillator/mutations/update-order",
        conditions=["weak"],
        n_attempts=1,
        seed=1729,
    )

    task_dir, task, mutation_dir, mutation, source, prompt = _resolve_assets(
        experiment
    )

    assert task_dir == Path("tasks/oscillator")
    assert mutation_dir == Path("tasks/oscillator/mutations/update-order")
    assert task.id == "oscillator_verlet"
    assert mutation.id == "update-order"
    assert "v = v_half + 0.5 * dt * a" in source
    assert "{condition_context}" in prompt
    assert "{source}" in prompt


def test_path_based_config_audits_legacy_study_two_mutation_id():
    experiment = ExperimentConfig(
        name="legacy-study-two",
        task_suite="configs/task-suites/v1-smoke.yaml",
        model="configs/models/replay-first-model.yaml",
        runner="repair",
        task="tasks/oscillator",
        mutation="tasks/oscillator/mutations/update-order",
        conditions=["weak"],
        n_attempts=1,
        seed=1729,
    )
    schedule = _build_schedule(["weak"], 1, 1729, False)
    records = [
        {
            "experiment": "legacy-study-two",
            "model": "test/model",
            "mutation": "update-order",
            "seed": 1729,
            "condition": "weak",
            "trial": 1,
            "successful_repair": True,
            "scientific_regression": False,
            "severity": {"worst_scientific_ratio": 0.1},
        }
    ]

    canonical, audit = _audit_records(
        records,
        schedule,
        experiment,
        "test/model",
    )

    assert canonical == records
    assert audit["integrity_ok"] is True
    assert audit["complete"] is True


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_RESULT = {
    "public": {},
    "scientific": {},
    "public_passed": True,
    "scientific_passed": False,
    "metrics": {
        "max_state_relative_error": 0.02,
        "max_energy_relative_drift": 0.04,
    },
}
REPAIRED_RESULT = {
    "public": {},
    "scientific": {},
    "public_passed": True,
    "scientific_passed": True,
    "metrics": {
        "max_state_relative_error": 1e-6,
        "max_energy_relative_drift": 1e-6,
    },
}


def _write_repair_config(tmp_path: Path, n_attempts: int = 2) -> Path:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join(
            [
                "name: timeout-infra-test",
                f"task_suite: {(REPO_ROOT / 'configs/task-suites/v1-smoke.yaml').as_posix()}",
                f"model: {(REPO_ROOT / 'configs/models/replay-first-model.yaml').as_posix()}",
                "runner: repair",
                f"task: {(REPO_ROOT / 'tasks/oscillator').as_posix()}",
                "mutation: "
                f"{(REPO_ROOT / 'tasks/oscillator/mutations/update-order').as_posix()}",
                "conditions: [weak]",
                f"n_attempts: {n_attempts}",
                "seed: 1729",
                "randomize_order: false",
                "container_image: python:3.12-slim",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config


class FakeDocker:
    """Stand-in for ``subprocess.run`` that scripts ``docker run`` outcomes."""

    def __init__(self, candidate_outcomes: list[object]):
        self.candidate_outcomes = list(candidate_outcomes)
        self.run_names: list[str] = []
        self.killed: list[str] = []
        self.baseline_outcome: object = BASELINE_RESULT

    def __call__(self, args, **kwargs):
        if args[:2] == ["docker", "kill"]:
            self.killed.append(args[2])
            return subprocess.CompletedProcess(args, 0, "", "")
        assert args[:2] == ["docker", "run"]
        name = args[args.index("--name") + 1]
        self.run_names.append(name)
        if len(self.run_names) == 1:
            outcome = self.baseline_outcome
        else:
            outcome = self.candidate_outcomes.pop(0)
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        if isinstance(outcome, tuple):
            code, stderr = outcome
            return subprocess.CompletedProcess(args, code, "", stderr)
        return subprocess.CompletedProcess(args, 0, json.dumps(outcome) + "\n", "")


def _install_fake_docker(monkeypatch, fake: FakeDocker) -> None:
    monkeypatch.setattr(repair.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(repair.subprocess, "run", fake)


def _events(output: Path) -> list[dict]:
    path = output / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _status(output: Path) -> dict:
    return json.loads((output / "run-status.json").read_text(encoding="utf-8"))


def test_candidate_timeout_is_recorded_and_container_killed(tmp_path, monkeypatch):
    fake = FakeDocker(["timeout", REPAIRED_RESULT])
    _install_fake_docker(monkeypatch, fake)
    output = tmp_path / "run"

    run_repair_experiment(_write_repair_config(tmp_path), output)

    events = _events(output)
    assert len(events) == 2
    timed_out = events[0]
    assert timed_out["timeout"] is True
    assert timed_out["repaired"]["public_passed"] is False
    assert timed_out["repaired"]["scientific_passed"] is False
    assert "timed out" in timed_out["candidate_error"]
    assert events[1]["timeout"] is False
    assert events[1]["successful_repair"] is True
    assert fake.killed == [fake.run_names[1]]
    assert len(set(fake.run_names)) == len(fake.run_names)
    assert _status(output)["status"] == "complete"


def test_docker_exit_125_pauses_without_record_and_resumes(tmp_path, monkeypatch):
    config = _write_repair_config(tmp_path)
    output = tmp_path / "run"
    fake = FakeDocker([(125, "toomanyrequests: You have reached your pull rate limit")])
    _install_fake_docker(monkeypatch, fake)

    run_repair_experiment(config, output)

    status = _status(output)
    assert status["status"] == "paused_infrastructure"
    assert "toomanyrequests" in status["reason"]
    assert status["completed_cells"] == 0
    assert _events(output) == []

    _install_fake_docker(monkeypatch, FakeDocker([REPAIRED_RESULT, REPAIRED_RESULT]))
    run_repair_experiment(config, output)

    assert _status(output)["status"] == "complete"
    assert len(_events(output)) == 2


def test_baseline_infrastructure_failure_pauses(tmp_path, monkeypatch):
    fake = FakeDocker([])
    fake.baseline_outcome = (125, "Cannot connect to the Docker daemon")
    _install_fake_docker(monkeypatch, fake)
    output = tmp_path / "run"

    run_repair_experiment(_write_repair_config(tmp_path), output)

    assert _status(output)["status"] == "paused_infrastructure"
    assert _events(output) == []


def test_missing_docker_binary_pauses(tmp_path, monkeypatch):
    monkeypatch.setattr(repair.shutil, "which", lambda _name: None)
    output = tmp_path / "run"

    run_repair_experiment(_write_repair_config(tmp_path), output)

    assert _status(output)["status"] == "paused_infrastructure"
    assert _events(output) == []


def test_candidate_exit_failure_is_recorded_as_failed_repair(tmp_path, monkeypatch):
    fake = FakeDocker([(1, "Traceback: ZeroDivisionError"), REPAIRED_RESULT])
    _install_fake_docker(monkeypatch, fake)
    output = tmp_path / "run"

    run_repair_experiment(_write_repair_config(tmp_path), output)

    events = _events(output)
    assert len(events) == 2
    failed = events[0]
    assert failed["timeout"] is False
    assert failed["successful_repair"] is False
    assert failed["repaired"]["public_passed"] is False
    assert "Candidate evaluation failed" in failed["candidate_error"]
    assert fake.killed == []
    assert _status(output)["status"] == "complete"
