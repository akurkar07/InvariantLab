"""Unit tests for the generic repair experiment architecture."""

import json
import subprocess
from pathlib import Path

import pytest

from invariantlab.config import ExperimentConfig, load_model_config
from invariantlab.experiments import repair
from invariantlab.experiments.repair import (
    _audit_records,
    _build_schedule,
    _condition_context,
    _resolve_assets,
    run_repair_experiment,
)
from invariantlab.models import ReplayMissError, resolve_model_id
from invariantlab.schema import (
    load_mutation_definition,
    load_task_definition,
)


@pytest.fixture(autouse=True)
def _no_docker_provenance(monkeypatch):
    monkeypatch.setattr(
        repair, "_resolve_image_digest", lambda image: f"python@sha256:{'a' * 64}"
    )
    monkeypatch.setattr(repair, "_git_state", lambda: ("0" * 40, False))


def test_loads_config_driven_task_and_mutation():
    task = load_task_definition("tasks/oscillator")
    mutation = load_mutation_definition(
        "tasks/oscillator/mutations/update-order"
    )

    assert task.id == "oscillator_verlet"
    assert task.verifier == "verifier.py"
    assert mutation.id == "update-order"
    assert mutation.task_id == task.id


def test_oscillator_study_gate_thresholds_stay_at_study_values():
    task = load_task_definition("tasks/oscillator")

    assert {
        name: spec.threshold for name, spec in task.feedback_metrics.items()
    } == {
        "max_state_relative_error": 1e-3,
        "max_energy_relative_drift": 1e-3,
    }
    assert "threshold of 1e-3" in task.interpreted_feedback


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
        model="configs/models/reference-stub-oscillator.yaml",
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
        model="configs/models/reference-stub-oscillator.yaml",
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


def _content_based_evaluator(source: str, *_args):
    if "v = v_half + 0.5 * dt * a_new" in source:
        return REPAIRED_RESULT
    return BASELINE_RESULT


def _write_replay_fixture_config(
    tmp_path: Path, n_attempts: int = 2, model: Path | None = None
) -> Path:
    model = model or (
        REPO_ROOT / "tests/fixtures/replay/oscillator-update-order/model.yaml"
    )
    config = tmp_path / "replay-fixture-experiment.yaml"
    config.write_text(
        "\n".join(
            [
                "name: replay-fixture-oscillator",
                f"task_suite: {(REPO_ROOT / 'configs/task-suites/v1-smoke.yaml').as_posix()}",
                f"model: {model.as_posix()}",
                "runner: repair",
                f"task: {(REPO_ROOT / 'tasks/oscillator').as_posix()}",
                "mutation: "
                f"{(REPO_ROOT / 'tasks/oscillator/mutations/update-order').as_posix()}",
                "conditions: [weak, metrics]",
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


def test_repair_runner_replays_fixture_events(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(repair, "_evaluate_source", _content_based_evaluator)
    config_path = _write_replay_fixture_config(tmp_path)
    output = tmp_path / "run"

    run_repair_experiment(config_path, output)

    events = _events(output)
    assert len(events) == 4
    assert [
        (event["condition"], event["trial"], event["successful_repair"])
        for event in events
    ] == [
        ("weak", 1, False),
        ("weak", 2, True),
        ("metrics", 1, True),
        ("metrics", 2, True),
    ]
    assert all(event["model"] == "replay/fixture/scripted-oscillator" for event in events)
    assert events[0]["usage"]["input_tokens"] == 100
    assert _status(output)["status"] == "complete"
    assert repair.audit_repair_experiment(config_path, output)["integrity_ok"] is True


def test_repair_runner_fails_when_fixture_responses_are_exhausted(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(repair, "_evaluate_source", _content_based_evaluator)

    with pytest.raises(ReplayMissError):
        run_repair_experiment(
            _write_replay_fixture_config(tmp_path, n_attempts=3),
            tmp_path / "run",
        )


def _write_repair_config(
    tmp_path: Path, n_attempts: int = 2, model: Path | None = None
) -> Path:
    model = model or REPO_ROOT / "configs/models/reference-stub-oscillator.yaml"
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join(
            [
                "name: timeout-infra-test",
                f"task_suite: {(REPO_ROOT / 'configs/task-suites/v1-smoke.yaml').as_posix()}",
                f"model: {model.as_posix()}",
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
    """Stand-in for ``subprocess.run`` that scripts two-stage ``docker run`` outcomes.

    Each outcome covers one evaluation (candidate stage then verifier stage); the
    first evaluation is the baseline. ``"timeout"`` and ``(125, stderr)`` apply to
    the candidate stage; ``(code, stderr)`` is a candidate-stage exit after which
    the verifier reports a failed verdict.
    """

    def __init__(self, candidate_outcomes: list[object]):
        self.candidate_outcomes = list(candidate_outcomes)
        self.run_names: list[str] = []
        self.candidate_stage_names: list[str] = []
        self.killed: list[str] = []
        self.baseline_outcome: object = BASELINE_RESULT
        self._current: object = None

    def __call__(self, args, **kwargs):
        if args[:2] == ["docker", "kill"]:
            self.killed.append(args[2])
            return subprocess.CompletedProcess(args, 0, "", "")
        assert args[:2] == ["docker", "run"]
        name = args[args.index("--name") + 1]
        self.run_names.append(name)
        if "/work/candidate_runner.py" in args:
            self.candidate_stage_names.append(name)
            if len(self.candidate_stage_names) == 1:
                self._current = self.baseline_outcome
            else:
                self._current = self.candidate_outcomes.pop(0)
            if self._current == "timeout":
                raise subprocess.TimeoutExpired(args, kwargs["timeout"])
            if isinstance(self._current, tuple):
                code, stderr = self._current
                return subprocess.CompletedProcess(args, code, "", stderr)
            return subprocess.CompletedProcess(args, 0, "", "")
        assert "/verifier/verifier.py" in args
        if isinstance(self._current, tuple):
            verdict = {
                "public": {},
                "scientific": {},
                "public_passed": False,
                "scientific_passed": False,
                "metrics": {"error": "missing short trajectory"},
            }
        else:
            verdict = self._current
        return subprocess.CompletedProcess(args, 0, json.dumps(verdict) + "\n", "")


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
    assert fake.killed == [fake.candidate_stage_names[1]]
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
    assert failed["repaired"]["metrics"] == {"error": "missing short trajectory"}
    assert fake.killed == []
    assert _status(output)["status"] == "complete"


def test_repair_runner_records_usage_and_finish_reason(tmp_path, monkeypatch):
    from invariantlab.experiments import repair
    from invariantlab.models import ModelResponse

    class FakeAdapter:
        model_id = "reference_stub/oscillator-verlet"

        def generate(self, prompt: str) -> str:
            return self.complete(prompt).text

        def complete(self, prompt: str) -> ModelResponse:
            del prompt
            return ModelResponse(
                text="```python\nprint('fixed')\n```",
                input_tokens=321,
                output_tokens=54,
                finish_reason="stop",
            )

    evaluations = iter(
        [
            {"public_passed": True, "scientific_passed": False, "metrics": {}},
            {"public_passed": True, "scientific_passed": True, "metrics": {}},
        ]
    )
    monkeypatch.setattr(repair, "build_adapter", lambda config: FakeAdapter())
    monkeypatch.setattr(repair, "_evaluate_source", lambda *args: next(evaluations))
    config_path = _write_repair_config(tmp_path, n_attempts=1)
    output = tmp_path / "run"

    repair.run_repair_experiment(config_path, output)

    lines = (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    assert record["usage"] == {"input_tokens": 321, "output_tokens": 54}
    assert record["finish_reason"] == "stop"

    audit = repair.audit_repair_experiment(config_path, output)
    assert audit["integrity_ok"] is True


def test_audit_resolves_replay_model_id_like_runner(tmp_path, monkeypatch):
    events_path = (
        REPO_ROOT / "tests/fixtures/replay/oscillator-update-order/events.jsonl"
    ).as_posix()
    model = tmp_path / "model.yaml"
    model.write_text(
        f'adapter: replay\nextra:\n  events_path: "{events_path}"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(repair, "_evaluate_source", _content_based_evaluator)
    config_path = _write_replay_fixture_config(tmp_path, model=model)
    output = tmp_path / "run"

    repair.run_repair_experiment(config_path, output)

    expected = resolve_model_id(load_model_config(model))
    assert _events(output)[0]["model"] == expected
    assert expected.startswith("replay/")
    audit = repair.audit_repair_experiment(config_path, output)
    assert audit["integrity_ok"] is True
    assert audit["metadata_mismatches"] == []


def test_audit_run_missing_run_dir_fails_without_writing(tmp_path):
    from typer.testing import CliRunner

    from invariantlab.cli import app

    config_path = _write_repair_config(tmp_path, n_attempts=1)
    missing = tmp_path / "missing-run"

    result = CliRunner().invoke(
        app,
        ["audit-run", "--experiment", str(config_path), "--run-dir", str(missing)],
    )

    assert result.exit_code != 0
    assert "Run directory does not exist" in result.output
    assert "missing-run" in result.output
    assert not missing.exists()
