"""Unit tests for the generic repair experiment architecture."""

import json
from pathlib import Path

from invariantlab.config import ExperimentConfig
from invariantlab.experiments.repair import (
    _audit_records,
    _build_schedule,
    _condition_context,
    _resolve_assets,
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


def _write_runner_config(tmp_path: Path) -> Path:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join(
            [
                "name: usage-recording",
                "task_suite: configs/task-suites/v1-smoke.yaml",
                "model: configs/models/replay-first-model.yaml",
                "runner: repair",
                "task: tasks/oscillator",
                "mutation: tasks/oscillator/mutations/update-order",
                "conditions: [weak]",
                "n_attempts: 1",
                "seed: 1729",
                "randomize_order: false",
                "container_image: python:3.12-slim",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def test_repair_runner_records_usage_and_finish_reason(tmp_path, monkeypatch):
    from invariantlab.experiments import repair
    from invariantlab.models import ModelResponse

    class FakeAdapter:
        model_id = "replay/oscillator-reference"

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
    config_path = _write_runner_config(tmp_path)
    output = tmp_path / "run"

    repair.run_repair_experiment(config_path, output)

    lines = (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    assert record["usage"] == {"input_tokens": 321, "output_tokens": 54}
    assert record["finish_reason"] == "stop"

    audit = repair.audit_repair_experiment(config_path, output)
    assert audit["integrity_ok"] is True
