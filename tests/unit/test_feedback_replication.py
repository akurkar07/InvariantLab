"""Tests for the update-order feedback replication study."""

import json

from invariantlab.config import ExperimentConfig
from invariantlab.experiments.feedback_replication import (
    CONDITIONS,
    _audit_records,
    _build_schedule,
    _condition_context,
    _severity_ratios,
    _write_run_status,
)

BASELINE = {
    "metrics": {
        "max_state_relative_error": 0.021,
        "max_energy_relative_drift": 0.046,
    }
}


def test_schedule_is_balanced_and_reproducible():
    first = _build_schedule(list(CONDITIONS), 30, 1729, True)
    second = _build_schedule(list(CONDITIONS), 30, 1729, True)

    assert first == second
    assert len(first) == 120
    for condition in CONDITIONS:
        assert sum(cell_condition == condition for cell_condition, _ in first) == 30


def test_schedule_randomisation_changes_grouped_order():
    grouped = _build_schedule(list(CONDITIONS), 3, 1729, False)
    randomised = _build_schedule(list(CONDITIONS), 3, 1729, True)

    assert grouped != randomised
    assert sorted(grouped) == sorted(randomised)


def test_feedback_conditions_are_distinct():
    contexts = {
        condition: _condition_context(condition, BASELINE)
        for condition in CONDITIONS
    }

    assert contexts["weak"] == ""
    assert "0.021" not in contexts["placebo"]
    assert "0.021" in contexts["metrics"]
    assert "acceptance threshold" not in contexts["metrics"]
    assert "acceptance threshold" in contexts["interpreted"]


def test_severity_ratios_capture_worse_repair():
    repaired = {
        "metrics": {
            "max_state_relative_error": 1.97,
            "max_energy_relative_drift": 8.0,
        }
    }

    ratios = _severity_ratios(BASELINE, repaired)

    assert ratios["state_error_ratio"] == 1.97 / 0.021
    assert ratios["energy_drift_ratio"] == 8.0 / 0.046
    assert ratios["worst_scientific_ratio"] == 8.0 / 0.046


def test_run_status_records_resumable_progress(tmp_path):
    _write_run_status(
        tmp_path,
        status="paused_rate_limit",
        completed=46,
        target=120,
        reason="rate limited",
    )

    payload = json.loads((tmp_path / "run-status.json").read_text(encoding="utf-8"))

    assert payload == {
        "completed_cells": 46,
        "reason": "rate limited",
        "status": "paused_rate_limit",
        "target_cells": 120,
    }


def _integrity_experiment() -> ExperimentConfig:
    return ExperimentConfig(
        name="integrity-test",
        task_suite="configs/task-suites/v1-smoke.yaml",
        model="configs/models/replay.yaml",
        runner="feedback_replication",
        mutation="update-order",
        conditions=["weak", "metrics"],
        n_attempts=2,
        seed=1729,
    )


def _record(condition: str, trial: int, *, model: str = "test/model"):
    return {
        "experiment": "integrity-test",
        "model": model,
        "mutation": "update-order",
        "seed": 1729,
        "condition": condition,
        "trial": trial,
        "successful_repair": True,
        "scientific_regression": False,
        "severity": {"worst_scientific_ratio": 0.1},
    }


def test_audit_accepts_exact_unique_schedule():
    experiment = _integrity_experiment()
    schedule = _build_schedule(
        list(experiment.conditions),
        experiment.n_attempts,
        experiment.seed,
        False,
    )
    records = [
        _record("weak", 1),
        _record("weak", 2),
        _record("metrics", 1),
        _record("metrics", 2),
    ]

    canonical, audit = _audit_records(
        records,
        schedule,
        experiment,
        "test/model",
    )

    assert len(canonical) == 4
    assert audit["integrity_ok"] is True
    assert audit["complete"] is True
    assert audit["raw_records"] == 4
    assert audit["canonical_cells"] == 4


def test_audit_rejects_duplicate_scheduled_cell():
    experiment = _integrity_experiment()
    schedule = _build_schedule(
        list(experiment.conditions),
        experiment.n_attempts,
        experiment.seed,
        False,
    )
    records = [
        _record("weak", 1),
        _record("weak", 1),
        _record("weak", 2),
    ]

    canonical, audit = _audit_records(
        records,
        schedule,
        experiment,
        "test/model",
    )

    assert len(canonical) == 2
    assert audit["integrity_ok"] is False
    assert audit["duplicate_records"] == 1
    assert audit["duplicate_keys"][0]["condition"] == "weak"
    assert audit["duplicate_keys"][0]["trial"] == 1


def test_audit_rejects_out_of_schedule_trial():
    experiment = _integrity_experiment()
    schedule = _build_schedule(
        list(experiment.conditions),
        experiment.n_attempts,
        experiment.seed,
        False,
    )
    records = [_record("weak", 3)]

    canonical, audit = _audit_records(
        records,
        schedule,
        experiment,
        "test/model",
    )

    assert canonical == []
    assert audit["integrity_ok"] is False
    assert audit["unexpected_records"] == [
        {"line": 1, "condition": "weak", "trial": 3}
    ]


def test_audit_rejects_mixed_model_record():
    experiment = _integrity_experiment()
    schedule = _build_schedule(
        list(experiment.conditions),
        experiment.n_attempts,
        experiment.seed,
        False,
    )
    records = [_record("weak", 1, model="other/model")]

    canonical, audit = _audit_records(
        records,
        schedule,
        experiment,
        "test/model",
    )

    assert canonical == []
    assert audit["integrity_ok"] is False
    assert audit["metadata_mismatches"][0]["fields"]["model"] == {
        "expected": "test/model",
        "actual": "other/model",
    }
