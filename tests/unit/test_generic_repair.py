"""Tests for the config-driven generic repair experiment."""

from invariantlab.config import ExperimentConfig
from invariantlab.experiments.generic_repair import (
    _condition_context,
    _schedule,
    _validate,
)
from invariantlab.schema import (
    MutationFamily,
    RepairExperimentSpec,
    RepairMutationSpec,
    RepairTaskSpec,
    TaskFamily,
)


def _task() -> RepairTaskSpec:
    return RepairTaskSpec(
        id="oscillator-verlet",
        family=TaskFamily.OSCILLATOR,
        description="Repair an oscillator solver",
        verifier="tasks/oscillator/scientific_verifier.py",
        function_name="solve_oscillator_verlet",
        metric_keys=["max_state_relative_error", "max_energy_relative_drift"],
    )


def _mutation() -> RepairMutationSpec:
    return RepairMutationSpec(
        id="oscillator-update-order",
        task="oscillator-verlet",
        family=MutationFamily.UPDATE_ORDER_ERROR,
        source="tasks/oscillator/mutations/update_order_solver.py",
        location="second half-step",
        expected_effect="energy drift",
    )


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        name="generic-test",
        task="tasks/oscillator/repair-task.yaml",
        mutation_config="tasks/oscillator/mutations/update-order.yaml",
        model="configs/models/ollama-qwen2.5-coder-7b.yaml",
        runner="generic_repair",
        conditions=["weak", "metrics"],
        n_attempts=2,
        seed=1729,
        randomize_order=True,
    )


def test_repair_experiment_types_compose():
    spec = RepairExperimentSpec(
        task=_task(),
        mutation=_mutation(),
        model="example-model",
        conditions=["weak", "metrics"],
        n_attempts=2,
    )
    assert spec.task.id == spec.mutation.task
    assert spec.mutation.family is MutationFamily.UPDATE_ORDER_ERROR


def test_schedule_contains_each_condition_trial_once():
    schedule = _schedule(_config())
    assert len(schedule) == 4
    assert set(schedule) == {
        ("weak", 1),
        ("weak", 2),
        ("metrics", 1),
        ("metrics", 2),
    }


def test_condition_context_exposes_metrics_only_when_requested():
    baseline = {
        "metrics": {
            "max_state_relative_error": 0.02,
            "max_energy_relative_drift": 0.04,
        }
    }
    assert _condition_context("weak", _task(), baseline) == ""
    metrics = _condition_context("metrics", _task(), baseline)
    assert "max_state_relative_error" in metrics
    interpreted = _condition_context("interpreted", _task(), baseline)
    assert "scientific acceptance threshold" in interpreted


def test_validate_rejects_task_mutation_mismatch():
    mutation = _mutation().model_copy(update={"task": "different-task"})
    try:
        _validate(_config(), _task(), mutation)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("Expected task/mutation mismatch to fail")
