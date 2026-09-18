"""Unit tests for InvariantLab."""

from invariantlab.schema import (
    ExperimentDefinition,
    MutationDefinition,
    MutationFamily,
    TaskContract,
    TaskDefinition,
    TaskFamily,
)


def test_schema_imports():
    """Schema module imports cleanly."""
    assert TaskFamily.OSCILLATOR == "oscillator"


def test_task_contract_requires_id():
    """TaskContract requires an id field."""
    import pytest

    with pytest.raises(Exception):  # noqa: B017
        TaskContract(family=TaskFamily.OSCILLATOR)  # type: ignore[call-arg]


def test_generic_task_mutation_experiment_types():
    task = TaskDefinition(
        id="oscillator_verlet",
        family=TaskFamily.OSCILLATOR,
        verifier="verifier.py",
        prompt_template="repair_prompt.txt",
    )
    mutation = MutationDefinition(
        id="update-order",
        task_id=task.id,
        family=MutationFamily.UPDATE_ORDER_ERROR,
        source="solver.py",
    )
    experiment = ExperimentDefinition(
        task="tasks/oscillator",
        mutation="tasks/oscillator/mutations/update-order",
        model="configs/models/replay-first-model.yaml",
        conditions=["weak", "metrics"],
        n_attempts=2,
        seed=1729,
    )

    assert task.id == "oscillator_verlet"
    assert mutation.task_id == task.id
    assert experiment.conditions == ["weak", "metrics"]
