"""Unit tests for InvariantLab."""

from invariantlab.schema import TaskContract, TaskFamily


def test_schema_imports():
    """Schema module imports cleanly."""
    assert TaskFamily.OSCILLATOR == "oscillator"


def test_task_contract_requires_id():
    """TaskContract requires an id field."""
    import pytest

    with pytest.raises(Exception):  # noqa: B017
        TaskContract(family=TaskFamily.OSCILLATOR)  # type: ignore[call-arg]
