"""Unit tests for InvariantLab task-contract schemas."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from invariantlab.schema import TaskContract, TaskFamily, load_task_contract


def _contract_data() -> dict[str, object]:
    return {
        "id": "fixture_task",
        "family": "oscillator",
        "language": "python",
        "entrypoint": "src/solver.py",
        "public_tests": "tests/public",
        "scientific_tests": "tests/scientific",
        "output": {
            "path": "result.npz",
            "arrays": [
                {"name": "time", "shape": [None], "dtype": "float64"},
                {"name": "state", "shape": [None, 2], "dtype": "float64"},
            ],
        },
    }


def test_schema_imports() -> None:
    """Schema module imports cleanly."""
    assert TaskFamily.OSCILLATOR == "oscillator"


def test_task_contract_requires_id() -> None:
    """TaskContract requires an id field."""
    data = _contract_data()
    del data["id"]

    with pytest.raises(ValidationError):
        TaskContract(**data)


def test_load_task_contract_preserves_output_declaration(tmp_path: Path) -> None:
    """Output paths, names, dynamic shapes, and dtypes survive contract loading."""
    data = _contract_data()
    (tmp_path / "contract.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")

    contract = load_task_contract(tmp_path)

    assert contract.output.path == "result.npz"
    assert [(array.name, array.shape, array.dtype) for array in contract.output.arrays] == [
        ("time", [None], "float64"),
        ("state", [None, 2], "float64"),
    ]


@pytest.mark.parametrize(
    "output",
    [
        {"path": "result.npz", "arrays": [{"name": "state", "shape": [[None]], "dtype": "float64"}]},
        {"path": "result.npz", "arrays": [{"name": "state", "shape": [None], "dtype": "float32"}]},
        {"path": "result.npz", "arrays": [{"name": "", "shape": [None], "dtype": "float64"}]},
        {"path": "result.npz", "arrays": [{"name": "state", "shape": [0], "dtype": "float64"}]},
        {"path": "result.npz", "arrays": [{"name": " state ", "shape": [None], "dtype": "float64"}]},
        {"path": " result.npz ", "arrays": [{"name": "state", "shape": [None], "dtype": "float64"}]},
        {"path": ".", "arrays": [{"name": "state", "shape": [None], "dtype": "float64"}]},
        {"path": "result.txt", "arrays": [{"name": "state", "shape": [None], "dtype": "float64"}]},
        {"path": "result.npz", "arrays": []},
    ],
)
def test_rejects_malformed_output_declarations(output: dict[str, object]) -> None:
    """Output declarations require supported dtype and a non-empty flat shape."""
    data = _contract_data()
    data["output"] = output

    with pytest.raises(ValidationError):
        TaskContract(**data)


def test_rejects_duplicate_output_array_names() -> None:
    """NPZ key declarations must remain unambiguous."""
    data = _contract_data()
    output = data["output"]
    assert isinstance(output, dict)
    arrays = output["arrays"]
    assert isinstance(arrays, list)
    arrays[1] = {"name": "time", "shape": [None, 2], "dtype": "float64"}

    with pytest.raises(ValidationError, match="unique"):
        TaskContract(**data)


@pytest.mark.parametrize(
    "location",
    ("contract", "output", "array"),
)
def test_rejects_unknown_contract_fields(location: str) -> None:
    """Task and nested output declarations fail closed on unknown fields."""
    data = _contract_data()
    if location == "contract":
        data["unknown"] = "value"
    elif location == "output":
        output = data["output"]
        assert isinstance(output, dict)
        output["unknown"] = "value"
    else:
        output = data["output"]
        assert isinstance(output, dict)
        arrays = output["arrays"]
        assert isinstance(arrays, list)
        array = arrays[0]
        assert isinstance(array, dict)
        array["unknown"] = "value"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TaskContract(**data)


@pytest.mark.parametrize(
    ("task_name", "expected_arrays"),
    [
        ("oscillator", [("time", [None], "float64"), ("state", [None, 2], "float64")]),
        ("kepler", [("time", [None], "float64"), ("state", [None, 4], "float64")]),
        ("heat1d", [("x", [None], "float64"), ("state", [None], "float64")]),
        ("wave1d", [("x", [None], "float64"), ("state", [None], "float64")]),
    ],
)
def test_real_contracts_preserve_exact_output_declarations(
    task_name: str, expected_arrays: list[tuple[str, list[int | None], str]]
) -> None:
    """Every fixed V1 contract declares the NPZ archive it actually writes."""
    contract = load_task_contract(Path("tasks") / task_name)

    assert contract.output.path == "result.npz"
    assert [(array.name, array.shape, array.dtype) for array in contract.output.arrays] == expected_arrays
