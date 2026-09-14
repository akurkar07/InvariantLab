"""Tests for safe, on-disk task artifact validation."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
import yaml

from invariantlab.schema import TaskContract
from invariantlab.tasks.validation import validate_task_artifacts

if TYPE_CHECKING:
    from pathlib import Path

TASK_ID = "fixture_task"
PATH_FIELDS = ("entrypoint", "public_tests", "scientific_tests")


def _contract_data(**overrides: str) -> dict[str, object]:
    data: dict[str, object] = {
        "id": TASK_ID,
        "family": "oscillator",
        "language": "python",
        "entrypoint": "src/solver.py",
        "public_tests": "tests/public",
        "scientific_tests": "tests/scientific",
    }
    data.update(overrides)
    return data


def _make_task(task_root: Path, **overrides: str) -> TaskContract:
    (task_root / "src").mkdir(parents=True)
    (task_root / "tests" / "public").mkdir(parents=True)
    (task_root / "tests" / "scientific").mkdir(parents=True)
    (task_root / "src" / "solver.py").write_text("pass\n", encoding="utf-8")
    (task_root / "tests" / "public" / "test_solver.py").write_text("pass\n", encoding="utf-8")
    (task_root / "tests" / "scientific" / "test_reference.py").write_text(
        "pass\n", encoding="utf-8"
    )
    data = _contract_data(**overrides)
    (task_root / "contract.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return TaskContract(**data)


def _validate(task_root: Path, **overrides: str) -> list[str]:
    contract = _make_task(task_root, **overrides)
    return validate_task_artifacts(task_root, contract)


def test_valid_task_artifacts_pass_in_temporary_package(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    assert validate_task_artifacts(task_root, _make_task(task_root)) == []


@pytest.mark.parametrize("field", PATH_FIELDS)
@pytest.mark.parametrize(
    "declared_path",
    [
        "/etc/passwd",
        r"C:\Windows\System32",
        "C:/Windows/System32",
        r"\\server\share",
        "../outside",
        r"..\outside",
        "nested/../../outside",
        r"nested\..\..\outside",
    ],
)
def test_rejects_absolute_and_traversing_paths_across_platform_syntax(
    tmp_path: Path, field: str, declared_path: str
) -> None:
    errors = _validate(tmp_path / "task", **{field: declared_path})

    assert errors == [f"task '{TASK_ID}': field '{field}' must be a safe relative path"]


@pytest.mark.parametrize(
    ("field", "declared_path", "message"),
    [
        ("entrypoint", "src/missing.py", "does not exist"),
        ("public_tests", "tests/missing-public", "does not exist"),
        ("scientific_tests", "tests/missing-scientific", "does not exist"),
        ("entrypoint", "tests/public", "must be a regular file"),
        ("public_tests", "src/solver.py", "must be a directory"),
        ("scientific_tests", "src/solver.py", "must be a directory"),
    ],
)
def test_rejects_missing_and_wrong_artifact_types(
    tmp_path: Path, field: str, declared_path: str, message: str
) -> None:
    errors = _validate(tmp_path / "task", **{field: declared_path})

    assert errors == [f"task '{TASK_ID}': field '{field}' {message}"]


@pytest.mark.parametrize("field", ("public_tests", "scientific_tests"))
def test_rejects_test_directories_without_discoverable_tests(tmp_path: Path, field: str) -> None:
    task_root = tmp_path / "task"
    contract = _make_task(task_root)
    directory = task_root / getattr(contract, field)
    for test_file in directory.rglob("test_*.py"):
        test_file.unlink()

    errors = validate_task_artifacts(task_root, contract)

    assert errors == [f"task '{TASK_ID}': field '{field}' does not contain discoverable tests"]


@pytest.mark.parametrize("field", PATH_FIELDS)
def test_rejects_symlink_escape_from_task_root(tmp_path: Path, field: str) -> None:
    task_root = tmp_path / "task"
    outside = tmp_path / "outside"
    outside.mkdir()
    if field == "entrypoint":
        target = outside / "solver.py"
        target.write_text("pass\n", encoding="utf-8")
    else:
        target = outside / field
        target.mkdir()
        (target / "test_outside.py").write_text("pass\n", encoding="utf-8")
    link = task_root / "escape"
    task_root.mkdir()
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    contract = _make_task(task_root, **{field: "escape"})
    errors = validate_task_artifacts(task_root, contract)

    assert errors == [f"task '{TASK_ID}': field '{field}' escapes the task root"]


@pytest.mark.parametrize("field", ("public_tests", "scientific_tests"))
def test_accepts_pytest_suffix_test_modules(tmp_path: Path, field: str) -> None:
    task_root = tmp_path / "task"
    contract = _make_task(task_root)
    directory = task_root / getattr(contract, field)
    for test_file in directory.rglob("test_*.py"):
        test_file.unlink()
    (directory / "solver_test.py").write_text("def test_example(): pass\n", encoding="utf-8")

    assert validate_task_artifacts(task_root, contract) == []


@pytest.mark.parametrize("field", ("public_tests", "scientific_tests"))
def test_rejects_symlinked_test_file_escape(tmp_path: Path, field: str) -> None:
    task_root = tmp_path / "task"
    contract = _make_task(task_root)
    directory = task_root / getattr(contract, field)
    for test_file in directory.rglob("test_*.py"):
        test_file.unlink()
    outside = tmp_path / "outside_test.py"
    outside.write_text("def test_outside(): pass\n", encoding="utf-8")
    link = directory / "test_outside.py"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    errors = validate_task_artifacts(task_root, contract)

    assert errors == [f"task '{TASK_ID}': field '{field}' contains a path that escapes the task root"]


@pytest.mark.parametrize("field", ("public_tests", "scientific_tests"))
@pytest.mark.parametrize("name", ("test_data.txt", "test_helper"))
def test_rejects_non_python_files_that_only_resemble_test_modules(
    tmp_path: Path, field: str, name: str
) -> None:
    task_root = tmp_path / "task"
    contract = _make_task(task_root)
    directory = task_root / getattr(contract, field)
    for test_file in directory.rglob("test_*.py"):
        test_file.unlink()
    (directory / name).write_text("not a Python test\n", encoding="utf-8")

    errors = validate_task_artifacts(task_root, contract)

    assert errors == [f"task '{TASK_ID}': field '{field}' does not contain discoverable tests"]


@pytest.mark.parametrize("field", ("public_tests", "scientific_tests"))
def test_rejects_test_directory_link_cycles(tmp_path: Path, field: str) -> None:
    task_root = tmp_path / "task"
    contract = _make_task(task_root)
    directory = task_root / getattr(contract, field)
    loop = directory / "loop"
    try:
        loop.symlink_to(directory, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    errors = validate_task_artifacts(task_root, contract)

    assert errors == [f"task '{TASK_ID}': field '{field}' contains a directory cycle"]


def test_reports_multiple_invalid_artifacts_with_task_id_and_field(tmp_path: Path) -> None:
    errors = _validate(
        tmp_path / "task",
        entrypoint="src/missing.py",
        public_tests="tests/missing-public",
        scientific_tests="src/solver.py",
    )

    assert errors == [
        f"task '{TASK_ID}': field 'entrypoint' does not exist",
        f"task '{TASK_ID}': field 'public_tests' does not exist",
        f"task '{TASK_ID}': field 'scientific_tests' must be a directory",
    ]


def test_cli_distinguishes_schema_and_artifact_errors(tmp_path: Path) -> None:
    schema_root = tmp_path / "schema"
    schema_task = schema_root / "bad-schema"
    schema_task.mkdir(parents=True)
    (schema_task / "contract.yaml").write_text("id: bad\nfamily: invalid\n", encoding="utf-8")

    artifact_root = tmp_path / "artifact"
    artifact_task = artifact_root / "bad-artifact"
    _make_task(artifact_task, entrypoint="src/missing.py")

    schema = subprocess.run(
        [sys.executable, "scripts/validate_task.py", "--task-dir", str(schema_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    artifact = subprocess.run(
        [sys.executable, "scripts/validate_task.py", "--task-dir", str(artifact_root)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert schema.returncode != 0
    assert "schema error" in schema.stderr.lower()
    assert "field 'entrypoint'" not in schema.stderr
    assert artifact.returncode != 0
    assert f"task '{TASK_ID}': field 'entrypoint' does not exist" in artifact.stderr


def test_cli_returns_zero_for_valid_package_and_nonzero_for_artifact_error(tmp_path: Path) -> None:
    valid_root = tmp_path / "valid"
    _make_task(valid_root / "task")
    invalid_root = tmp_path / "invalid"
    _make_task(invalid_root / "task", public_tests="tests/missing-public")

    valid = subprocess.run(
        [sys.executable, "scripts/validate_task.py", "--task-dir", str(valid_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    invalid = subprocess.run(
        [sys.executable, "scripts/validate_task.py", "--task-dir", str(invalid_root)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert valid.returncode == 0
    assert "All task contracts and artifacts are valid." in valid.stdout
    assert invalid.returncode != 0
    assert "field 'public_tests' does not exist" in invalid.stderr
