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


def _contract_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": TASK_ID,
        "family": "oscillator",
        "language": "python",
        "entrypoint": "src/solver.py",
        "public_tests": "tests/public",
        "scientific_tests": "tests/scientific",
        "output": {
            "path": "result.npz",
            "arrays": [{"name": "state", "shape": [None], "dtype": "float64"}],
        },
    }
    data.update(overrides)
    return data


def _make_task(task_root: Path, **overrides: object) -> TaskContract:
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


def _validate(task_root: Path, **overrides: object) -> list[str]:
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
    "declared_path",
    [
        "/tmp/result.npz",
        "../result.npz",
        ".." + chr(92) + "result.npz",
        "C:" + chr(92) + "temp" + chr(92) + "result.npz",
        chr(92) * 2 + "server" + chr(92) + "share" + chr(92) + "result.npz",
    ],
)
def test_rejects_unsafe_output_paths(tmp_path: Path, declared_path: str) -> None:
    errors = _validate(
        tmp_path / "task",
        output={
            "path": declared_path,
            "arrays": [{"name": "state", "shape": [None], "dtype": "float64"}],
        },
    )

    assert errors == [f"task '{TASK_ID}': field 'output.path' must be a safe relative path"]


def test_rejects_output_path_escape_through_link(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    outside = tmp_path / "outside"
    outside.mkdir()
    task_root.mkdir()
    try:
        (task_root / "escape").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    contract = _make_task(
        task_root,
        output={
            "path": "escape/result.npz",
            "arrays": [{"name": "state", "shape": [None], "dtype": "float64"}],
        },
    )

    assert validate_task_artifacts(task_root, contract) == [
        f"task '{TASK_ID}': field 'output.path' must be a safe relative path"
    ]


def test_rejects_output_path_that_is_an_existing_directory(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    contract = _make_task(task_root)
    (task_root / contract.output.path).mkdir()

    assert validate_task_artifacts(task_root, contract) == [
        f"task '{TASK_ID}': field 'output.path' must not be a directory"
    ]


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
    schema_task = schema_root / "oscillator"
    schema_task.mkdir(parents=True)
    (schema_task / "contract.yaml").write_text("id: bad\nfamily: invalid\n", encoding="utf-8")

    artifact_root = tmp_path / "artifact"
    artifact_task = artifact_root / "oscillator"
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


def _run_validation_cli(tasks_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/validate_task.py", "--task-dir", str(tasks_root)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_root_validation_requires_every_fixed_v1_package(tmp_path: Path) -> None:
    result = _run_validation_cli(tmp_path / "tasks")

    assert result.returncode != 0
    for task_name in ("oscillator", "kepler", "heat1d", "wave1d"):
        assert f"required V1 task package '{task_name}' does not exist" in result.stderr


def test_root_validation_requires_contract_and_specification_for_fixed_packages(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    for task_name in ("oscillator", "kepler", "heat1d", "wave1d"):
        (tasks_root / task_name).mkdir(parents=True)

    result = _run_validation_cli(tasks_root)

    assert result.returncode != 0
    for task_name in ("oscillator", "kepler", "heat1d", "wave1d"):
        assert f"task '{task_name}': field 'contract.yaml' does not exist" in result.stderr
        assert f"task '{task_name}': field 'specification.md' does not exist" in result.stderr
