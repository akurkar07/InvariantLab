"""Acceptance tests for InvariantLab."""

import subprocess
import sys
from pathlib import Path

from invariantlab.schema import load_task_contract
from invariantlab.tasks.validation import validate_task_artifacts

EXPECTED_TASKS = ("oscillator", "kepler", "heat1d", "wave1d")


def test_package_imports():
    """The invariantlab package can be imported."""
    import invariantlab

    assert invariantlab is not None


def test_cli_version():
    """The CLI version command runs without error."""
    result = subprocess.run(
        [sys.executable, "-m", "invariantlab.cli", "version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_validate_task_accepts_all_committed_task_packages():
    """All four task contracts pass the same artifact validation used by the CLI."""
    tasks_root = Path("tasks")
    for task_name in EXPECTED_TASKS:
        task_root = tasks_root / task_name
        assert task_root.is_dir(), f"missing committed task package: {task_name}"
        assert validate_task_artifacts(task_root, load_task_contract(task_root)) == []

    result = subprocess.run(
        [sys.executable, "scripts/validate_task.py", "--task-dir", "tasks/"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "all task contracts and artifacts are valid" in result.stdout.lower()


def test_fixed_v1_specifications_document_the_required_sections() -> None:
    """Every mandatory V1 package supplies a concise technical specification."""
    for task_name in EXPECTED_TASKS:
        specification = (Path("tasks") / task_name / "specification.md").read_text(encoding="utf-8")
        for heading in ("## Input parameters", "## Numerical method", "## Output archive"):
            assert heading in specification


def test_wave_contract_does_not_advertise_unsupported_energy_drift() -> None:
    """A final displacement archive cannot evidence a temporal energy-drift metric."""
    contract = load_task_contract(Path("tasks/wave1d"))

    assert [array.name for array in contract.output.arrays] == ["x", "state"]
    assert "energy_relative_drift" not in contract.numerics.tolerances
