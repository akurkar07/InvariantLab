"""Acceptance tests for InvariantLab."""

import subprocess
import sys


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


def test_validate_task_no_contracts():
    """validate_task prints valid when no contracts exist yet."""
    result = subprocess.run(
        [sys.executable, "scripts/validate_task.py", "--task-dir", "tasks/"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "valid" in result.stdout.lower()
