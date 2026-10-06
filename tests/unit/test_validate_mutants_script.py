from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "validate_mutants.py"
MUTANT_SOURCE = REPO_ROOT / "tasks" / "wave1d" / "mutations" / "sign-error-startup"


def make_tasks_root(tmp_path: Path, *, include_mutant: bool = False) -> Path:
    tasks_root = tmp_path / "tasks"
    shutil.copytree(
        REPO_ROOT / "tasks",
        tasks_root,
        ignore=shutil.ignore_patterns("mutations", "__pycache__", ".pytest_cache", "result.npz"),
    )
    if include_mutant:
        mutant_dir = tasks_root / "wave1d" / "mutations" / "sign-error-startup"
        mutant_dir.parent.mkdir(parents=True)
        shutil.copytree(MUTANT_SOURCE, mutant_dir)
    return tasks_root


def run_validator(tasks_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task-dir",
            str(tasks_root),
            *arguments,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_valid_mutant_passes_for_filtered_task(tmp_path: Path) -> None:
    tasks_root = make_tasks_root(tmp_path, include_mutant=True)

    result = run_validator(tasks_root, "--task", "wave1d")

    assert result.returncode == 0
    assert "PASS reference wave1d" in result.stdout
    assert "PASS wave1d/sign-error-startup sign_error" in result.stdout
    assert "sign_error: 1/1 passed" in result.stdout


def test_invalid_expected_failure_message_fails_mutant(tmp_path: Path) -> None:
    tasks_root = make_tasks_root(tmp_path, include_mutant=True)
    manifest = tasks_root / "wave1d" / "mutations" / "sign-error-startup" / "mutation.yaml"
    contents = manifest.read_text(encoding="utf-8")
    target = 'message: "state_relative_l2="'
    assert contents.count(target) == 1
    manifest.write_text(contents.replace(target, 'message: "no-such-message"'), encoding="utf-8")

    result = run_validator(tasks_root, "--task", "wave1d")

    assert result.returncode == 1
    assert "FAIL wave1d/sign-error-startup sign_error:" in result.stdout


def test_no_mutants_is_a_failure(tmp_path: Path) -> None:
    tasks_root = make_tasks_root(tmp_path)

    result = run_validator(tasks_root)

    assert result.returncode == 1
    assert "no mutants discovered" in result.stderr


def test_missing_tasks_directory_is_a_failure(tmp_path: Path) -> None:
    result = run_validator(tmp_path / "nope")

    assert result.returncode == 1


def test_task_filter_with_no_mutants_is_a_failure(tmp_path: Path) -> None:
    tasks_root = make_tasks_root(tmp_path, include_mutant=True)

    result = run_validator(tasks_root, "--task", "oscillator")

    assert result.returncode == 1
    assert "no mutants discovered" in result.stderr


def test_registry_error_is_a_failure(tmp_path: Path) -> None:
    tasks_root = make_tasks_root(tmp_path, include_mutant=True)
    manifest = tasks_root / "wave1d" / "mutations" / "sign-error-startup" / "mutation.yaml"
    contents = manifest.read_text(encoding="utf-8")
    target = "id: sign-error-startup"
    assert contents.count(target) == 1
    manifest.write_text(contents.replace(target, "id: wrong-id"), encoding="utf-8")

    result = run_validator(tasks_root)

    assert result.returncode == 1
    assert "mutant registry" in result.stderr
