"""Filesystem and path validation for task contracts."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from invariantlab.schema import TaskContract

_ARTIFACTS = (
    ("entrypoint", "file"),
    ("public_tests", "directory"),
    ("scientific_tests", "directory"),
)


def _has_safe_relative_syntax(declared_path: str) -> bool:
    """Return whether a contract path is relative in both POSIX and Windows syntax."""
    posix_path = PurePosixPath(declared_path)
    windows_path = PureWindowsPath(declared_path)
    return not (
        not declared_path
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    )


def _resolve_under_task_root(task_root: Path, declared_path: str) -> Path | None:
    """Resolve a declared relative path and return it only if it stays in ``task_root``."""
    if not _has_safe_relative_syntax(declared_path):
        return None
    try:
        candidate = (task_root / declared_path).resolve(strict=False)
        candidate.relative_to(task_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate


def _test_directory_status(task_root: Path, test_directory: Path) -> tuple[bool, str | None]:
    """Return discoverability and any unsafe descendant-layout error."""
    pending = [test_directory]
    visited_directories: set[Path] = set()
    has_tests = False
    while pending:
        current = pending.pop()
        try:
            resolved_current = current.resolve(strict=False)
            resolved_current.relative_to(task_root)
        except (OSError, RuntimeError, ValueError):
            return False, "contains a path that escapes the task root"
        if resolved_current in visited_directories:
            return False, "contains a directory cycle"
        visited_directories.add(resolved_current)
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    descendant = Path(entry.path)
                    try:
                        descendant.resolve(strict=False).relative_to(task_root)
                    except (OSError, RuntimeError, ValueError):
                        return False, "contains a path that escapes the task root"
                    if entry.is_dir(follow_symlinks=True):
                        pending.append(descendant)
                    elif (
                        entry.is_file(follow_symlinks=True)
                        and descendant.suffix == ".py"
                        and (
                            descendant.name.startswith("test_")
                            or descendant.name.endswith("_test.py")
                        )
                    ):
                        has_tests = True
        except OSError:
            return False, "cannot be scanned safely"
    return has_tests, None


def _error(task_id: str, field: str, detail: str) -> str:
    return f"task {task_id!r}: field {field!r} {detail}"


def validate_task_artifacts(task_dir: str | Path, contract: TaskContract) -> list[str]:
    """Return layout errors for paths declared by ``contract``.

    This validates only the task package layout. It never imports or executes a
    declared entrypoint or test module.
    """
    task_root = Path(task_dir).resolve()
    errors: list[str] = []
    for field, expected_kind in _ARTIFACTS:
        declared_path = getattr(contract, field)
        candidate = _resolve_under_task_root(task_root, declared_path)
        if candidate is None:
            syntax_is_safe = _has_safe_relative_syntax(declared_path)
            detail = "escapes the task root" if syntax_is_safe else "must be a safe relative path"
            errors.append(_error(contract.id, field, detail))
            continue
        if not candidate.exists():
            errors.append(_error(contract.id, field, "does not exist"))
            continue
        if expected_kind == "file":
            if not candidate.is_file():
                errors.append(_error(contract.id, field, "must be a regular file"))
            continue
        if not candidate.is_dir():
            errors.append(_error(contract.id, field, "must be a directory"))
            continue
        has_tests, layout_error = _test_directory_status(task_root, candidate)
        if layout_error is not None:
            errors.append(_error(contract.id, field, layout_error))
        elif not has_tests:
            errors.append(_error(contract.id, field, "does not contain discoverable tests"))

    output_path = contract.output.path
    output_candidate = _resolve_under_task_root(task_root, output_path)
    if output_candidate is None:
        errors.append(_error(contract.id, "output.path", "must be a safe relative path"))
    elif output_candidate.exists() and output_candidate.is_dir():
        errors.append(_error(contract.id, "output.path", "must not be a directory"))
    return errors
