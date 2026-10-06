"""Copy-based workspaces keep hidden evaluation tests outside the agent boundary."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from invariantlab.schema import load_task_contract
from invariantlab.tasks.validation import _resolve_under_task_root

if TYPE_CHECKING:
    from invariantlab.schema import TaskContract


def _reject_symlinks(source: Path) -> None:
    if source.is_symlink():
        raise ValueError(f"Symlinks are not allowed in copied source: {source}")
    if source.is_dir():

        def raise_walk_error(error: OSError) -> None:
            raise ValueError(f"Could not inspect copied source {source}: {error}") from error

        for current, directories, files in os.walk(
            source, followlinks=False, onerror=raise_walk_error
        ):
            for name in (*directories, *files):
                path = Path(current) / name
                if path.is_symlink():
                    raise ValueError(f"Symlinks are not allowed in copied source: {path}")


def _require_file(source: Path) -> None:
    _reject_symlinks(source)
    if not source.is_file():
        raise ValueError(f"Required source file does not exist: {source}")


def _require_directory(source: Path) -> None:
    _reject_symlinks(source)
    if not source.is_dir():
        raise ValueError(f"Required source directory does not exist: {source}")


def _contract_path(task_root: Path, contract: TaskContract, field: str) -> Path:
    declared_path = getattr(contract, field)
    resolved_path = _resolve_under_task_root(task_root, declared_path)
    if resolved_path is None:
        raise ValueError(f"Contract field {field!r} must resolve within task root")
    _reject_symlinks(task_root / declared_path)
    return resolved_path


def _prepare_destination(dest: Path, forbidden_roots: tuple[Path, ...]) -> Path:
    try:
        resolved_dest = dest.resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Could not resolve workspace destination {dest}: {error}") from error
    if any(resolved_dest.is_relative_to(root) for root in forbidden_roots):
        raise ValueError(f"Workspace destination must be outside source roots: {resolved_dest}")
    if resolved_dest.exists() and (not resolved_dest.is_dir() or any(resolved_dest.iterdir())):
        raise ValueError(f"Workspace destination must be empty: {resolved_dest}")
    try:
        resolved_dest.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        message = f"Could not create workspace destination {resolved_dest}: {error}"
        raise ValueError(message) from error
    return resolved_dest


def _copy_file(source: Path, dest: Path) -> None:
    _require_file(source)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    except OSError as error:
        raise ValueError(f"Could not copy {source} to {dest}: {error}") from error


def _copy_tree(source: Path, dest: Path, *, dirs_exist_ok: bool = False) -> None:
    _require_directory(source)
    try:
        shutil.copytree(
            source,
            dest,
            dirs_exist_ok=dirs_exist_ok,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
        )
    except OSError as error:
        raise ValueError(f"Could not copy {source} to {dest}: {error}") from error


def _load_contract(task_dir: Path, task_root: Path) -> TaskContract:
    try:
        return load_task_contract(task_dir)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not load task contract from {task_root}: {error}") from error


def _task_paths(task_root: Path, contract: TaskContract) -> tuple[Path, Path, Path, Path]:
    entrypoint = _contract_path(task_root, contract, "entrypoint")
    public_tests = _contract_path(task_root, contract, "public_tests")
    scientific_tests = _contract_path(task_root, contract, "scientific_tests")
    _require_file(entrypoint)
    _require_directory(public_tests)
    _require_directory(scientific_tests)
    source_tree = entrypoint.parent
    if source_tree == task_root:
        raise ValueError("Contract entrypoint cannot be at the task root")
    return entrypoint, source_tree, public_tests, scientific_tests


def build_agent_workspace(task_dir: Path, dest: Path) -> Path:
    task_dir_path = Path(task_dir)
    if task_dir_path.is_symlink():
        raise ValueError(f"Task root cannot be a symlink: {task_dir_path}")
    task_root = task_dir_path.resolve()
    contract = _load_contract(task_dir_path, task_root)
    _, source_tree, public_tests, scientific_tests = _task_paths(task_root, contract)
    if scientific_tests.is_relative_to(source_tree) or scientific_tests.is_relative_to(
        public_tests
    ):
        raise ValueError("Scientific tests cannot be inside the agent source or public-test tree")

    _require_file(task_root / "contract.yaml")
    _require_file(task_root / "specification.md")
    _require_file(task_root / "tests" / "conftest.py")
    source_root = _prepare_destination(Path(dest), (task_root,))

    _copy_file(task_root / "contract.yaml", source_root / "contract.yaml")
    _copy_file(task_root / "specification.md", source_root / "specification.md")
    _copy_tree(source_tree, source_root / source_tree.relative_to(task_root))
    _copy_file(task_root / "tests" / "conftest.py", source_root / "tests" / "conftest.py")
    _copy_tree(public_tests, source_root / public_tests.relative_to(task_root))
    return source_root


def build_evaluation_workspace(task_dir: Path, candidate_workspace: Path, dest: Path) -> Path:
    task_dir_path = Path(task_dir)
    if task_dir_path.is_symlink():
        raise ValueError(f"Task root cannot be a symlink: {task_dir_path}")
    task_root = task_dir_path.resolve()
    contract = _load_contract(task_dir_path, task_root)
    _, source_tree, public_tests, scientific_tests = _task_paths(task_root, contract)
    candidate_root = Path(candidate_workspace)
    _reject_symlinks(candidate_root)
    candidate_root = candidate_root.resolve()
    if not candidate_root.is_dir():
        raise ValueError(f"Candidate workspace must be a directory: {candidate_root}")
    candidate_source = candidate_root / source_tree.relative_to(task_root)
    _require_directory(candidate_source)

    tests_tree = task_root / "tests"
    _require_file(task_root / "contract.yaml")
    _require_file(task_root / "specification.md")
    _require_directory(tests_tree)
    source_root = _prepare_destination(Path(dest), (task_root, candidate_root))

    _copy_tree(candidate_source, source_root / candidate_source.relative_to(candidate_root))
    _copy_file(task_root / "contract.yaml", source_root / "contract.yaml")
    _copy_file(task_root / "specification.md", source_root / "specification.md")
    _copy_tree(tests_tree, source_root / "tests")
    for test_tree in (public_tests, scientific_tests):
        if not test_tree.is_relative_to(tests_tree):
            _copy_tree(
                test_tree,
                source_root / test_tree.relative_to(task_root),
                dirs_exist_ok=True,
            )
    return source_root
