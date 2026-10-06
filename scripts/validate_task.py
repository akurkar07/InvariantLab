"""Validate fixed V1 task contracts and their declared on-disk artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from invariantlab.schema import load_task_contract
from invariantlab.tasks.validation import validate_task_artifacts

REQUIRED_V1_TASKS = ("oscillator", "kepler", "heat1d", "wave1d")


def validate_task_dir(task_dir: Path) -> list[str]:
    """Validate one contract's schema separately from its filesystem layout."""
    contract_file = task_dir / "contract.yaml"
    if not contract_file.exists():
        return []
    try:
        contract = load_task_contract(task_dir)
    except Exception as error:
        return [f"schema error in {contract_file}: {error}"]
    return validate_task_artifacts(task_dir, contract)


def validate_tasks_root(tasks_root: Path) -> list[str]:
    """Validate the complete, fixed V1 task package set."""
    errors: list[str] = []
    for task_name in REQUIRED_V1_TASKS:
        task_dir = tasks_root / task_name
        if not task_dir.is_dir():
            errors.append(f"required V1 task package {task_name!r} does not exist")
            continue

        contract_file = task_dir / "contract.yaml"
        if not contract_file.is_file():
            detail = "does not exist" if not contract_file.exists() else "must be a regular file"
            errors.append(f"task {task_name!r}: field 'contract.yaml' {detail}")
        else:
            errors.extend(validate_task_dir(task_dir))

        specification = task_dir / "specification.md"
        if not specification.is_file():
            detail = "does not exist" if not specification.exists() else "must be a regular file"
            errors.append(f"task {task_name!r}: field 'specification.md' {detail}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task-dir", required=True, help="Root tasks directory.")
    args = parser.parse_args()

    all_errors = validate_tasks_root(Path(args.task_dir))
    if all_errors:
        for error in all_errors:
            print(f"  ✗ {error}", file=sys.stderr)
        print(f"\n{len(all_errors)} validation error(s)", file=sys.stderr)
        sys.exit(1)
    print("All task contracts and artifacts are valid.")


if __name__ == "__main__":
    main()
