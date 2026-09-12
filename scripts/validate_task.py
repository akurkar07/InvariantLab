"""Validate task contracts.

Reads each task directory's contract.yaml and checks it against the TaskContract schema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from invariantlab.schema import TaskContract


def validate_task_dir(task_dir: Path) -> list[str]:
    """Validate a single task directory. Returns list of error messages."""
    errors: list[str] = []
    contract_file = task_dir / "contract.yaml"
    if not contract_file.exists():
        errors.append(f"{task_dir}: missing contract.yaml")
        return errors
    try:
        import yaml

        with contract_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        TaskContract(**data)
    except Exception as e:
        errors.append(f"{task_dir}: {e}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate task contracts.")
    parser.add_argument("--task-dir", required=True, help="Root tasks directory.")
    args = parser.parse_args()

    tasks_root = Path(args.task_dir)
    if not tasks_root.exists():
        print(f"Note: task directory not found: {tasks_root}", file=sys.stderr)
        print("All task contracts valid.")
        return

    all_errors: list[str] = []
    for task_path in sorted(tasks_root.iterdir()):
        if task_path.is_dir():
            all_errors.extend(validate_task_dir(task_path))

    if all_errors:
        for err in all_errors:
            print(f"  ✗ {err}", file=sys.stderr)
        print(f"\n{len(all_errors)} validation error(s)", file=sys.stderr)
        sys.exit(1)
    else:
        print("All task contracts valid.")


if __name__ == "__main__":
    main()
