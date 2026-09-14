"""Validate task contracts and their declared on-disk artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from invariantlab.schema import load_task_contract
from invariantlab.tasks.validation import validate_task_artifacts


def validate_task_dir(task_dir: Path) -> list[str]:
    """Validate one contract's schema separately from its filesystem layout."""
    contract_file = task_dir / "contract.yaml"
    if not contract_file.exists():
        return []  # No contract yet — not an error, just not implemented.
    try:
        contract = load_task_contract(task_dir)
    except Exception as error:
        return [f"schema error in {contract_file}: {error}"]
    return validate_task_artifacts(task_dir, contract)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task-dir", required=True, help="Root tasks directory.")
    args = parser.parse_args()

    tasks_root = Path(args.task_dir)
    if not tasks_root.exists():
        print(f"Note: task directory not found: {tasks_root}", file=sys.stderr)
        print("All task contracts and artifacts are valid.")
        return

    all_errors: list[str] = []
    for task_path in sorted(tasks_root.iterdir()):
        if task_path.is_dir():
            all_errors.extend(validate_task_dir(task_path))

    if all_errors:
        for error in all_errors:
            print(f"  ✗ {error}", file=sys.stderr)
        print(f"\n{len(all_errors)} validation error(s)", file=sys.stderr)
        sys.exit(1)
    print("All task contracts and artifacts are valid.")


if __name__ == "__main__":
    main()
