"""Validate controlled defect mutations.

Checks that each mutant:
  1. passes its designated weak public test profile,
  2. fails at least one predeclared scientific property for the expected reason.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate mutants.")
    parser.add_argument("--task-dir", required=True, help="Root tasks directory.")
    args = parser.parse_args()

    tasks_root = Path(args.task_dir)
    if not tasks_root.exists():
        print(f"ERROR: task directory not found: {tasks_root}", file=sys.stderr)
        sys.exit(1)

    # TODO: implement full mutant validation in M4
    print("Mutant validation: no tasks found (expected pre-M2).")


if __name__ == "__main__":
    main()
