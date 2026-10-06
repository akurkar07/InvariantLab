"""Validate every registered task mutant against its reference and declared behavior."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from invariantlab.mutations import (
    MutationRegistryError,
    RegisteredMutant,
    discover_mutants,
    validate_mutant,
    validate_reference,
)


def _format_reasons(reasons: list[str]) -> str:
    return "; ".join(" ".join(reason.split()) for reason in reasons)


def validate_mutants(tasks_root: Path, task: str | None = None) -> int:
    """Validate all discovered mutants, returning a process exit code."""
    if not tasks_root.is_dir():
        print(f"FAIL task dir {tasks_root} does not exist", file=sys.stderr)
        return 1
    if task is not None and not (tasks_root / task).is_dir():
        print(f"FAIL task {task} does not exist under {tasks_root}", file=sys.stderr)
        return 1

    try:
        mutants = discover_mutants(tasks_root)
    except MutationRegistryError as error:
        print(f"FAIL mutant registry: {error}", file=sys.stderr)
        return 1

    if task is not None:
        mutants = [mutant for mutant in mutants if mutant.task_dir.name == task]
    if not mutants:
        detail = f" for task {task}" if task is not None else ""
        print(f"FAIL no mutants discovered under {tasks_root}{detail}", file=sys.stderr)
        return 1

    mutants_by_task: dict[str, list[RegisteredMutant]] = defaultdict(list)
    for mutant in mutants:
        mutants_by_task[mutant.task_dir.name].append(mutant)

    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    reference_passed = 0
    reference_total = 0
    mutant_passed = 0
    mutant_total = 0

    for task_name in sorted(mutants_by_task):
        task_mutants = mutants_by_task[task_name]
        reference = validate_reference(task_mutants[0].task_dir)
        reference_total += 1
        if reference.passed:
            reference_passed += 1
            print(f"PASS reference {task_name}")
        else:
            print(f"FAIL reference {task_name}: {_format_reasons(reference.reasons)}")

        for mutant in task_mutants:
            family = mutant.definition.family.value
            result = validate_mutant(mutant)
            mutant_total += 1
            family_counts[family]["total"] += 1
            if result.passed:
                mutant_passed += 1
                family_counts[family]["passed"] += 1
                print(f"PASS {task_name}/{mutant.definition.id} {family}")
            else:
                reasons = _format_reasons(result.reasons)
                print(f"FAIL {task_name}/{mutant.definition.id} {family}: {reasons}")

    for family in sorted(family_counts):
        counts = family_counts[family]
        print(f"{family}: {counts['passed']}/{counts['total']} passed")
    print(
        f"{mutant_passed}/{mutant_total} mutants passed, "
        f"{reference_passed}/{reference_total} references passed"
    )

    return int(mutant_passed != mutant_total or reference_passed != reference_total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task-dir", required=True, help="Root tasks directory.")
    parser.add_argument("--task", help="Validate mutants for one task directory name.")
    args = parser.parse_args()
    sys.exit(validate_mutants(Path(args.task_dir), args.task))


if __name__ == "__main__":
    main()
