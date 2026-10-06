"""Discovery and validation of evaluator-controlled task mutants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from invariantlab.schema import (
    MutationDefinition,
    TaskContract,
    load_mutation_definition,
    load_task_contract,
)
from invariantlab.tasks.validation import _resolve_under_task_root

if TYPE_CHECKING:
    from pathlib import Path


class MutationRegistryError(ValueError):
    """All manifest problems found while discovering task mutants."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__(str(self))

    def __str__(self) -> str:
        count = len(self.problems)
        noun = "problem" if count == 1 else "problems"
        return f"{count} mutant registry {noun}:\n" + "\n".join(
            f"- {problem}" for problem in self.problems
        )


@dataclass(frozen=True)
class RegisteredMutant:
    """A validated mutant manifest and its task-local paths."""

    task_dir: Path
    contract: TaskContract
    definition: MutationDefinition
    mutation_dir: Path
    source_path: Path


def discover_mutants(tasks_root: Path, *, include_legacy: bool = False) -> list[RegisteredMutant]:
    """Discover and validate every mutant manifest beneath a task collection."""
    problems: list[str] = []
    registered: list[RegisteredMutant] = []

    task_dirs = sorted(
        (
            path
            for path in tasks_root.iterdir()
            if path.is_dir() and (path / "contract.yaml").is_file()
        ),
        key=lambda path: path.name,
    )
    for task_dir in task_dirs:
        try:
            contract = load_task_contract(task_dir)
        except (OSError, ValueError, yaml.YAMLError) as error:
            problems.append(
                f"{(task_dir / 'contract.yaml').as_posix()}: unable to load task contract: {error}"
            )
            continue

        mutations_dir = task_dir / "mutations"
        if not mutations_dir.is_dir():
            continue

        seen_ids: dict[str, Path] = {}
        for mutation_dir in sorted(
            (path for path in mutations_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            manifest_path = mutation_dir / "mutation.yaml"
            if not manifest_path.is_file():
                problems.append(f"{manifest_path.as_posix()}: missing mutation.yaml")
                continue

            try:
                definition = load_mutation_definition(mutation_dir)
            except (OSError, ValueError, yaml.YAMLError) as error:
                problems.append(f"{manifest_path.as_posix()}: {error}")
                continue

            if definition.id != mutation_dir.name:
                problems.append(
                    f"{manifest_path.as_posix()}: id {definition.id!r} does not match "
                    f"directory name {mutation_dir.name!r}"
                )
            if definition.task_id != contract.id:
                problems.append(
                    f"{manifest_path.as_posix()}: task_id {definition.task_id!r} does not "
                    f"match contract id {contract.id!r}"
                )
            if definition.id in seen_ids:
                problems.append(
                    f"{manifest_path.as_posix()}: duplicate id {definition.id!r} "
                    f"(also declared by {(seen_ids[definition.id] / 'mutation.yaml').as_posix()})"
                )
            else:
                seen_ids[definition.id] = mutation_dir

            source_path = _resolve_under_task_root(mutation_dir.resolve(), definition.source)
            if source_path is None:
                problems.append(
                    f"{manifest_path.as_posix()}: source must be a safe relative path "
                    "inside the mutant directory"
                )
            elif not source_path.is_file():
                problems.append(
                    f"{manifest_path.as_posix()}: source {definition.source!r} does not exist"
                )

            for expected_failure in definition.expected_failures:
                test_file = expected_failure.test.split("::", 1)[0]
                test_path = _resolve_under_task_root(task_dir.resolve(), test_file)
                if test_path is None or not test_path.is_file():
                    problems.append(
                        f"{manifest_path.as_posix()}: expected failure test "
                        f"{expected_failure.test!r} does not exist inside the task"
                    )

            if source_path is not None and (definition.interface == "package" or include_legacy):
                registered.append(
                    RegisteredMutant(
                        task_dir=task_dir,
                        contract=contract,
                        definition=definition,
                        mutation_dir=mutation_dir,
                        source_path=mutation_dir / definition.source,
                    )
                )

    if problems:
        raise MutationRegistryError(problems)

    return sorted(registered, key=lambda mutant: (mutant.task_dir.name, mutant.definition.id))
