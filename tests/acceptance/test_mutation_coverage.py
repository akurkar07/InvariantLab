"""Every MutationFamily is realised by at least one registered package mutant."""

from __future__ import annotations

from pathlib import Path

import pytest

from invariantlab.mutations import discover_mutants
from invariantlab.schema import MutationFamily

REPO_ROOT = Path(__file__).resolve().parents[2]
# Families with no feasible package mutant must cite the issue where infeasibility was found.
INFEASIBLE_FAMILIES: dict[MutationFamily, str] = {}


@pytest.fixture
def covered_families(monkeypatch: pytest.MonkeyPatch) -> set[MutationFamily]:
    monkeypatch.chdir(REPO_ROOT)
    return {mutant.definition.family for mutant in discover_mutants(Path("tasks"))}


@pytest.mark.parametrize("family", list(MutationFamily), ids=lambda family: family.value)
def test_every_mutation_family_has_a_mutant(
    family: MutationFamily, covered_families: set[MutationFamily]
) -> None:
    assert family in covered_families or family in INFEASIBLE_FAMILIES, (
        f"{family.value!r} has no package mutant; add one or cite its infeasibility issue "
        "in INFEASIBLE_FAMILIES."
    )


def test_infeasible_families_have_no_mutants(
    covered_families: set[MutationFamily],
) -> None:
    overlap = covered_families.intersection(INFEASIBLE_FAMILIES)
    assert not overlap, (
        f"infeasible families already have mutants: {sorted(f.value for f in overlap)}"
    )


def test_catalogue_lists_every_package_mutant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    catalogue = (REPO_ROOT / "docs" / "experiment-authoring.md").read_text(encoding="utf-8")
    for mutant in discover_mutants(Path("tasks")):
        row = f"| {mutant.task_dir.name} | `{mutant.definition.id}` |"
        assert row in catalogue, (
            f"catalogue is missing {mutant.task_dir.name}/{mutant.definition.id}"
        )
