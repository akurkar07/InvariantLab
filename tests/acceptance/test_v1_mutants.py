"""V1-AC2: every registered mutant passes its weak profile and fails its expected gate."""

from __future__ import annotations

import re
import tempfile
from collections import Counter
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from invariantlab.mutations import (
    RegisteredMutant,
    discover_mutants,
    validate_mutant,
    validate_reference,
)
from invariantlab.tasks.workspace import build_agent_workspace
from invariantlab.verification.verify import SCIENTIFIC_LAYERS, verify_candidate

if TYPE_CHECKING:
    from invariantlab.schema import GateResult, VerificationResult

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = REPO_ROOT / "tasks"
REGISTRY = discover_mutants(TASKS_ROOT, include_legacy=True)
PACKAGE_MUTANTS = [mutant for mutant in REGISTRY if mutant.definition.interface == "package"]
LEGACY_MUTANTS = [mutant for mutant in REGISTRY if mutant.definition.interface == "legacy_study"]
# Execution and public-test failures are never acceptable collateral for a controlled defect.
NEVER_COLLATERAL_LAYERS = ("L0", "L1")


def _mutant_id(mutant: RegisteredMutant) -> str:
    return f"{mutant.task_dir.name}/{mutant.definition.id}"


def _verify(task_dir: Path, mutant: RegisteredMutant | None) -> VerificationResult:
    with tempfile.TemporaryDirectory(
        prefix="invariantlab-v1-ac2-", ignore_cleanup_errors=True
    ) as temporary:
        work = Path(temporary)
        workspace = build_agent_workspace(task_dir, work / "agent")
        attempt_id = "reference"
        if mutant is not None:
            (workspace / mutant.contract.entrypoint).write_bytes(mutant.source_path.read_bytes())
            attempt_id = mutant.definition.id
        return verify_candidate(task_dir, workspace, attempt_id, work / "verify")


@cache
def _reference_verification(task_dir: Path) -> VerificationResult:
    return _verify(task_dir, None)


@cache
def _reference_validation_reasons(task_dir: Path) -> tuple[bool, tuple[str, ...]]:
    result = validate_reference(task_dir)
    return result.passed, tuple(result.reasons)


def _failed_gates(
    result: VerificationResult, layers: tuple[str, ...]
) -> list[tuple[str, GateResult]]:
    return [
        (layer, gate)
        for layer in layers
        for gate in result.layers.get(layer, [])
        if not gate.passed
    ]


def _describe(gates: list[tuple[str, GateResult]]) -> str:
    return "; ".join(f"{layer}/{gate.name}: {gate.detail}" for layer, gate in gates) or "none"


@pytest.mark.v1_acceptance("V1-AC2")
def test_mutation_registry_is_non_empty() -> None:
    assert PACKAGE_MUTANTS, f"no package mutants registered under {TASKS_ROOT}"
    counts = Counter(mutant.task_dir.name for mutant in PACKAGE_MUTANTS)
    print(
        "V1-AC2 mutants per task: "
        + ", ".join(f"{task}={count}" for task, count in sorted(counts.items()))
        + f" (total {len(PACKAGE_MUTANTS)})"
    )
    skipped = ", ".join(_mutant_id(mutant) for mutant in LEGACY_MUTANTS) or "none"
    print(
        f"V1-AC2 skipped legacy_study mutants (graded by the legacy verifier until #120): {skipped}"
    )


@pytest.mark.v1_acceptance("V1-AC2")
@pytest.mark.parametrize("mutant", PACKAGE_MUTANTS, ids=_mutant_id)
def test_mutant_passes_weak_profile_and_fails_expected_gate(mutant: RegisteredMutant) -> None:
    name = _mutant_id(mutant)
    declared = mutant.definition.expected_failures
    assert declared, f"{name} declares no expected_failures gate"

    validation = validate_mutant(mutant)
    reasons = "\n".join(validation.reasons)
    unexpected = [reason for reason in validation.reasons if "unexpected failure" in reason]
    assert not unexpected, f"{name} fails undeclared scientific gates:\n" + "\n".join(unexpected)
    assert validation.public_passed, f"{name} does not pass its weak public profile:\n{reasons}"
    assert validation.expected_failures_matched, (
        f"{name} does not fail its declared expected_failures:\n{reasons}"
    )
    assert validation.passed, f"{name} failed mutant validation:\n{reasons}"

    reference_passed, reference_reasons = _reference_validation_reasons(mutant.task_dir)
    assert reference_passed, (
        f"reference {mutant.task_dir.name} fails the gates {name} is expected to fail:\n"
        + "\n".join(reference_reasons)
    )

    result = _verify(mutant.task_dir, mutant)
    collateral = _failed_gates(result, NEVER_COLLATERAL_LAYERS)
    assert not collateral, f"{name} fails unexpected gates: {_describe(collateral)}"
    assert result.public_passed, f"{name} fails verify_candidate public tests"
    failed = _failed_gates(result, SCIENTIFIC_LAYERS)
    assert not result.scientific_passed and failed, f"{name} passes every verify_candidate gate"
    matched = [
        (layer, gate)
        for layer, gate in failed
        if any(re.search(expected.message, gate.detail) for expected in declared)
    ]
    assert matched, (
        f"no failed verify_candidate gate matches {name}'s declared expected_failures "
        f"{[expected.message for expected in declared]}; failed gates: {_describe(failed)}"
    )

    reference = _reference_verification(mutant.task_dir)
    reference_failed = {
        (layer, gate.name) for layer, gate in _failed_gates(reference, SCIENTIFIC_LAYERS)
    }
    regressed = [(layer, gate) for layer, gate in failed if (layer, gate.name) in reference_failed]
    assert not regressed, (
        f"reference {mutant.task_dir.name} also fails gates {name} fails: {_describe(regressed)}"
    )
    assert reference.passed_all, f"reference {mutant.task_dir.name} fails verify_candidate"


@pytest.mark.parametrize("mutant", LEGACY_MUTANTS, ids=_mutant_id)
def test_legacy_study_mutants_are_excluded(mutant: RegisteredMutant) -> None:
    pytest.skip(
        f"{_mutant_id(mutant)} uses interface legacy_study; graded by the legacy verifier until #120"
    )
