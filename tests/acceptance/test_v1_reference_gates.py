"""V1-AC1: every trusted reference implementation passes every scientific gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from invariantlab.schema import TaskContract, VerificationResult, load_task_contract
from invariantlab.tasks.validation import REQUIRED_V1_TASKS
from invariantlab.tasks.workspace import build_agent_workspace
from invariantlab.verification import robustness
from invariantlab.verification.execution import GATE_NAMES
from invariantlab.verification.verify import (
    LAYER_NAMES,
    SCIENTIFIC_LAYERS,
    canonical_parameters,
    verify_candidate,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = REPOSITORY_ROOT / "tasks"
pytestmark = pytest.mark.v1_acceptance("V1-AC1")

VERLET_BLOCK = (
    "        v_half = v - 0.5 * dt * omega2 * x\n"
    "        x = x + dt * v_half\n"
    "        v = v_half - 0.5 * dt * omega2 * x\n"
)
# Update order swapped: the second velocity half-step reuses the stale acceleration.
STALE_ACCELERATION_BLOCK = (
    "        a = -omega2 * x\n"
    "        v_half = v + 0.5 * dt * a\n"
    "        x = x + dt * v_half\n"
    "        v = v_half + 0.5 * dt * a\n"
)


def _expected_gate_names(contract: TaskContract) -> dict[tuple[str, ...], set[str]]:
    """Gates the verifier must report for this contract, keyed by the layers that hold them.

    Declared tolerances are graded by the oracle (L2) or invariant (L3) gate of the same name.
    """
    tolerances = set(contract.numerics.tolerances)
    valid_cases, rejection_cases = robustness._cases(contract.id)
    layer6 = {f"robustness[{case_id}]/{name}" for case_id, _ in valid_cases for name in GATE_NAMES}
    layer6 |= {f"robustness[{case_id}]/{name}" for case_id, _ in valid_cases for name in tolerances}
    layer6 |= {f"robustness[{case_id}]/rejected" for case_id, _ in rejection_cases}
    return {
        ("L0",): set(GATE_NAMES),
        ("L1",): {"public_tests"},
        ("L2", "L3"): tolerances,
        ("L6",): layer6,
    }


def _assert_every_gate_passed(result: VerificationResult, contract: TaskContract) -> None:
    assert tuple(result.layers) == LAYER_NAMES
    assert sum(len(gates) for gates in result.layers.values()) > 0, "verifier reported no gates"
    for layer in LAYER_NAMES:
        assert result.layers[layer], f"{contract.id}: layer {layer} reported no gates"
    for layers, expected in _expected_gate_names(contract).items():
        reported = {gate.name for layer in layers for gate in result.layers[layer]}
        missing = expected - reported
        assert not missing, f"{contract.id}: {'/'.join(layers)} is missing gates {sorted(missing)}"
    failed = [
        f"{layer}:{gate.name} ({gate.detail})"
        for layer, gates in result.layers.items()
        for gate in gates
        if not gate.passed
    ]
    assert failed == [], f"{contract.id}: failed gates {failed}"
    assert result.public_passed and result.scientific_passed and result.passed_all


@pytest.fixture(scope="module")
def reference_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, VerificationResult]:
    results: dict[str, VerificationResult] = {}
    for task_name in REQUIRED_V1_TASKS:
        task_dir = TASKS_ROOT / task_name
        work_dir = tmp_path_factory.mktemp(f"v1-ac1-{task_name}")
        results[task_name] = verify_candidate(task_dir, task_dir, "reference", work_dir)
    return results


def test_every_v1_task_is_covered(reference_results: dict[str, VerificationResult]) -> None:
    on_disk = {path.parent.name for path in TASKS_ROOT.glob("*/contract.yaml")}
    assert set(REQUIRED_V1_TASKS) == on_disk
    assert set(reference_results) == set(REQUIRED_V1_TASKS)


@pytest.mark.parametrize("task_name", REQUIRED_V1_TASKS)
def test_reference_passes_every_scientific_gate(
    task_name: str, reference_results: dict[str, VerificationResult]
) -> None:
    contract = load_task_contract(TASKS_ROOT / task_name)
    result = reference_results[task_name]

    assert result.task_id == contract.id
    _assert_every_gate_passed(result, contract)


@pytest.mark.parametrize("task_name", ["heat1d", "wave1d"])
def test_gated_cases_include_domain_length_other_than_one(task_name: str) -> None:
    contract = load_task_contract(TASKS_ROOT / task_name)
    valid_cases, _ = robustness._cases(contract.id)

    assert canonical_parameters(contract.id)["length"] != 1.0
    assert any(parameters["length"] != 1.0 for _, parameters in valid_cases)


def test_missing_gate_fails_the_check(reference_results: dict[str, VerificationResult]) -> None:
    contract = load_task_contract(TASKS_ROOT / "oscillator")
    result = reference_results["oscillator"]
    layers = {layer: list(gates) for layer, gates in result.layers.items()}
    layers["L6"] = layers["L6"][1:]

    with pytest.raises(AssertionError, match="missing gates"):
        _assert_every_gate_passed(result.model_copy(update={"layers": layers}), contract)
    with pytest.raises(AssertionError, match="no gates"):
        _assert_every_gate_passed(
            result.model_copy(update={"layers": {layer: [] for layer in LAYER_NAMES}}), contract
        )


def test_broken_reference_fails_a_scientific_gate(tmp_path: Path) -> None:
    task_dir = TASKS_ROOT / "oscillator"
    workspace = build_agent_workspace(task_dir, tmp_path / "agent")
    solver = workspace / "src" / "solver.py"
    source = solver.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert source.count(VERLET_BLOCK) == 1
    solver.write_text(source.replace(VERLET_BLOCK, STALE_ACCELERATION_BLOCK), encoding="utf-8")

    result = verify_candidate(task_dir, workspace, "broken", tmp_path / "work")

    assert result.scientific_passed is False
    assert any(not gate.passed for layer in SCIENTIFIC_LAYERS for gate in result.layers[layer])
    with pytest.raises(AssertionError):
        _assert_every_gate_passed(result, load_task_contract(task_dir))
