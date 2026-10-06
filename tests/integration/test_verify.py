"""Integration tests for the layered `verify_candidate` entrypoint and `invariantlab verify`."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from invariantlab.cli import app
from invariantlab.schema import VerificationResult
from invariantlab.tasks.workspace import build_agent_workspace
from invariantlab.verification.verify import LAYER_NAMES, verify_candidate

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = REPO_ROOT / "tasks"
TRUSTED_TASKS = ("oscillator", "kepler", "heat1d", "wave1d")

# Update-order defect (stale acceleration in the second velocity half-step) in the
# CLI/NPZ path only. `solve()` stays correct, so the public small-example test passes.
STALE_ACCELERATION_SOLVER = """

def _solve_cli(x0: float, v0: float, omega: float, dt: float, n_steps: int) -> np.ndarray:
    state = np.empty((n_steps + 1, 2), dtype=np.float64)
    state[0] = (x0, v0)
    omega2 = omega * omega
    x, v = float(x0), float(v0)
    for i in range(n_steps):
        a = -omega2 * x
        v_half = v + 0.5 * dt * a
        x = x + dt * v_half
        v = v_half + 0.5 * dt * a
        state[i + 1] = (x, v)
    return state
"""


def _agent_workspace(task_name: str, dest: Path) -> Path:
    return build_agent_workspace(TASKS_ROOT / task_name, dest)


def _update_order_candidate(dest: Path) -> Path:
    workspace = _agent_workspace("oscillator", dest)
    solver = workspace / "src" / "solver.py"
    source = solver.read_text(encoding="utf-8").replace("\r\n", "\n")
    call = "    state = solve(x0, v0, omega, dt, n_steps)\n"
    anchor = "\n\ndef run(input_path: Path, output_path: Path) -> None:\n"
    assert source.count(call) == 1
    assert source.count(anchor) == 1
    source = source.replace(call, call.replace("solve(", "_solve_cli("))
    source = source.replace(anchor, STALE_ACCELERATION_SOLVER + anchor)
    solver.write_text(source, encoding="utf-8")
    return workspace


@pytest.mark.parametrize("task_name", TRUSTED_TASKS)
def test_trusted_agent_workspace_passes_all_layers(tmp_path: Path, task_name: str) -> None:
    workspace = _agent_workspace(task_name, tmp_path / "agent")

    result = verify_candidate(TASKS_ROOT / task_name, workspace, "trusted", tmp_path / "work")

    failed = [
        (layer, gate) for layer, gates in result.layers.items() for gate in gates if not gate.passed
    ]
    assert failed == []
    assert tuple(result.layers) == LAYER_NAMES
    assert all(result.layers[layer] for layer in LAYER_NAMES)
    assert result.public_passed and result.scientific_passed and result.passed_all
    assert result.attempt_id == "trusted"


def test_update_order_defect_passes_public_but_fails_scientific(tmp_path: Path) -> None:
    workspace = _update_order_candidate(tmp_path / "agent")

    result = verify_candidate(TASKS_ROOT / "oscillator", workspace, "defect", tmp_path / "work")

    assert result.public_passed is True
    assert result.scientific_passed is False
    assert result.passed_all is False
    assert all(gate.passed for gate in result.layers["L0"])
    assert not all(gate.passed for gate in result.layers["L2"])
    assert not all(gate.passed for gate in result.layers["L3"])


def test_layer0_failure_reports_skipped_scientific_layers(tmp_path: Path) -> None:
    workspace = _agent_workspace("oscillator", tmp_path / "agent")
    solver = workspace / "src" / "solver.py"
    solver.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")

    result = verify_candidate(TASKS_ROOT / "oscillator", workspace, "crash", tmp_path / "work")

    assert tuple(result.layers) == LAYER_NAMES
    assert result.layers["L0"][0].name == "execution"
    assert not result.layers["L0"][0].passed
    for layer in ("L2", "L3", "L4", "L5", "L6"):
        assert [gate.passed for gate in result.layers[layer]] == [False]
        assert result.layers[layer][0].detail.startswith("skipped")
    assert result.public_passed is False
    assert result.scientific_passed is False


def test_verification_is_deterministic(tmp_path: Path) -> None:
    workspace = _update_order_candidate(tmp_path / "agent")
    task_dir = TASKS_ROOT / "oscillator"

    first = verify_candidate(task_dir, workspace, "a", tmp_path / "first")
    second = verify_candidate(task_dir, workspace, "a", tmp_path / "second")

    def scientific(result: VerificationResult) -> dict[str, object]:
        dump = result.model_dump()
        dump["layers"].pop("L1")  # pytest summary includes wall-clock duration
        return dump

    assert scientific(first) == scientific(second)
    assert first.layers["L1"][0].passed == second.layers["L1"][0].passed


def test_cli_writes_round_trippable_result(tmp_path: Path) -> None:
    output = tmp_path / "out" / "result.json"
    result = CliRunner().invoke(
        app,
        [
            "verify",
            "--task",
            str(TASKS_ROOT / "oscillator"),
            "--candidate",
            str(TASKS_ROOT / "oscillator"),
            "--output",
            str(output),
            "--attempt-id",
            "cli-1",
        ],
    )

    assert result.exit_code == 0, result.output
    parsed = VerificationResult.model_validate_json(output.read_text(encoding="utf-8"))
    assert parsed.attempt_id == "cli-1"
    assert parsed.task_id == "oscillator_verlet"
    assert parsed.passed_all is True
    assert tuple(parsed.layers) == LAYER_NAMES


def test_cli_exits_zero_on_failing_verdict(tmp_path: Path) -> None:
    workspace = _update_order_candidate(tmp_path / "agent")
    output = tmp_path / "result.json"
    result = CliRunner().invoke(
        app,
        [
            "verify",
            "--task",
            str(TASKS_ROOT / "oscillator"),
            "--candidate",
            str(workspace),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    parsed = VerificationResult.model_validate_json(output.read_text(encoding="utf-8"))
    assert parsed.passed_all is False
    assert parsed.attempt_id == "local"


def test_cli_exits_nonzero_on_infrastructure_error(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    result = CliRunner().invoke(
        app,
        [
            "verify",
            "--task",
            str(TASKS_ROOT / "oscillator"),
            "--candidate",
            str(tmp_path / "missing"),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert not output.exists()
