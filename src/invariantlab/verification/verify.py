"""Layered verification entrypoint: Layers 0-6 for one candidate workspace."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from invariantlab.schema import GateResult, VerificationResult, load_task_contract
from invariantlab.tasks.workspace import build_evaluation_workspace
from invariantlab.verification.analytical import kepler_circular_orbit
from invariantlab.verification.convergence import check_convergence
from invariantlab.verification.execution import current_executor, run_task
from invariantlab.verification.invariants import check_invariants
from invariantlab.verification.metamorphic import check_metamorphic
from invariantlab.verification.oracles import check_oracle
from invariantlab.verification.robustness import check_robustness

if TYPE_CHECKING:
    from pathlib import Path

LAYER_NAMES = ("L0", "L1", "L2", "L3", "L4", "L5", "L6")
SCIENTIFIC_LAYERS = ("L0", "L2", "L3", "L4", "L5", "L6")


def canonical_parameters(task_id: str) -> dict[str, object]:
    """Return the task's canonical case (the scientific-test parameters)."""
    if task_id == "oscillator_verlet":
        parameters: dict[str, object] = {
            "x0": 0.7,
            "v0": -0.35,
            "omega": 1.7,
            "dt": 1e-3,
            "n_steps": 15_000,
        }
    elif task_id == "kepler_verlet":
        orbit = kepler_circular_orbit(0.0, 2.5, 1.7, 0.37)
        parameters = {
            "rx": orbit[0],
            "ry": orbit[1],
            "vx": orbit[3],
            "vy": orbit[4],
            "mu": 2.5,
            "dt": 0.004,
            "n_steps": 1_080,
        }
    elif task_id == "heat_ftcs":
        parameters = {"nx": 161, "nt": 1_800, "alpha": 0.17, "length": 1.3, "t_final": 0.237}
    elif task_id == "wave_leapfrog":
        parameters = {"nx": 401, "nt": 400, "c": 0.65, "length": 1.3, "t_final": 0.39}
    else:
        raise ValueError(f"no canonical case for task id {task_id!r}")
    return parameters


def _run_public_tests(workspace: Path, public_tests: str, timeout: float) -> GateResult:
    try:
        completed = current_executor().run_public_tests(workspace, public_tests, timeout)
    except subprocess.TimeoutExpired:
        return GateResult(
            name="public_tests", passed=False, detail=f"timed out after {timeout:g} s"
        )
    lines = completed.stdout.strip().splitlines()
    summary = lines[-1] if lines else completed.stderr.strip()[-200:]
    return GateResult(
        name="public_tests",
        passed=completed.returncode == 0,
        detail=f"exit code {completed.returncode}: {summary}",
    )


def verify_candidate(
    task_dir: Path,
    candidate_workspace: Path,
    attempt_id: str,
    work_dir: Path,
) -> VerificationResult:
    """Run Layers 0-6 on a candidate workspace against the trusted task."""
    contract = load_task_contract(task_dir)
    workspace = build_evaluation_workspace(task_dir, candidate_workspace, work_dir / "workspace")
    layers: dict[str, list[GateResult]] = {}

    layers["L1"] = [
        _run_public_tests(workspace, contract.public_tests, contract.budgets.wall_seconds)
    ]

    parameters = canonical_parameters(contract.id)
    run = run_task(task_dir, workspace, parameters, work_dir / "L0")
    layers["L0"] = list(run.gates)
    if run.passed:
        assert run.arrays is not None
        layers["L2"] = check_oracle(contract, parameters, run.arrays)
        layers["L3"] = check_invariants(contract, parameters, run.arrays)
        layers["L4"] = check_convergence(task_dir, workspace, work_dir / "L4")[0]
        layers["L5"] = check_metamorphic(task_dir, workspace, parameters, work_dir / "L5")
        layers["L6"] = check_robustness(task_dir, workspace, work_dir / "L6")
    else:
        skipped = GateResult(name="skipped", passed=False, detail="skipped: L0 failed")
        for layer in SCIENTIFIC_LAYERS[1:]:
            layers[layer] = [skipped]

    public_passed = layers["L1"][0].passed
    scientific_passed = all(gate.passed for layer in SCIENTIFIC_LAYERS for gate in layers[layer])
    return VerificationResult(
        task_id=contract.id,
        attempt_id=attempt_id,
        passed_all=public_passed and scientific_passed,
        public_passed=public_passed,
        scientific_passed=scientific_passed,
        layers={layer: layers[layer] for layer in LAYER_NAMES},
    )
