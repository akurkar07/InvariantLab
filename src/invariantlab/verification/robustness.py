"""Layer 6 held-out robustness cases: a fixed table of valid and rejection inputs per task.

The tables live in trusted code only; agent workspaces never contain them. Comments give the
trusted package's ``state_relative_l2`` at each valid case and its margin to the contract
tolerance (1e-5 for all four tasks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from invariantlab.schema import GateResult, load_task_contract
from invariantlab.verification.execution import run_task
from invariantlab.verification.invariants import check_invariants
from invariantlab.verification.oracles import check_oracle

if TYPE_CHECKING:
    from pathlib import Path

Case = tuple[str, dict[str, object]]

OSCILLATOR_VALID_CASES: tuple[Case, ...] = (
    # 3.7e-7 (27x)
    ("oscillator-slow-long", {"x0": 1.2, "v0": 0.4, "omega": 0.6, "dt": 2e-3, "n_steps": 10_000}),
    # 3.3e-7 (30x)
    ("oscillator-fast", {"x0": -0.3, "v0": 1.1, "omega": 4.3, "dt": 2e-4, "n_steps": 20_000}),
    # 4.1e-7 (24x)
    ("oscillator-from-rest", {"x0": 0.0, "v0": -0.9, "omega": 2.6, "dt": 4e-4, "n_steps": 15_000}),
)
OSCILLATOR_REJECTION_CASES: tuple[Case, ...] = (
    (
        "oscillator-negative-omega",
        {"x0": 0.4, "v0": 0.1, "omega": -1.2, "dt": 1e-3, "n_steps": 100},
    ),
)

KEPLER_VALID_CASES: tuple[Case, ...] = (
    # 7.7e-7 (13x)
    (
        "kepler-bound-inner",
        {"rx": 1.0, "ry": 0.3, "vx": -0.2, "vy": 1.1, "mu": 1.3, "dt": 0.001, "n_steps": 4_000},
    ),
    # 1.0e-6 (10x)
    (
        "kepler-wide-retrograde",
        {"rx": -2.4, "ry": 0.5, "vx": -0.15, "vy": -0.8, "mu": 1.6, "dt": 0.004, "n_steps": 2_500},
    ),
    # 9.2e-7 (11x)
    (
        "kepler-eccentric-light",
        {"rx": 0.0, "ry": 1.1, "vx": -0.85, "vy": 0.1, "mu": 0.9, "dt": 0.001, "n_steps": 6_000},
    ),
)
KEPLER_REJECTION_CASES: tuple[Case, ...] = (
    (
        "kepler-nonfinite-dt",
        {"rx": 1.0, "ry": 0.0, "vx": 0.0, "vy": 1.0, "mu": 1.0, "dt": float("nan"), "n_steps": 10},
    ),
)

HEAT_VALID_CASES: tuple[Case, ...] = (
    # r = 0.25; 8.1e-7 (12x)
    (
        "heat-short-rod",
        {"nx": 321, "nt": 8_320, "alpha": 0.05, "length": 0.8, "t_final": 0.26},
    ),
    # r = 0.30; 1.2e-6 (8.6x)
    (
        "heat-long-rod-fast",
        {"nx": 321, "nt": 6_400, "alpha": 0.6, "length": 2.2, "t_final": 0.15},
    ),
    # r = 0.34; 9.7e-7 (10x)
    (
        "heat-unit-rod-fine",
        {"nx": 401, "nt": 9_000, "alpha": 0.21, "length": 1.0, "t_final": 0.09},
    ),
    # r = 0.08; 1.3e-6 (7.8x)
    (
        "heat-small-ratio-long-horizon",
        {"nx": 241, "nt": 12_600, "alpha": 0.09, "length": 1.6, "t_final": 0.5},
    ),
)
HEAT_REJECTION_CASES: tuple[Case, ...] = (
    # r = 0.52
    (
        "heat-unstable-ratio",
        {"nx": 51, "nt": 200, "alpha": 0.1, "length": 1.0, "t_final": 0.416},
    ),
    (
        "heat-negative-alpha",
        {"nx": 51, "nt": 200, "alpha": -0.1, "length": 1.0, "t_final": 0.1},
    ),
)

WAVE_VALID_CASES: tuple[Case, ...] = (
    # Courant 0.56; 1.2e-6 (8.7x)
    (
        "wave-long-domain",
        {"nx": 601, "nt": 900, "c": 0.8, "length": 2.1, "t_final": 2.2},
    ),
    # Courant 0.93; 6.9e-7 (15x)
    (
        "wave-short-domain-fast",
        {"nx": 401, "nt": 820, "c": 2.3, "length": 0.75, "t_final": 0.62},
    ),
    # Courant 0.94; 8.4e-7 (12x)
    (
        "wave-unit-domain-coarse",
        {"nx": 257, "nt": 300, "c": 1.0, "length": 1.0, "t_final": 1.1},
    ),
)
WAVE_REJECTION_CASES: tuple[Case, ...] = (
    # Courant 1.2
    (
        "wave-courant-above-one",
        {"nx": 101, "nt": 50, "c": 1.0, "length": 1.0, "t_final": 0.6},
    ),
)


def _cases(task_id: str) -> tuple[tuple[Case, ...], tuple[Case, ...]]:
    if task_id == "oscillator_verlet":
        cases = OSCILLATOR_VALID_CASES, OSCILLATOR_REJECTION_CASES
    elif task_id == "kepler_verlet":
        cases = KEPLER_VALID_CASES, KEPLER_REJECTION_CASES
    elif task_id == "heat_ftcs":
        cases = HEAT_VALID_CASES, HEAT_REJECTION_CASES
    elif task_id == "wave_leapfrog":
        cases = WAVE_VALID_CASES, WAVE_REJECTION_CASES
    else:
        raise ValueError(f"no robustness cases for task id {task_id!r}")
    return cases


def check_robustness(task_dir: Path, candidate_root: Path, work_dir: Path) -> list[GateResult]:
    """Run every held-out case for the task and return its Layer 6 gates."""
    contract = load_task_contract(task_dir)
    valid_cases, rejection_cases = _cases(contract.id)
    gates: list[GateResult] = []

    for case_id, parameters in valid_cases:
        run = run_task(task_dir, candidate_root, parameters, work_dir / case_id)
        case_gates = list(run.gates)
        if run.passed:
            assert run.arrays is not None
            case_gates += check_oracle(contract, parameters, run.arrays)
            case_gates += check_invariants(contract, parameters, run.arrays)
        gates += [
            gate.model_copy(update={"name": f"robustness[{case_id}]/{gate.name}"})
            for gate in case_gates
        ]

    for case_id, parameters in rejection_cases:
        case_dir = work_dir / case_id
        run = run_task(task_dir, candidate_root, parameters, case_dir)
        archive_written = (case_dir / contract.output.path).exists()
        exited_nonzero = run.returncode is not None and run.returncode != 0
        gates.append(
            GateResult(
                name=f"robustness[{case_id}]/rejected",
                passed=exited_nonzero and not archive_written,
                detail=f"exit code {run.returncode}; archive written: {archive_written}",
            )
        )
    return gates
