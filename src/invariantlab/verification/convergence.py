"""Layer 4 convergence gate: rerun a candidate on a 3-level refinement plan."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING

from invariantlab.schema import GateResult, load_task_contract
from invariantlab.verification.analytical import kepler_circular_orbit
from invariantlab.verification.execution import run_task
from invariantlab.verification.oracles import check_oracle

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

ORDER_MIN = 1.8
ORDER_MAX = 2.2


def observed_orders(errors: Sequence[float], ratio: float = 2.0) -> list[float]:
    """Return p = log(E_coarse / E_fine) / log(ratio) for adjacent errors."""
    return [math.log(coarse / fine) / math.log(ratio) for coarse, fine in pairwise(errors)]


def _refinement_plan(task_id: str) -> list[tuple[dict[str, object], dict[str, float]]]:
    """Return per-level (parameters, table row) pairs for the task's 3-level plan."""
    if task_id == "oscillator_verlet":
        # Same case as test_ode_convergence.py; measured through the CLI/NPZ
        # boundary: errors 4.1199165e-05, 1.0301745e-05, 2.5756873e-06; orders
        # 1.999726, 1.999859.
        x0, v0, omega, horizon = 0.7, -0.35, 1.7, 4.32
        return [
            (
                {"x0": x0, "v0": v0, "omega": omega, "dt": horizon / n_steps, "n_steps": n_steps},
                {"level": float(level), "dt": horizon / n_steps},
            )
            for level, n_steps in enumerate((540, 1_080, 2_160))
        ]
    elif task_id == "kepler_verlet":
        # Same case as test_ode_convergence.py; measured through the CLI/NPZ
        # boundary: errors 1.6486982e-05, 4.1190165e-06, 1.0294072e-06; orders
        # 2.000955, 2.000486.
        mu, horizon = 2.5, 4.32
        orbit = kepler_circular_orbit(0.0, mu, 1.7, 0.37)
        return [
            (
                {
                    "rx": orbit[0],
                    "ry": orbit[1],
                    "vx": orbit[3],
                    "vy": orbit[4],
                    "mu": mu,
                    "dt": horizon / n_steps,
                    "n_steps": n_steps,
                },
                {"level": float(level), "dt": horizon / n_steps},
            )
            for level, n_steps in enumerate((540, 1_080, 2_160))
        ]
    elif task_id == "heat_ftcs":
        # CFL-coupled refinement (alpha*dt/dx**2 = 0.4) as in
        # test_pde_convergence.py; measured through the CLI/NPZ boundary:
        # errors 7.1110593e-05, 1.7762056e-05, 4.4395407e-06; orders 2.001266,
        # 2.000316.
        length, alpha, t_final = 1.3, 0.1 * 1.3**2, 0.1
        return [
            (
                {"nx": nx, "nt": nt, "alpha": alpha, "length": length, "t_final": t_final},
                {
                    "level": float(level),
                    "dt": t_final / nt,
                    "dx": length / (nx - 1),
                },
            )
            for level, (nx, nt) in enumerate(((41, 40), (81, 160), (161, 640)))
        ]
    elif task_id == "wave_leapfrog":
        # Fixed Courant 0.3 as in test_pde_convergence.py; measured through
        # the CLI/NPZ boundary: errors 3.5299546e-05, 8.8250708e-06,
        # 2.2062792e-06; orders 1.999970, 1.999992.
        c, length, t_final = 0.75, 1.3, 0.39
        return [
            (
                {"nx": nx, "nt": nt, "c": c, "length": length, "t_final": t_final},
                {
                    "level": float(level),
                    "dt": t_final / nt,
                    "dx": length / (nx - 1),
                },
            )
            for level, (nx, nt) in enumerate(((81, 60), (161, 120), (321, 240)))
        ]
    raise ValueError(f"no convergence plan for task id {task_id!r}")


def check_convergence(
    task_dir: Path,
    candidate_root: Path,
    work_dir: Path,
) -> tuple[list[GateResult], list[dict[str, float]]]:
    """Run the task's refinement plan and return observed-order gates plus a table."""
    contract = load_task_contract(task_dir)
    plan = _refinement_plan(contract.id)
    gate_count = len(plan) - 1

    errors: list[float] = []
    rows: list[dict[str, float]] = []
    for level, (parameters, row) in enumerate(plan):
        run = run_task(task_dir, candidate_root, parameters, work_dir / f"level{level}")
        failure: str | None = None
        if not run.passed:
            failed = next(gate for gate in run.gates if not gate.passed)
            failure = f"level {level} run failed Layer 0 {failed.name}: {failed.detail}"
        else:
            assert run.arrays is not None
            oracle = check_oracle(contract, parameters, run.arrays)[0]
            if oracle.deviation is None:
                failure = f"level {level} oracle: {oracle.detail}"
            elif oracle.deviation <= 0.0:
                failure = f"level {level} oracle error is not positive: {oracle.deviation:.3e}"
            else:
                errors.append(oracle.deviation)
        if failure is not None:
            return (
                [
                    GateResult(
                        name=f"observed_order[{k}]",
                        passed=False,
                        deviation=None,
                        detail=failure,
                    )
                    for k in range(gate_count)
                ],
                rows,
            )
        row["error"] = errors[-1]
        if level >= 1:
            row["order"] = math.log(errors[-2] / errors[-1]) / math.log(2.0)
        rows.append(row)

    gates = []
    for k, order in enumerate(observed_orders(errors)):
        passed = ORDER_MIN <= order <= ORDER_MAX
        gates.append(
            GateResult(
                name=f"observed_order[{k}]",
                passed=passed,
                deviation=order,
                threshold=None,
                detail=(
                    f"order={order:.4f} {'in' if passed else 'outside'} "
                    f"band [1.8, 2.2] (levels {k}->{k + 1})"
                ),
            )
        )
    return gates, rows
