"""Layer 2 oracle-comparison gate for the trusted V1 tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from invariantlab.schema import GateResult
from invariantlab.verification.analytical import (
    heat_trajectory,
    oscillator_trajectory,
    wave_standing_trajectory,
)
from invariantlab.verification.kepler_oracle import solve_kepler_high_accuracy

if TYPE_CHECKING:
    from invariantlab.schema import TaskContract

GATE_NAME = "state_relative_l2"


def _exact_state(
    contract_id: str,
    parameters: dict[str, object],
    arrays: dict[str, np.ndarray],
) -> np.ndarray:
    p = parameters
    if contract_id == "oscillator_verlet":
        x0 = float(cast("float", p["x0"]))
        v0 = float(cast("float", p["v0"]))
        omega = float(cast("float", p["omega"]))
        return oscillator_trajectory(arrays["time"], x0, v0, omega)[:, 1:]
    elif contract_id == "kepler_verlet":
        rx = float(cast("float", p["rx"]))
        ry = float(cast("float", p["ry"]))
        vx = float(cast("float", p["vx"]))
        vy = float(cast("float", p["vy"]))
        mu = float(cast("float", p["mu"]))
        return solve_kepler_high_accuracy(rx, ry, vx, vy, mu, arrays["time"])
    elif contract_id == "heat_ftcs":
        t_final = float(cast("float", p["t_final"]))
        alpha = float(cast("float", p["alpha"]))
        length = float(cast("float", p["length"]))
        return heat_trajectory(arrays["x"], t_final, alpha, length=length)
    elif contract_id == "wave_leapfrog":
        t_final = float(cast("float", p["t_final"]))
        c = float(cast("float", p["c"]))
        length = float(cast("float", p["length"]))
        return wave_standing_trajectory(arrays["x"], t_final, c, length=length)
    raise ValueError(f"no oracle for task contract '{contract_id}'")


def check_oracle(
    contract: TaskContract,
    parameters: dict[str, object],
    arrays: dict[str, np.ndarray],
) -> list[GateResult]:
    threshold = contract.numerics.tolerances[GATE_NAME]
    if contract.id == "kepler_verlet":
        try:
            exact = _exact_state(contract.id, parameters, arrays)
        except (ValueError, RuntimeError) as error:
            return [
                GateResult(
                    name=GATE_NAME,
                    passed=False,
                    deviation=None,
                    threshold=threshold,
                    detail=f"oracle failed: {type(error).__name__}: {error}",
                )
            ]
    else:
        exact = _exact_state(contract.id, parameters, arrays)

    state = arrays["state"]
    if state.shape != exact.shape:
        return [
            GateResult(
                name=GATE_NAME,
                passed=False,
                deviation=None,
                threshold=threshold,
                detail=f"state shape {state.shape} does not match oracle shape {exact.shape}",
            )
        ]

    exact_norm = np.linalg.norm(exact)
    if exact_norm == 0:
        return [
            GateResult(
                name=GATE_NAME,
                passed=False,
                deviation=None,
                threshold=threshold,
                detail="oracle state has zero norm",
            )
        ]

    deviation = float(np.linalg.norm(state - exact) / exact_norm)
    return [
        GateResult(
            name=GATE_NAME,
            passed=deviation < threshold,
            deviation=deviation,
            threshold=threshold,
            detail=f"state_relative_l2={deviation:.3e} (threshold {threshold:.1e})",
        )
    ]
