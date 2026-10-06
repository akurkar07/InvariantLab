"""Layer 3 physical-invariant gates evaluated on a validated output archive."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from invariantlab.schema import GateResult
from invariantlab.verification.analytical import kepler_angular_momentum, kepler_orbital_energy

if TYPE_CHECKING:
    from collections.abc import Mapping

    from invariantlab.schema import TaskContract

# Trusted heat1d/wave1d packages write boundaries as exact zeros (measured 0.0).
DIRICHLET_BOUNDARY_TOLERANCE = 1e-12
# FTCS maximum principle: max|u_T| / max|u_0| <= 1 (trusted heat1d measured 0.79).
HEAT_MAX_PRINCIPLE_BOUND = 1.0
# FTCS L2 decay: ||u_T|| / ||u_0|| < 1 (trusted heat1d measured 0.79).
HEAT_L2_DECAY_BOUND = 1.0
# Trusted wave1d amplitude ratio is 1 - 1.7e-11 over one full period (0.82 at science cases).
WAVE_AMPLITUDE_SLACK = 1e-4


def _gate(name: str, deviation: float, threshold: float, *, strict: bool = True) -> GateResult:
    passed = deviation < threshold if strict else deviation <= threshold
    comparison = "<" if strict else "<="
    return GateResult(
        name=name,
        passed=bool(passed),
        deviation=deviation,
        threshold=threshold,
        detail=f"{name}={deviation:.3e} (required {comparison} {threshold:.3e})",
    )


def _relative_drift(values: np.ndarray) -> float:
    return float(np.max(np.abs(values - values[0])) / abs(values[0]))


def _dirichlet_boundary(state: np.ndarray) -> GateResult:
    deviation = float(max(abs(state[0]), abs(state[-1])))
    return _gate("dirichlet_boundary", deviation, DIRICHLET_BOUNDARY_TOLERANCE, strict=False)


def check_invariants(
    contract: TaskContract,
    parameters: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> list[GateResult]:
    """Return the Layer 3 invariant gates for a Layer 0-validated archive."""
    tolerances = contract.numerics.tolerances
    if contract.id == "oscillator_verlet":
        omega = float(parameters["omega"])  # type: ignore[arg-type]
        state = arrays["state"]
        energy = 0.5 * state[:, 1] ** 2 + 0.5 * omega**2 * state[:, 0] ** 2
        return [
            _gate(
                "energy_relative_drift",
                _relative_drift(energy),
                tolerances["energy_relative_drift"],
            )
        ]
    if contract.id == "kepler_verlet":
        mu = float(parameters["mu"])  # type: ignore[arg-type]
        rows = arrays["state"].tolist()
        energy = np.array([kepler_orbital_energy(vx, vy, rx, ry, mu) for rx, ry, vx, vy in rows])
        angular_momentum = np.array(
            [kepler_angular_momentum(rx, ry, vx, vy) for rx, ry, vx, vy in rows]
        )
        return [
            _gate(
                "energy_relative_drift",
                _relative_drift(energy),
                tolerances["energy_relative_drift"],
            ),
            _gate(
                "angular_momentum_relative_drift",
                _relative_drift(angular_momentum),
                tolerances["angular_momentum_relative_drift"],
            ),
        ]
    if contract.id in ("heat_ftcs", "wave_leapfrog"):
        length = float(parameters["length"])  # type: ignore[arg-type]
        x, state = arrays["x"], arrays["state"]
        initial = np.sin(np.pi * x / length)
        initial[[0, -1]] = 0.0
        amplitude_ratio = float(np.max(np.abs(state)) / np.max(np.abs(initial)))
        if contract.id == "heat_ftcs":
            return [
                _dirichlet_boundary(state),
                _gate("max_principle", amplitude_ratio, HEAT_MAX_PRINCIPLE_BOUND, strict=False),
                _gate(
                    "l2_decay",
                    float(np.linalg.norm(state) / np.linalg.norm(initial)),
                    HEAT_L2_DECAY_BOUND,
                ),
            ]
        return [
            _dirichlet_boundary(state),
            _gate("amplitude_bound", amplitude_ratio, 1.0 + WAVE_AMPLITUDE_SLACK, strict=False),
        ]
    raise ValueError(f"no Layer 3 invariants defined for task id {contract.id!r}")
