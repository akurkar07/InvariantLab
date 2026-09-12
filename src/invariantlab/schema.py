"""Core data schemas for InvariantLab.

Defines the task contract, verification results, and run manifest structures.
All schemas are Pydantic v2 models for validation and serialization.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Task Contract ────────────────────────────────────────────────────────────


class TaskFamily(str, Enum):
    """V1 problem families."""

    OSCILLATOR = "oscillator"
    KEPLER_TWO_BODY = "kepler_two_body"
    HEAT_1D = "heat_1d"
    WAVE_1D = "wave_1d"


class Language(str, Enum):
    """Supported implementation languages."""

    PYTHON = "python"


class MutationFamily(str, Enum):
    """Controlled defect categories."""

    SIGN_ERROR = "sign_error"
    BOUNDARY_ERROR = "boundary_error"
    DISCRETISATION_ERROR = "discretisation_error"
    STABILITY_ERROR = "stability_error"
    UNIT_ERROR = "unit_error"
    NON_CONSERVATIVE_UPDATE = "non_conservative_update"
    HARD_CODED_SHORTCUT = "hard_coded_shortcut"
    PRECISION_DEFECT = "precision_defect"
    TERMINATION_DEFECT = "termination_defect"


class MutationSpec(BaseModel):
    """A single controlled defect applied to a task."""

    family: MutationFamily
    location: str = Field(..., description="Where the defect is introduced.")
    expected_effect: str = Field(..., description="Expected scientific failure mode.")


class NumericsSpec(BaseModel):
    """Numerical requirements for a task."""

    dtype: Literal["float32", "float64"] = "float64"
    seed: int = 1729
    tolerances: dict[str, float] = Field(default_factory=dict)


class BudgetSpec(BaseModel):
    """Resource limits for agent execution."""

    wall_seconds: int = 900
    model_tokens: int = 32_000


class TaskContract(BaseModel):
    """Complete task specification (the agent-facing contract)."""

    id: str
    family: TaskFamily
    language: Language = Language.PYTHON
    entrypoint: str = "src/solver.py"
    public_tests: str = "tests/public"
    scientific_tests: str = "tests/scientific"
    mutation: MutationSpec | None = None
    budgets: BudgetSpec = Field(default_factory=BudgetSpec)
    numerics: NumericsSpec = Field(default_factory=NumericsSpec)
    description: str = ""


# ── Verification Results ─────────────────────────────────────────────────────


class GateResult(BaseModel):
    """Result of a single verification gate."""

    name: str
    passed: bool
    deviation: float | None = None
    threshold: float | None = None
    detail: str = ""


class VerificationResult(BaseModel):
    """Complete verification result for one attempt."""

    task_id: str
    attempt_id: str
    passed_all: bool
    public_passed: bool
    scientific_passed: bool
    layers: dict[str, list[GateResult]] = Field(default_factory=dict)


# ── Run Manifest ─────────────────────────────────────────────────────────────


class RunManifest(BaseModel):
    """Immutable record of one evaluation run."""

    run_id: str
    experiment: str
    model: str
    seed: int
    container_image: str
    task_results: list[VerificationResult] = Field(default_factory=list)
