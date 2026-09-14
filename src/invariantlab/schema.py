"""Core data schemas for InvariantLab.

Defines the task contract, verification results, and run manifest structures.
All schemas are Pydantic v2 models for validation and serialization.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

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
    UPDATE_ORDER_ERROR = "update_order_error"
    BOUNDARY_ERROR = "boundary_error"
    DISCRETISATION_ERROR = "discretisation_error"
    STABILITY_ERROR = "stability_error"
    UNIT_ERROR = "unit_error"
    NON_CONSERVATIVE_UPDATE = "non_conservative_update"
    HARD_CODED_SHORTCUT = "hard_coded_shortcut"
    PRECISION_DEFECT = "precision_defect"
    TERMINATION_DEFECT = "termination_defect"


class ContractModel(BaseModel):
    """Base model for fail-closed task-contract declarations."""

    model_config = ConfigDict(extra="forbid")


class MutationSpec(ContractModel):
    """A single controlled defect applied to a task."""

    family: MutationFamily
    location: str = Field(..., description="Where the defect is introduced.")
    expected_effect: str = Field(..., description="Expected scientific failure mode.")


class NumericsSpec(ContractModel):
    """Numerical requirements for a task."""

    dtype: Literal["float32", "float64"] = "float64"
    seed: int = 1729
    tolerances: dict[str, float] = Field(default_factory=dict)


class BudgetSpec(ContractModel):
    """Resource limits for agent execution."""

    wall_seconds: int = 900
    model_tokens: int = 32_000


class OutputArray(ContractModel):
    """One array declared in a task's NPZ output archive."""

    name: Annotated[str, StringConstraints(min_length=1)]
    shape: list[Annotated[StrictInt, Field(gt=0)] | None] = Field(min_length=1)
    dtype: Literal["float64"]

    @field_validator("name")
    @classmethod
    def name_has_no_surrounding_whitespace(cls, value: str) -> str:
        """Reject ambiguous NPZ keys instead of normalizing them."""
        if value != value.strip():
            raise ValueError("output array name must not have surrounding whitespace")
        return value


class OutputSpec(ContractModel):
    """The exact NPZ archive produced by a task entrypoint."""

    path: Annotated[str, StringConstraints(min_length=1)]
    arrays: list[OutputArray] = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def path_is_exact_npz_filename(cls, value: str) -> str:
        """Require an unnormalized NPZ archive declaration."""
        if value != value.strip():
            raise ValueError("output path must not have surrounding whitespace")
        if not value.endswith(".npz"):
            raise ValueError("output path must name an .npz archive")
        return value

    @model_validator(mode="after")
    def array_names_are_unique(self) -> OutputSpec:
        """Keep NPZ key declarations unambiguous."""
        names = [array.name for array in self.arrays]
        if len(names) != len(set(names)):
            raise ValueError("output array names must be unique")
        return self


class TaskContract(ContractModel):
    """Complete task specification (the agent-facing contract)."""

    id: str
    family: TaskFamily
    language: Language = Language.PYTHON
    entrypoint: str = "src/solver.py"
    public_tests: str = "tests/public"
    scientific_tests: str = "tests/scientific"
    output: OutputSpec
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


# ── Loading ──────────────────────────────────────────────────────────────────


def load_task_contract(task_dir: str | Path) -> TaskContract:
    """Load a task contract from a directory containing contract.yaml."""
    import yaml

    task_path = Path(task_dir) if isinstance(task_dir, str) else task_dir
    contract_file = task_path / "contract.yaml"
    if not contract_file.exists():
        raise FileNotFoundError(f"No contract.yaml in {task_path}")
    with contract_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return TaskContract(**data)
