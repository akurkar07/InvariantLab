"""Core data schemas for InvariantLab.

Defines task contracts, generic repair experiment types, verification results, and run
manifest structures. All schemas are Pydantic v2 models for validation and serialisation.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


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


class MutationSpec(BaseModel):
    """A single controlled defect applied to a task contract."""

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
    """Complete task specification used by the benchmark task package."""

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


class RepairTaskSpec(BaseModel):
    """Config-driven task definition for model repair experiments."""

    id: str
    family: TaskFamily
    description: str
    verifier: str = Field(..., description="Path to an executable scientific verifier.")
    entrypoint: str = "solver.py"
    function_name: str
    metric_keys: list[str] = Field(default_factory=list)
    scientific_threshold: float = 1e-3


class RepairMutationSpec(BaseModel):
    """A concrete source mutation used by a repair experiment."""

    id: str
    task: str
    family: MutationFamily
    source: str = Field(..., description="Path to the mutated implementation shown to the model.")
    location: str
    expected_effect: str


class RepairExperimentSpec(BaseModel):
    """Resolved task/mutation/model/condition inputs for one generic repair study."""

    task: RepairTaskSpec
    mutation: RepairMutationSpec
    model: str
    conditions: list[str] = Field(default_factory=list)
    n_attempts: int = Field(default=1, ge=1)
    seed: int = Field(default=1729, ge=0)
    randomize_order: bool = True


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


class RunManifest(BaseModel):
    """Immutable record of one evaluation run."""

    run_id: str
    experiment: str
    model: str
    seed: int
    container_image: str
    task_results: list[VerificationResult] = Field(default_factory=list)


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
