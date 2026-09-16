"""Core data schemas for InvariantLab.

Defines the task contract, verification results, and run manifest structures.
All schemas are Pydantic v2 models for validation and serialization.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
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


class FeedbackMetricSpec(BaseModel):
    """Metadata for one scientific metric exposed in feedback conditions."""

    label: str
    threshold: float | None = None
    interpretation: str = ""


class TaskDefinition(BaseModel):
    """Evaluator-facing task metadata for generic repair experiments."""

    id: str
    family: TaskFamily
    contract: str = "contract.yaml"
    verifier: str
    prompt_template: str
    feedback_metrics: dict[str, FeedbackMetricSpec] = Field(default_factory=dict)
    interpreted_feedback: str = ""


class MutationDefinition(BaseModel):
    """A config-driven controlled defect used by a repair experiment."""

    id: str
    task_id: str
    family: MutationFamily
    source: str
    expected_effect: str = ""


class ExperimentDefinition(BaseModel):
    """Generic task, mutation, model and condition binding."""

    task: str
    mutation: str
    model: str
    conditions: list[str] = Field(default_factory=list)
    n_attempts: int = Field(default=1, ge=1)
    seed: int = Field(default=42, ge=0)
    randomize_order: bool = False
    container_image: str = "python:3.12-slim"


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


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def load_task_contract(task_dir: str | Path) -> TaskContract:
    """Load a task contract from a directory containing contract.yaml."""

    task_path = Path(task_dir) if isinstance(task_dir, str) else task_dir
    contract_file = task_path / "contract.yaml"
    if not contract_file.exists():
        raise FileNotFoundError(f"No contract.yaml in {task_path}")
    return TaskContract(**_load_yaml_mapping(contract_file))


def load_task_definition(task_dir: str | Path) -> TaskDefinition:
    """Load evaluator-facing metadata from task.yaml."""

    task_path = Path(task_dir) if isinstance(task_dir, str) else task_dir
    definition_file = task_path / "task.yaml"
    if not definition_file.exists():
        raise FileNotFoundError(f"No task.yaml in {task_path}")
    definition = TaskDefinition(**_load_yaml_mapping(definition_file))
    contract = load_task_contract(task_path)
    if definition.id != contract.id:
        raise ValueError(
            f"Task definition id {definition.id!r} does not match contract id {contract.id!r}"
        )
    if definition.family != contract.family:
        raise ValueError("Task definition family does not match the task contract")
    return definition


def load_mutation_definition(mutation_dir: str | Path) -> MutationDefinition:
    """Load a controlled defect definition from mutation.yaml."""

    mutation_path = Path(mutation_dir) if isinstance(mutation_dir, str) else mutation_dir
    definition_file = mutation_path / "mutation.yaml"
    if not definition_file.exists():
        raise FileNotFoundError(f"No mutation.yaml in {mutation_path}")
    return MutationDefinition(**_load_yaml_mapping(definition_file))
