"""Core data schemas for InvariantLab.

Defines the task contract, verification results, and run manifest structures.
All schemas are Pydantic v2 models for validation and serialization.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

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
    candidate_runner: str = "candidate_runner.py"
    prompt_template: str
    feedback_metrics: dict[str, FeedbackMetricSpec] = Field(default_factory=dict)
    interpreted_feedback: str = ""


class ExpectedFailure(ContractModel):
    """A scientific test failure expected from a package mutant."""

    test: str
    message: str

    @field_validator("test")
    @classmethod
    def test_is_scientific_node_id(cls, value: str) -> str:
        """Require a safe scientific test path."""
        from invariantlab.tasks.validation import _has_safe_relative_syntax

        test_file = value.split("::", 1)[0]
        if not value.startswith("tests/scientific/"):
            raise ValueError("expected failure test must start with 'tests/scientific/'")
        if not _has_safe_relative_syntax(test_file):
            raise ValueError("expected failure test must use a safe relative path")
        if not test_file.endswith(".py"):
            raise ValueError("expected failure test file must end with '.py'")
        return value

    @field_validator("message")
    @classmethod
    def message_is_valid_regex(cls, value: str) -> str:
        """Require the expected failure message to be a valid regex."""
        try:
            re.compile(value)
        except re.error as error:
            raise ValueError(f"{value!r} is not a valid regex: {error}") from error
        return value


class MutationDefinition(ContractModel):
    """A mutant manifest; legacy_study supports the old oscillator repair API until #120."""

    id: str
    task_id: str
    family: MutationFamily
    interface: Literal["package", "legacy_study"] = "package"
    source: str = "solver.py"
    expected_effect: str
    expected_failures: list[ExpectedFailure] = Field(default_factory=list)
    max_changed_lines: int = Field(default=10, ge=1)

    @field_validator("expected_effect")
    @classmethod
    def expected_effect_is_nonempty(cls, value: str) -> str:
        """Require a meaningful expected effect."""
        if not value.strip():
            raise ValueError("expected_effect must not be empty")
        return value

    @model_validator(mode="after")
    def package_mutants_declare_expected_failures(self) -> MutationDefinition:
        """Require package mutants to declare a scientific failure."""
        if self.interface == "package" and not self.expected_failures:
            raise ValueError("package mutants must declare at least one expected_failures entry")
        return self


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
