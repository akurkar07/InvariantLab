"""Configuration loading and management for InvariantLab."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from invariantlab.schema import RepairMutationSpec, RepairTaskSpec


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    name: str
    description: str = ""
    task_suite: str | None = Field(default=None, description="Path to task suite config.")
    task: str | None = Field(default=None, description="Path to generic repair task config.")
    mutation_config: str | None = Field(
        default=None, description="Path to generic repair mutation config."
    )
    model: str = Field(..., description="Model adapter identifier.")
    runner: str = "first_model"
    mutation: str | None = None
    conditions: list[str] = Field(default_factory=list)
    randomize_order: bool = False
    n_attempts: int = Field(default=1, ge=1)
    seed: int = Field(default=42, ge=0)
    container_image: str = Field(
        default="python:3.12-slim", description="Container image digest or tag."
    )


class TaskSuiteConfig(BaseModel):
    """A named set of tasks for evaluation."""

    name: str
    tasks: list[str] = Field(default_factory=list)
    description: str = ""


class ModelConfig(BaseModel):
    """Configuration for a model backend."""

    adapter: str = Field(..., description="Adapter identifier (e.g., 'anthropic', 'hf', 'openai').")
    model_id: str = ""
    temperature: float = 0.0
    max_tokens: int = 32_000
    extra: dict[str, Any] = Field(default_factory=dict)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top level of {path}")
    return data


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment configuration."""
    return ExperimentConfig(**load_yaml(path))


def load_task_suite_config(path: Path) -> TaskSuiteConfig:
    """Load and validate a task suite configuration."""
    return TaskSuiteConfig(**load_yaml(path))


def load_model_config(path: Path) -> ModelConfig:
    """Load and validate a model configuration."""
    return ModelConfig(**load_yaml(path))


def load_repair_task_config(path: Path) -> RepairTaskSpec:
    """Load a generic repair task definition."""
    return RepairTaskSpec(**load_yaml(path))


def load_repair_mutation_config(path: Path) -> RepairMutationSpec:
    """Load a generic repair mutation definition."""
    return RepairMutationSpec(**load_yaml(path))
