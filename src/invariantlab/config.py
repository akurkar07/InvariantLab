"""Configuration loading and management for InvariantLab."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    name: str
    description: str = ""
    task_suite: str = Field(..., description="Path to task suite config.")
    model: str = Field(..., description="Model adapter identifier.")
    runner: str = "first_model"
    task: str | None = Field(
        default=None,
        description="Task directory used by config-driven repair runners.",
    )
    mutation: str | None = None
    conditions: list[str] = Field(default_factory=list)
    randomize_order: bool = False
    n_attempts: int = Field(default=1, ge=1)
    seed: int = Field(default=42, ge=0)
    container_image: str = Field(
        default="python:3.12-slim", description="Container image digest or tag."
    )

    @field_validator("container_image")
    @classmethod
    def _reject_placeholder_image(cls, value: str) -> str:
        return validate_container_image(value)


def validate_container_image(image: str) -> str:
    """Reject empty or placeholder image references instead of substituting them."""

    if not image.strip():
        raise ValueError("container_image must not be empty")
    if "placeholder" in image:
        raise ValueError(
            f"container_image {image!r} is a placeholder; set a real image tag or "
            "digest (e.g. python:3.12-slim@sha256:<digest>)"
        )
    return image


class TaskSuiteConfig(BaseModel):
    """A named set of tasks for evaluation."""

    name: str
    tasks: list[str] = Field(default_factory=list)
    description: str = ""


class ModelConfig(BaseModel):
    """Configuration for a model backend."""

    adapter: str = Field(
        ..., description="Adapter identifier: 'replay', 'ollama' or 'openai_compatible'."
    )
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
    data = load_yaml(path)
    return ExperimentConfig(**data)


def load_task_suite_config(path: Path) -> TaskSuiteConfig:
    """Load and validate a task suite configuration."""
    data = load_yaml(path)
    return TaskSuiteConfig(**data)


def load_model_config(path: Path) -> ModelConfig:
    """Load and validate a model configuration."""
    data = load_yaml(path)
    return ModelConfig(**data)
