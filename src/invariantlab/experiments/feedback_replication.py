"""Backward-compatible Study 2 facade over the generic repair runner."""

from pathlib import Path
from typing import Any

from invariantlab.experiments.repair import (
    CONDITIONS,
    ArtifactIntegrityError,
    _audit_records,
    _build_schedule,
    _condition_context,
    _severity_ratios,
    _summary,
    _write_run_status,
    audit_repair_experiment,
    run_repair_experiment,
)


def audit_feedback_replication(
    config_path: Path,
    run_dir: Path,
    *,
    write_canonical: bool = False,
) -> dict[str, Any]:
    """Audit a Study 2 run through the generic repair audit path."""

    return audit_repair_experiment(
        config_path,
        run_dir,
        write_canonical=write_canonical,
    )


def run_feedback_replication(
    config_path: Path,
    output_dir: Path | None = None,
    max_new_attempts: int | None = None,
) -> Path:
    """Run a Study 2 config through the generic repair runner."""

    return run_repair_experiment(
        config_path,
        output_dir,
        max_new_attempts=max_new_attempts,
    )


__all__ = [
    "CONDITIONS",
    "ArtifactIntegrityError",
    "_audit_records",
    "_build_schedule",
    "_condition_context",
    "_severity_ratios",
    "_summary",
    "_write_run_status",
    "audit_feedback_replication",
    "run_feedback_replication",
]
