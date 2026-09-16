"""Generic config-driven repair experiment runner."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, cast

from invariantlab.config import (
    ExperimentConfig,
    load_experiment_config,
    load_model_config,
    load_repair_mutation_config,
    load_repair_task_config,
)
from invariantlab.models import (
    ModelConnectionError,
    ModelRateLimitError,
    ModelRequestError,
    build_adapter,
)
from invariantlab.schema import RepairMutationSpec, RepairTaskSpec

CONDITIONS = ("weak", "placebo", "metrics", "interpreted")


def evaluate_candidate(
    source: str,
    task: RepairTaskSpec,
    image: str,
) -> dict[str, Any]:
    """Evaluate candidate source with the verifier declared by the task."""
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required to execute model-generated code safely")

    verifier_path = Path(task.verifier)
    if not verifier_path.exists():
        raise FileNotFoundError(f"Task verifier not found: {verifier_path}")

    with tempfile.TemporaryDirectory(prefix="invariantlab-") as tmp:
        work = Path(tmp)
        (work / task.entrypoint).write_text(source, encoding="utf-8")
        (work / "evaluate.py").write_text(verifier_path.read_text(encoding="utf-8"), encoding="utf-8")
        completed = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--memory",
                "256m",
                "--cpus",
                "1",
                "--pids-limit",
                "64",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=16m",
                "-v",
                f"{work.resolve()}:/work:ro",
                image,
                "python",
                "/work/evaluate.py",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Candidate evaluation failed: {detail}")
        return cast("dict[str, Any]", json.loads(completed.stdout.strip().splitlines()[-1]))


def _extract_python(text: str, function_name: str) -> str:
    import re

    match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + "\n"
    if f"def {function_name}" in text:
        return text.strip() + "\n"
    raise ValueError("Model response did not contain a Python implementation")


def _condition_context(condition: str, task: RepairTaskSpec, baseline: dict[str, Any]) -> str:
    metrics = baseline.get("metrics", {})
    metric_lines = [f"- {key}: {metrics.get(key)!r}" for key in task.metric_keys]
    if condition == "weak":
        return ""
    if condition == "placebo":
        return (
            "\nAdditional evaluator metadata:\n"
            "- the evaluator completed and emitted a valid structured report\n"
            "- the reporting pipeline completed normally\n"
        )
    if condition == "metrics":
        return "\nAdditional verifier output:\n" + "\n".join(metric_lines) + "\n"
    if condition == "interpreted":
        return (
            "\nAdditional verifier output:\n"
            + "\n".join(metric_lines)
            + f"\nThe scientific acceptance threshold is {task.scientific_threshold:g}. "
            "Use these measurements to diagnose the scientific failure rather than only the public tests.\n"
        )
    raise ValueError(f"Unknown feedback condition: {condition}")


def _build_prompt(
    task: RepairTaskSpec,
    mutation: RepairMutationSpec,
    condition: str,
    source: str,
    baseline: dict[str, Any],
) -> str:
    context = _condition_context(condition, task, baseline)
    return (
        "You are repairing scientific Python software.\n\n"
        f"Task: {task.description}\n"
        f"The implementation must provide `{task.function_name}` and preserve its signature.\n"
        "Public checks currently pass, but hidden scientific verification fails.\n"
        f"The injected defect family is not disclosed to you.\n{context}"
        "Return a complete replacement Python file. Do not use third-party packages unless the task requires them.\n\n"
        "Current implementation:\n\n```python\n"
        f"{source.rstrip()}\n```\n"
    )


def _schedule(config: ExperimentConfig) -> list[tuple[str, int]]:
    cells = [
        (condition, trial)
        for condition in config.conditions
        for trial in range(1, config.n_attempts + 1)
    ]
    if config.randomize_order:
        random.Random(config.seed).shuffle(cells)
    return cells


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summary(config: ExperimentConfig, model_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for condition in config.conditions:
        subset = [row for row in records if row["condition"] == condition]
        passed = sum(bool(row["successful_repair"]) for row in subset)
        by_condition[condition] = {
            "completed": len(subset),
            "target": config.n_attempts,
            "scientific_passes": passed,
            "scientific_pass_rate": passed / len(subset) if subset else None,
        }
    target = len(config.conditions) * config.n_attempts
    return {
        "experiment": config.name,
        "model": model_id,
        "target_cells": target,
        "completed_cells": len(records),
        "complete": len(records) == target,
        "by_condition": by_condition,
    }


def _write_status(output: Path, status: str, completed: int, target: int, reason: str = "") -> None:
    (output / "run-status.json").write_text(
        json.dumps(
            {"status": status, "completed_cells": completed, "target_cells": target, "reason": reason},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate(config: ExperimentConfig, task: RepairTaskSpec, mutation: RepairMutationSpec) -> None:
    if config.runner != "generic_repair":
        raise ValueError("Generic repair requires runner: generic_repair")
    if mutation.task != task.id:
        raise ValueError(f"Mutation task {mutation.task!r} does not match task {task.id!r}")
    if not config.conditions:
        raise ValueError("Generic repair requires at least one condition")
    unknown = set(config.conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown feedback conditions: {sorted(unknown)}")


def run_generic_repair(
    config_path: Path,
    output_dir: Path | None = None,
    max_new_attempts: int | None = None,
) -> Path:
    """Run a task/mutation/model/condition repair experiment."""
    config = load_experiment_config(config_path)
    if config.task is None or config.mutation_config is None:
        raise ValueError("generic_repair experiments require task and mutation_config")
    task = load_repair_task_config(Path(config.task))
    mutation = load_repair_mutation_config(Path(config.mutation_config))
    _validate(config, task, mutation)

    model_config = load_model_config(Path(config.model))
    adapter = build_adapter(model_config)
    output = output_dir or Path("runs") / config.name
    output.mkdir(parents=True, exist_ok=True)
    image = "python:3.12-slim" if "placeholder" in config.container_image else config.container_image

    source_path = Path(mutation.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Mutation source not found: {source_path}")
    mutated_source = source_path.read_text(encoding="utf-8")
    baseline = evaluate_candidate(mutated_source, task, image)
    if not baseline.get("public_passed") or baseline.get("scientific_passed"):
        raise RuntimeError("Mutation must pass public checks and fail scientific verification")

    schedule = _schedule(config)
    target = len(schedule)
    events_path = output / "events.jsonl"
    records = _read_events(events_path)
    completed = {(str(row["condition"]), int(row["trial"])) for row in records}
    if len(completed) != len(records):
        raise RuntimeError("Duplicate condition/trial cells found in existing events.jsonl")

    (output / "baseline_solver.py").write_text(mutated_source, encoding="utf-8")
    new_attempts = 0
    for schedule_index, (condition, trial) in enumerate(schedule, start=1):
        if (condition, trial) in completed:
            continue
        if max_new_attempts is not None and new_attempts >= max_new_attempts:
            _write_status(output, "batch_complete", len(records), target, f"Reached max_new_attempts={max_new_attempts}")
            return output

        prompt = _build_prompt(task, mutation, condition, mutated_source, baseline)
        started = time.perf_counter()
        try:
            response = adapter.generate(prompt)
        except ModelRateLimitError as exc:
            _write_status(output, "paused_rate_limit", len(records), target, str(exc))
            return output
        except ModelConnectionError as exc:
            _write_status(output, "paused_connection", len(records), target, str(exc))
            return output
        except ModelRequestError as exc:
            _write_status(output, "paused_provider_error", len(records), target, str(exc))
            return output
        latency = time.perf_counter() - started

        candidate_error = ""
        try:
            candidate = _extract_python(response, task.function_name)
            repaired = evaluate_candidate(candidate, task, image)
        except (RuntimeError, ValueError) as exc:
            candidate = ""
            candidate_error = str(exc)
            repaired = {
                "public": {},
                "scientific": {},
                "public_passed": False,
                "scientific_passed": False,
                "metrics": {"error": candidate_error},
            }

        record = {
            "experiment": config.name,
            "model": adapter.model_id,
            "task": task.id,
            "mutation": mutation.id,
            "mutation_family": mutation.family.value,
            "seed": config.seed,
            "condition": condition,
            "trial": trial,
            "schedule_index": schedule_index,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt": prompt,
            "baseline": baseline,
            "repaired": repaired,
            "successful_repair": bool(repaired.get("scientific_passed")),
            "latency_seconds": latency,
            "response": response,
            "candidate_source": candidate,
            "candidate_error": candidate_error,
        }
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        records.append(record)
        completed.add((condition, trial))
        new_attempts += 1
        (output / "study-summary.json").write_text(
            json.dumps(_summary(config, adapter.model_id, records), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_status(output, "running", len(records), target)

    _write_status(output, "complete", len(records), target)
    return output
