"""Replication study for feedback effects on a subtle oscillator repair."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

from invariantlab.config import ExperimentConfig, load_experiment_config, load_model_config
from invariantlab.experiments.first_model import _evaluate_in_docker, _extract_python
from invariantlab.models import (
    ModelConnectionError,
    ModelRateLimitError,
    ModelRequestError,
    build_adapter,
)

CONDITIONS = ("weak", "placebo", "metrics", "interpreted")

UPDATE_ORDER_SOLVER = """def solve_oscillator_verlet(x0, v0, omega, dt, n_steps):
    trajectory = [(0.0, float(x0), float(v0))]
    x = float(x0)
    v = float(v0)
    omega2 = float(omega) * float(omega)
    t = 0.0

    for _ in range(int(n_steps)):
        a = -omega2 * x
        v_half = v + 0.5 * dt * a
        x = x + dt * v_half
        a_new = -omega2 * x
        v = v_half + 0.5 * dt * a
        t += dt
        trajectory.append((t, x, v))

    return trajectory
"""

PROMPT_TEMPLATE = """You are repairing a numerical solver for the one-dimensional
harmonic oscillator

    d2x/dt2 = -omega^2 x

The intended method is velocity Verlet. The public tests only check interface behaviour,
the initial state, finiteness over a very short trajectory and a loose one-step sanity
check. They currently pass. Hidden scientific verification checks long-horizon agreement
with the analytical solution and conservation of total energy.
{condition_context}
Return a complete replacement solver.py. Do not change the function signature and do not
use third-party packages.

Current solver.py:

```python
{source}
```
"""


def _condition_context(condition: str, baseline: dict[str, Any]) -> str:
    """Return the condition-specific information shown to the model."""

    metrics = baseline.get("metrics", {})
    state_error = float(metrics["max_state_relative_error"])
    energy_drift = float(metrics["max_energy_relative_drift"])

    if condition == "weak":
        return ""
    if condition == "placebo":
        return (
            "\nAdditional run metadata from the current solver:\n"
            "- the evaluator completed and emitted a valid structured report\n"
            "- result rows were serialisable and the reporting pipeline completed normally\n"
        )
    if condition == "metrics":
        return (
            "\nAdditional verifier output from the current solver:\n"
            f"- max state relative error: {state_error:.12g}\n"
            f"- max energy relative drift: {energy_drift:.12g}\n"
        )
    if condition == "interpreted":
        return (
            "\nAdditional verifier output from the current solver:\n"
            f"- max state relative error: {state_error:.12g}\n"
            f"- max energy relative drift: {energy_drift:.12g}\n"
            "Both metrics exceed the hidden scientific acceptance threshold of 1e-3. "
            "The state metric measures long-horizon disagreement with the analytical "
            "solution, while the energy metric measures failure to conserve total energy.\n"
        )
    raise ValueError(f"Unknown feedback condition: {condition}")


def _build_schedule(
    conditions: list[str], n_attempts: int, seed: int, randomize_order: bool
) -> list[tuple[str, int]]:
    """Build a balanced condition-by-trial schedule."""

    schedule = [
        (condition, trial)
        for condition in conditions
        for trial in range(1, n_attempts + 1)
    ]
    if randomize_order:
        random.Random(seed).shuffle(schedule)
    return schedule


def _ratio(value: Any, baseline: Any) -> float | None:
    try:
        numerator = float(value)
        denominator = float(baseline)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
        return None
    return numerator / denominator


def _severity_ratios(
    baseline: dict[str, Any], repaired: dict[str, Any]
) -> dict[str, float | None]:
    """Compare repaired scientific errors with the original defect."""

    baseline_metrics = baseline.get("metrics", {})
    repaired_metrics = repaired.get("metrics", {})
    state_ratio = _ratio(
        repaired_metrics.get("max_state_relative_error"),
        baseline_metrics.get("max_state_relative_error"),
    )
    energy_ratio = _ratio(
        repaired_metrics.get("max_energy_relative_drift"),
        baseline_metrics.get("max_energy_relative_drift"),
    )
    finite = [value for value in (state_ratio, energy_ratio) if value is not None]
    return {
        "state_error_ratio": state_ratio,
        "energy_drift_ratio": energy_ratio,
        "worst_scientific_ratio": max(finite) if finite else None,
    }


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""

    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1.0 + (z * z / total)
    centre = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total)))
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _read_existing_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _write_run_status(
    output: Path,
    *,
    status: str,
    completed: int,
    target: int,
    reason: str = "",
) -> None:
    payload = {
        "status": status,
        "completed_cells": completed,
        "target_cells": target,
        "reason": reason,
    }
    (output / "run-status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _failed_repair_result(error: Exception) -> dict[str, Any]:
    return {
        "public": {},
        "scientific": {},
        "public_passed": False,
        "scientific_passed": False,
        "metrics": {"error": f"{type(error).__name__}: {error}"},
    }


def _summary(
    experiment: ExperimentConfig,
    model_id: str,
    baseline: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    weak_rate: float | None = None

    for condition in experiment.conditions:
        subset = [record for record in records if record["condition"] == condition]
        passed = sum(bool(record["successful_repair"]) for record in subset)
        total = len(subset)
        regressions = sum(bool(record["scientific_regression"]) for record in subset)
        ratios = [
            float(record["severity"]["worst_scientific_ratio"])
            for record in subset
            if record["severity"]["worst_scientific_ratio"] is not None
        ]
        low, high = _wilson_interval(passed, total)
        rate = passed / total if total else None
        by_condition[condition] = {
            "completed": total,
            "target": experiment.n_attempts,
            "scientific_passes": passed,
            "scientific_pass_rate": rate,
            "scientific_pass_rate_wilson95": [low, high],
            "scientific_regressions": regressions,
            "median_worst_scientific_ratio": statistics.median(ratios) if ratios else None,
            "max_worst_scientific_ratio": max(ratios) if ratios else None,
        }
        if condition == "weak":
            weak_rate = rate

    if weak_rate is not None:
        for stats in by_condition.values():
            rate = stats["scientific_pass_rate"]
            stats["pass_rate_difference_vs_weak"] = (
                None if rate is None else rate - weak_rate
            )

    target_total = len(experiment.conditions) * experiment.n_attempts
    return {
        "experiment": experiment.name,
        "model": model_id,
        "mutation": experiment.mutation,
        "hypotheses": {
            "H1": (
                "Diagnostic scientific feedback changes repair behaviour on subtle "
                "update-order defects."
            ),
            "H2": (
                "Diagnostic feedback increases repairs that remove the local defect but "
                "introduce a scientifically worse state-evolution failure."
            ),
        },
        "primary_endpoint": "scientific_pass_rate",
        "secondary_endpoints": [
            "scientific_regression_rate",
            "state_error_ratio",
            "energy_drift_ratio",
            "worst_scientific_ratio",
        ],
        "baseline": baseline,
        "target_cells": target_total,
        "completed_cells": len(records),
        "complete": len(records) == target_total,
        "by_condition": by_condition,
    }


def _write_summary(
    path: Path,
    experiment: ExperimentConfig,
    model_id: str,
    baseline: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    path.write_text(
        json.dumps(
            _summary(experiment, model_id, baseline, records),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_experiment(experiment: ExperimentConfig) -> None:
    if experiment.runner != "feedback_replication":
        raise ValueError("Feedback replication requires runner: feedback_replication")
    if experiment.mutation != "update-order":
        raise ValueError("Study 2 currently supports only mutation: update-order")
    if not experiment.conditions:
        raise ValueError("Feedback replication requires at least one condition")
    unknown = set(experiment.conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown feedback conditions: {sorted(unknown)}")
    if len(set(experiment.conditions)) != len(experiment.conditions):
        raise ValueError("Feedback conditions must not contain duplicates")


def run_feedback_replication(
    config_path: Path,
    output_dir: Path | None = None,
    max_new_attempts: int | None = None,
) -> Path:
    """Run the update-order feedback replication study with resumable evidence logging."""

    if max_new_attempts is not None and max_new_attempts < 1:
        raise ValueError("max_new_attempts must be at least 1")

    experiment = load_experiment_config(config_path)
    _validate_experiment(experiment)
    model_config = load_model_config(Path(experiment.model))
    adapter = build_adapter(model_config)
    output = output_dir or Path("runs") / experiment.name
    output.mkdir(parents=True, exist_ok=True)

    image = experiment.container_image
    if "placeholder" in image:
        image = "python:3.12-slim"

    baseline = _evaluate_in_docker(UPDATE_ORDER_SOLVER, image)
    if not baseline["public_passed"] or baseline["scientific_passed"]:
        raise RuntimeError("Update-order baseline no longer has the intended public/scientific gap")

    events_path = output / "events.jsonl"
    summary_path = output / "study-summary.json"
    (output / "baseline_solver.py").write_text(UPDATE_ORDER_SOLVER, encoding="utf-8")

    records = _read_existing_events(events_path)
    completed = {(str(record["condition"]), int(record["trial"])) for record in records}
    schedule = _build_schedule(
        list(experiment.conditions),
        experiment.n_attempts,
        experiment.seed,
        experiment.randomize_order,
    )
    target_total = len(schedule)
    new_attempts = 0
    _write_summary(summary_path, experiment, adapter.model_id, baseline, records)
    _write_run_status(
        output,
        status="running",
        completed=len(records),
        target=target_total,
    )

    for schedule_index, (condition, trial) in enumerate(schedule, start=1):
        if (condition, trial) in completed:
            continue

        if max_new_attempts is not None and new_attempts >= max_new_attempts:
            _write_run_status(
                output,
                status="batch_complete",
                completed=len(records),
                target=target_total,
                reason=f"Reached max_new_attempts={max_new_attempts}",
            )
            return output

        condition_context = _condition_context(condition, baseline)
        prompt = PROMPT_TEMPLATE.format(
            condition_context=condition_context,
            source=UPDATE_ORDER_SOLVER,
        )
        started = time.perf_counter()
        try:
            response = adapter.generate(prompt)
        except ModelRateLimitError as exc:
            _write_run_status(
                output,
                status="paused_rate_limit",
                completed=len(records),
                target=target_total,
                reason=str(exc),
            )
            return output
        except ModelConnectionError as exc:
            _write_run_status(
                output,
                status="paused_connection",
                completed=len(records),
                target=target_total,
                reason=str(exc),
            )
            return output
        except ModelRequestError as exc:
            _write_run_status(
                output,
                status="paused_provider_error",
                completed=len(records),
                target=target_total,
                reason=str(exc),
            )
            return output

        latency = time.perf_counter() - started
        candidate_error = ""
        try:
            repaired_source = _extract_python(response)
            repaired = _evaluate_in_docker(repaired_source, image)
        except (RuntimeError, ValueError) as exc:
            repaired_source = ""
            repaired = _failed_repair_result(exc)
            candidate_error = str(exc)

        severity = _severity_ratios(baseline, repaired)
        worst_ratio = severity["worst_scientific_ratio"]
        scientific_regression = bool(
            not repaired["scientific_passed"]
            and worst_ratio is not None
            and worst_ratio > 1.0
        )

        record = {
            "experiment": experiment.name,
            "model": adapter.model_id,
            "seed": experiment.seed,
            "mutation": experiment.mutation,
            "condition": condition,
            "trial": trial,
            "schedule_index": schedule_index,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt": prompt,
            "baseline": baseline,
            "repaired": repaired,
            "severity": severity,
            "successful_repair": bool(repaired["scientific_passed"]),
            "scientific_regression": scientific_regression,
            "needs_manual_failure_review": not bool(repaired["scientific_passed"]),
            "latency_seconds": latency,
            "response": response,
            "candidate_source": repaired_source,
            "candidate_error": candidate_error,
        }
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

        records.append(record)
        completed.add((condition, trial))
        new_attempts += 1
        _write_summary(summary_path, experiment, adapter.model_id, baseline, records)
        _write_run_status(
            output,
            status="running",
            completed=len(records),
            target=target_total,
        )

    _write_run_status(
        output,
        status="complete",
        completed=len(records),
        target=target_total,
    )
    return output
