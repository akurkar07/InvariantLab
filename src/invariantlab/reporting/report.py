"""Rebuild deterministic repair-run report artifacts from events.jsonl."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from invariantlab.config import load_experiment_config, load_model_config
from invariantlab.experiments import repair
from invariantlab.experiments.repair import ArtifactIntegrityError
from invariantlab.models import resolve_model_id


class ReportError(RuntimeError):
    """Raised when a run cannot be converted into consistent report artifacts."""


BY_CONDITION_COLUMNS = [
    "task",
    "mutation",
    "model",
    "condition",
    "n",
    "scientific_passes",
    "scientific_pass_rate",
    "wilson_low",
    "wilson_high",
    "scientific_regressions",
    "median_worst_scientific_ratio",
    "pass_rate_difference_vs_weak",
]
SAMPLE_COLUMNS = [
    "task",
    "mutation",
    "model",
    "condition",
    "trial",
    "schedule_index",
    "successful_repair",
    "scientific_regression",
    "worst_scientific_ratio",
    "candidate_error",
]


def _format_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(int(value))


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def build_report(
    experiment_config: Path,
    run_dir: Path,
    output_dir: Path,
    html: bool = False,
) -> dict[str, Any]:
    """Validate a repair run and rebuild its summary and CSV tables."""

    experiment = load_experiment_config(experiment_config)
    repair._validate_experiment(experiment)
    model_id = resolve_model_id(load_model_config(Path(experiment.model)))

    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        raise ReportError(f"Run events file not found: {events_path}")
    events_bytes = events_path.read_bytes()
    events_sha256 = hashlib.sha256(events_bytes).hexdigest()
    records = repair._read_existing_events(events_path)
    schedule = repair._build_schedule(
        list(experiment.conditions),
        experiment.n_attempts,
        experiment.seed,
        experiment.randomize_order,
    )
    canonical, audit = repair.audit_records(
        records,
        schedule,
        experiment,
        model_id,
    )
    if not audit["integrity_ok"]:
        raise ArtifactIntegrityError(repair._integrity_error_message(audit))

    baseline = canonical[0]["baseline"] if canonical else {}
    summary = repair.summarize_records(experiment, model_id, baseline, canonical)
    summary["source_records"] = len(canonical)
    summary["events_sha256"] = events_sha256

    stored_summary_path = run_dir / "study-summary.json"
    if stored_summary_path.exists():
        with stored_summary_path.open(encoding="utf-8") as handle:
            stored_summary = json.load(handle)
        stored_by_condition = stored_summary.get("by_condition", {})
        recomputed_by_condition = json.loads(json.dumps(summary["by_condition"]))
        differences: list[str] = []
        for condition in sorted(
            set(stored_by_condition) | set(recomputed_by_condition)
        ):
            if (
                condition not in stored_by_condition
                or condition not in recomputed_by_condition
            ):
                differences.append(condition)
                continue
            stored_stats = stored_by_condition[condition]
            recomputed_stats = recomputed_by_condition[condition]
            for key in sorted(set(stored_stats) | set(recomputed_stats)):
                if (
                    key not in stored_stats
                    or key not in recomputed_stats
                    or stored_stats[key] != recomputed_stats[key]
                ):
                    differences.append(f"{condition}.{key}")
        if differences:
            raise ReportError(
                "study-summary.json by_condition disagrees with events.jsonl: "
                + ", ".join(differences)
            )

    task_id = (
        canonical[0]["task"]
        if canonical
        else Path(experiment.task or "tasks/oscillator").name
    )
    mutation = summary["mutation"]
    by_condition_rows: list[list[Any]] = []
    for condition in experiment.conditions:
        stats = summary["by_condition"][condition]
        interval = stats["scientific_pass_rate_wilson95"]
        by_condition_rows.append(
            [
                task_id,
                mutation,
                model_id,
                condition,
                str(int(stats["completed"])),
                str(int(stats["scientific_passes"])),
                _format_number(stats["scientific_pass_rate"]),
                _format_number(interval[0] if interval is not None else None),
                _format_number(interval[1] if interval is not None else None),
                str(int(stats["scientific_regressions"])),
                _format_number(stats["median_worst_scientific_ratio"]),
                _format_number(stats.get("pass_rate_difference_vs_weak")),
            ]
        )

    condition_order = {
        condition: index for index, condition in enumerate(experiment.conditions)
    }
    ordered_records = sorted(
        canonical,
        key=lambda record: (
            condition_order[record["condition"]],
            int(record["trial"]),
        ),
    )
    sample_rows = [
        [
            record["task"],
            record["mutation"],
            model_id,
            record["condition"],
            str(int(record["trial"])),
            str(int(record["schedule_index"])),
            str(bool(record["successful_repair"])).lower(),
            str(bool(record["scientific_regression"])).lower(),
            _format_number(record["severity"]["worst_scientific_ratio"]),
            record.get("candidate_error", ""),
        ]
        for record in ordered_records
    ]
    condition_total = sum(
        int(summary["by_condition"][condition]["completed"])
        for condition in experiment.conditions
    )
    if condition_total != len(sample_rows):
        raise ReportError(
            f"by_condition n total {condition_total} != samples rows {len(sample_rows)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    by_condition_path = output_dir / "by_condition.csv"
    samples_path = output_dir / "samples.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    _write_csv(by_condition_path, BY_CONDITION_COLUMNS, by_condition_rows)
    _write_csv(samples_path, SAMPLE_COLUMNS, sample_rows)

    paths = {
        "summary": str(summary_path),
        "by_condition": str(by_condition_path),
        "samples": str(samples_path),
    }
    if html:
        from invariantlab.reporting.html import render_html_report

        baseline_path = run_dir / "baseline_solver.py"
        if baseline_path.exists():
            baseline_source = baseline_path.read_text(encoding="utf-8")
            baseline_source_label = "run_dir/baseline_solver.py"
        else:
            _, _, mutation_dir, mutation, baseline_source, _ = (
                repair._resolve_assets(experiment)
            )
            baseline_source_label = (
                f"mutation source {(mutation_dir / mutation.source).as_posix()}"
            )
        by_condition_dicts = [
            dict(zip(BY_CONDITION_COLUMNS, row, strict=True))
            for row in by_condition_rows
        ]
        html_path = output_dir / "report.html"
        rendered = render_html_report(
            experiment_name=experiment.name,
            model_id=model_id,
            seed=experiment.seed,
            events_sha256=events_sha256,
            generation_command=(
                "invariantlab report "
                f"--experiment {experiment_config.as_posix()} "
                f"--run-dir {run_dir.as_posix()} "
                f"--output {output_dir.as_posix()} --html"
            ),
            baseline=baseline,
            baseline_source=baseline_source,
            baseline_source_label=baseline_source_label,
            by_condition_rows=by_condition_dicts,
            records=ordered_records,
        )
        with html_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(rendered)
        paths["html"] = str(html_path)

    return {
        "summary": summary,
        "paths": paths,
    }
