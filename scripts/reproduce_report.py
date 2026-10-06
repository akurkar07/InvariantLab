"""Run the deterministic Docker smoke experiment and reproduce its report."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from invariantlab.experiments.repair import run_repair_experiment
from invariantlab.reporting.report import build_report

BY_CONDITION_FIELDS = (
    "completed",
    "target",
    "public_passes",
    "scientific_passes",
    "scientific_regressions",
    "public_pass_rate",
    "scientific_pass_rate",
    "public_pass_rate_wilson95",
    "scientific_pass_rate_wilson95",
    "pass_rate_difference_vs_weak",
    "verification_gap",
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def project(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep only stable, scientifically meaningful summary fields."""
    return {
        "baseline": {
            "public_passed": summary["baseline"]["public_passed"],
            "scientific_passed": summary["baseline"]["scientific_passed"],
        },
        "completed_cells": summary["completed_cells"],
        "target_cells": summary["target_cells"],
        "source_records": summary["source_records"],
        "by_condition": {
            condition: {field: stats[field] for field in BY_CONDITION_FIELDS}
            for condition, stats in summary["by_condition"].items()
        },
    }


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and value:
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else key
            flattened.update(flatten(item, child))
        return flattened
    return {prefix: value}


def compare(expected: Any, actual: Any) -> list[str]:
    expected_flat = flatten(expected)
    actual_flat = flatten(actual)
    differences = []
    for key in sorted(expected_flat.keys() | actual_flat.keys()):
        if key not in expected_flat:
            differences.append(
                f"  {key}: expected <missing>, got {json.dumps(actual_flat[key], sort_keys=True)}"
            )
        elif key not in actual_flat:
            differences.append(
                f"  {key}: expected {json.dumps(expected_flat[key], sort_keys=True)}, got <missing>"
            )
        elif expected_flat[key] != actual_flat[key]:
            differences.append(
                f"  {key}: expected "
                f"{json.dumps(expected_flat[key], sort_keys=True)}, got "
                f"{json.dumps(actual_flat[key], sort_keys=True)}"
            )
    return differences


def fail(reason: str) -> int:
    print(f"FAIL: {reason}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    os.chdir(REPO_ROOT)
    parser = argparse.ArgumentParser(
        description="Rebuild the deterministic Docker replay smoke report."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/replay-smoke.yaml"),
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=Path("tests/fixtures/expected/replay-smoke-summary.json"),
    )
    parser.add_argument("--update-expected", action="store_true")
    args = parser.parse_args(argv)
    if not args.smoke:
        parser.error("--smoke is required")

    experiment = args.experiment if args.experiment.is_absolute() else REPO_ROOT / args.experiment
    expected = args.expected if args.expected.is_absolute() else REPO_ROOT / args.expected
    output = (
        args.output
        if args.output is not None
        else Path(tempfile.mkdtemp(prefix="invariantlab-smoke-"))
    )
    if not output.is_absolute():
        output = REPO_ROOT / output
    output = output.resolve()

    if shutil.which("docker") is None:
        return fail("Docker is unavailable")
    try:
        docker_info = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return fail("Docker is unavailable")
    if docker_info.returncode != 0:
        return fail("Docker is unavailable")

    try:
        if (output / "run" / "events.jsonl").exists():
            return fail("output dir must be fresh")
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return fail(f"could not prepare output dir: {exc}")
    run_dir = output / "run"
    report_dir = output / "report"

    try:
        run_repair_experiment(experiment, output_dir=run_dir)
        status = json.loads((run_dir / "run-status.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return fail(f"experiment run failed: {exc}")
    if status.get("status") != "complete":
        return fail(
            f"experiment did not complete: status={status.get('status')!r}, "
            f"reason={status.get('reason', '')!r}"
        )

    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        events_path = run_dir / "events.jsonl"
        event_bytes = events_path.read_bytes()
        events = [
            json.loads(line) for line in event_bytes.decode("utf-8").splitlines() if line.strip()
        ]
        print(f"Docker image digest: {manifest['image_digest']}")
        for event in events:
            repaired = event["repaired"]
            print(
                f"{event['condition']} {event['trial']} "
                f"public={repaired['public_passed']} "
                f"scientific={repaired['scientific_passed']}"
            )
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return fail(f"could not read run evidence: {exc}")

    try:
        summary = build_report(experiment, run_dir, report_dir)["summary"]
        projection = json.loads(json.dumps(project(summary)))
    except Exception as exc:
        return fail(f"report reconstruction failed: {exc}")

    if args.update_expected:
        try:
            expected.parent.mkdir(parents=True, exist_ok=True)
            with expected.open("w", encoding="utf-8", newline="") as handle:
                handle.write(json.dumps(projection, indent=2, sort_keys=True) + "\n")
        except OSError as exc:
            return fail(f"could not update expected summary: {exc}")
    else:
        try:
            expected_summary = json.loads(expected.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return fail(f"could not read expected summary {expected}: {exc}")
        differences = compare(
            json.loads(json.dumps(expected_summary)),
            json.loads(json.dumps(projection)),
        )
        if differences:
            print(
                f"FAIL: summary differs from {expected}:\n" + "\n".join(differences),
                file=sys.stderr,
            )
            return 1

    record_count = len(event_bytes.splitlines())
    try:
        run_repair_experiment(experiment, output_dir=run_dir)
        rerun_status = json.loads((run_dir / "run-status.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return fail(f"idempotent rerun failed: {exc}")
    if rerun_status.get("status") != "complete":
        return fail(
            f"idempotent rerun did not complete: "
            f"status={rerun_status.get('status')!r}, "
            f"reason={rerun_status.get('reason', '')!r}"
        )
    try:
        rerun_events = (run_dir / "events.jsonl").read_bytes()
    except OSError as exc:
        return fail(f"could not reread events after rerun: {exc}")
    rerun_record_count = len(rerun_events.splitlines())
    if rerun_events != event_bytes:
        return fail(
            "events.jsonl changed on rerun "
            f"({record_count} records before, {rerun_record_count} after)"
        )
    try:
        rerun_summary = build_report(experiment, run_dir, report_dir)["summary"]
    except Exception as exc:
        return fail(f"report rebuild after rerun failed: {exc}")
    if rerun_summary != summary:
        return fail("summary changed after idempotent rerun")

    print(
        f"OK: replay smoke run reproduced {expected} ({record_count} records); "
        f"run: {run_dir}; report: {report_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
