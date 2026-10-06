"""Tests for rebuilding repair-run reports from event records."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from invariantlab.cli import app
from invariantlab.config import load_model_config
from invariantlab.experiments.repair import ArtifactIntegrityError
from invariantlab.models import resolve_model_id
from invariantlab.reporting import ReportError, build_report

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests/fixtures/runs/repair-mini"
EXPERIMENT = FIXTURE / "experiment.yaml"
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


@pytest.fixture(autouse=True)
def repo_working_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)


def _copy_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    shutil.copytree(FIXTURE, run_dir)
    return run_dir


def test_report_totals_wilson_intervals_and_stored_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    result = build_report(EXPERIMENT, FIXTURE, output_dir)

    with (output_dir / "by_condition.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == BY_CONDITION_COLUMNS
    weak, metrics = rows
    assert weak == {
        "task": "oscillator",
        "mutation": "update-order",
        "model": "fixture/replay-mini",
        "condition": "weak",
        "n": "2",
        "scientific_passes": "1",
        "scientific_pass_rate": "0.500000",
        "wilson_low": "0.094529",
        "wilson_high": "0.905471",
        "scientific_regressions": "1",
        "median_worst_scientific_ratio": "1.250025",
        "pass_rate_difference_vs_weak": "0.000000",
    }
    assert metrics == {
        "task": "oscillator",
        "mutation": "update-order",
        "model": "fixture/replay-mini",
        "condition": "metrics",
        "n": "2",
        "scientific_passes": "2",
        "scientific_pass_rate": "1.000000",
        "wilson_low": "0.342372",
        "wilson_high": "1.000000",
        "scientific_regressions": "0",
        "median_worst_scientific_ratio": "0.000050",
        "pass_rate_difference_vs_weak": "0.500000",
    }

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    stored_summary = json.loads((FIXTURE / "study-summary.json").read_text(encoding="utf-8"))
    assert summary["source_records"] == 4
    assert (
        summary["events_sha256"]
        == hashlib.sha256((FIXTURE / "events.jsonl").read_bytes()).hexdigest()
    )
    assert summary["by_condition"] == stored_summary["by_condition"]
    assert result["summary"] == summary
    assert not (FIXTURE / "artifact-integrity.json").exists()


def test_samples_are_grouped_by_condition_and_total_counts_match(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    result = build_report(EXPERIMENT, FIXTURE, output_dir)

    with (output_dir / "samples.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == SAMPLE_COLUMNS
    assert len(rows) == 4
    assert [(row["condition"], row["trial"]) for row in rows] == [
        ("weak", "1"),
        ("weak", "2"),
        ("metrics", "1"),
        ("metrics", "2"),
    ]
    with (output_dir / "by_condition.csv").open(encoding="utf-8", newline="") as handle:
        by_condition = list(csv.DictReader(handle))
    assert sum(int(row["n"]) for row in by_condition) == len(rows)
    assert sum(stats["completed"] for stats in result["summary"]["by_condition"].values()) == len(
        rows
    )


def test_report_output_is_deterministic_and_uses_lf(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    build_report(EXPERIMENT, FIXTURE, first_dir)
    build_report(EXPERIMENT, FIXTURE, second_dir)

    for filename in ("summary.json", "by_condition.csv", "samples.csv"):
        first = (first_dir / filename).read_bytes()
        second = (second_dir / filename).read_bytes()
        assert first == second
        assert b"\r\n" not in first


def test_duplicate_record_fails_integrity_without_writing_audit(tmp_path: Path) -> None:
    run_dir = _copy_run(tmp_path)
    events_path = run_dir / "events.jsonl"
    first_line = events_path.read_text(encoding="utf-8").splitlines()[0]
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(first_line + "\n")

    with pytest.raises(ArtifactIntegrityError):
        build_report(EXPERIMENT, run_dir, tmp_path / "out")
    assert not (run_dir / "artifact-integrity.json").exists()


def test_disagreeing_stored_summary_fails(tmp_path: Path) -> None:
    run_dir = _copy_run(tmp_path)
    summary_path = run_dir / "study-summary.json"
    stored_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stored_summary["by_condition"]["weak"]["scientific_passes"] = 2
    summary_path.write_text(
        json.dumps(stored_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ReportError, match=r"weak\.scientific_passes"):
        build_report(EXPERIMENT, run_dir, tmp_path / "out")


def test_report_resolves_model_id_like_runner(tmp_path: Path) -> None:
    """Report tables use the adapter-resolved model id, not the raw config field."""
    output_dir = tmp_path / "out"
    build_report(EXPERIMENT, FIXTURE, output_dir)

    expected = resolve_model_id(load_model_config(FIXTURE / "model.yaml"))
    assert expected == "fixture/replay-mini"
    with (output_dir / "by_condition.csv").open(encoding="utf-8", newline="") as handle:
        assert [row["model"] for row in csv.DictReader(handle)] == [
            expected,
            expected,
        ]


def test_report_rejects_replay_model_without_events_path(tmp_path: Path) -> None:
    """A replay model config that the runner cannot build is rejected."""
    model = tmp_path / "model.yaml"
    model.write_text("adapter: replay\nmodel_id: fixture/replay-mini\n", encoding="utf-8")
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        EXPERIMENT.read_text(encoding="utf-8").replace(
            "tests/fixtures/runs/repair-mini/model.yaml", model.as_posix()
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="events_path"):
        build_report(experiment, FIXTURE, tmp_path / "out")


def test_report_cli_success_and_integrity_failure(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "report",
            "--experiment",
            str(EXPERIMENT),
            "--run-dir",
            str(FIXTURE),
            "--output",
            str(tmp_path / "cli"),
        ],
    )
    assert result.exit_code == 0
    assert "by_condition.csv" in result.output

    run_dir = _copy_run(tmp_path)
    events_path = run_dir / "events.jsonl"
    first_line = events_path.read_text(encoding="utf-8").splitlines()[0]
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(first_line + "\n")
    failed = runner.invoke(
        app,
        [
            "report",
            "--experiment",
            str(EXPERIMENT),
            "--run-dir",
            str(run_dir),
            "--output",
            str(tmp_path / "cli-duplicate"),
        ],
    )
    assert failed.exit_code == 1
