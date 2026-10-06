"""V1-AC5/V1-AC6: report totals match sample records and tables rebuild offline."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import socket
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from invariantlab.cli import app
from invariantlab.config import load_experiment_config
from invariantlab.metrics import wilson_interval
from invariantlab.reporting import ReportError, build_report

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_REL = Path("tests/fixtures/runs/v1-smoke")
FIXTURE = REPO_ROOT / FIXTURE_REL
REPORT_FILES = ("summary.json", "by_condition.csv", "samples.csv")
ROW_KEYS = ("task", "mutation", "condition", "model")
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|secret|password|authorization|bearer\s|sk-[a-z0-9]{8,}|gh[pousr]_[a-z0-9]{8,})"
)


@pytest.fixture(autouse=True)
def repo_working_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[key]) for key in ROW_KEYS)


def assert_report_totals_match_records(run_dir: Path, output_dir: Path) -> None:
    """Rebuild the report for ``run_dir`` and check every total against the records."""

    experiment_path = run_dir / "experiment.yaml"
    experiment = load_experiment_config(experiment_path)
    records = _read_jsonl(run_dir / "events.canonical.jsonl")
    enumerated_cells = {
        (condition, trial)
        for condition in experiment.conditions
        for trial in range(1, experiment.n_attempts + 1)
    }
    assert {(r["condition"], int(r["trial"])) for r in records} == enumerated_cells
    assert len(records) == len(enumerated_cells)

    build_report(experiment_path, run_dir, output_dir)
    record_counts = Counter(_row_key(record) for record in records)

    by_condition = _read_csv(output_dir / "by_condition.csv")
    assert by_condition
    for row in by_condition:
        n = int(row["n"])
        assert n == record_counts[_row_key(row)], row
        assert 0 <= int(row["scientific_passes"]) <= n, row
        assert 0 <= int(row["scientific_regressions"]) <= n, row
    assert sum(int(row["n"]) for row in by_condition) == len(records)

    samples = _read_csv(output_dir / "samples.csv")
    assert Counter(_row_key(row) for row in samples) == record_counts
    assert {(row["condition"], int(row["trial"])) for row in samples} == enumerated_cells

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["source_records"] == len(records)
    for condition, stats in summary["by_condition"].items():
        n = sum(1 for record in records if record["condition"] == condition)
        assert stats["completed"] == n, condition
        assert 0 <= stats["scientific_passes"] <= n, condition
        assert 0 <= stats["public_passes"] <= n, condition
    assert sum(stats["completed"] for stats in summary["by_condition"].values()) == len(records)


@pytest.mark.v1_acceptance("V1-AC5")
def test_report_totals_equal_enumerated_sample_records(tmp_path: Path) -> None:
    assert_report_totals_match_records(FIXTURE_REL, tmp_path / "report")


@pytest.mark.v1_acceptance("V1-AC6")
def test_report_tables_regenerate_offline_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in list(os.environ):
        if name.endswith(("_API_KEY", "_TOKEN")):
            monkeypatch.delenv(name)

    def _no_network(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("network access is disabled for V1-AC6")

    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _no_network)
    with pytest.raises(OSError, match="disabled"):
        socket.create_connection(("127.0.0.1", 9))

    output_dir = tmp_path / "report"
    result = CliRunner().invoke(
        app,
        [
            "report",
            "--experiment",
            (FIXTURE_REL / "experiment.yaml").as_posix(),
            "--run-dir",
            FIXTURE_REL.as_posix(),
            "--output",
            output_dir.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in output_dir.iterdir()) == sorted(REPORT_FILES)
    for name in REPORT_FILES:
        assert (output_dir / name).read_bytes() == (FIXTURE / "expected" / name).read_bytes(), name


def test_wilson_interval_regression_for_identical_counts() -> None:
    low, high = wilson_interval(30, 30) or (None, None)
    assert (round(low, 3), round(high, 3)) == (0.886, 1.0)
    for total in range(1, 31):
        for successes in range(total + 1):
            assert wilson_interval(successes, total) == wilson_interval(successes, total)

    summary = json.loads((FIXTURE / "expected/summary.json").read_text(encoding="utf-8"))
    intervals: dict[tuple[int, int], list[float]] = {}
    for stats in summary["by_condition"].values():
        for kind in ("scientific", "public"):
            counts = (stats[f"{kind}_passes"], stats["completed"])
            interval = stats[f"{kind}_pass_rate_wilson95"]
            assert intervals.setdefault(counts, interval) == interval, counts
    assert len(intervals) < 2 * len(summary["by_condition"])


@pytest.mark.parametrize(
    ("deleted_from", "remove_stored_summary", "error"),
    [
        (("events.canonical.jsonl",), False, AssertionError),
        (("events.jsonl",), True, AssertionError),
        (("events.jsonl", "events.canonical.jsonl"), True, AssertionError),
        (("events.jsonl",), False, ReportError),
    ],
)
def test_v1_ac5_check_fails_when_a_record_is_deleted(
    tmp_path: Path,
    deleted_from: tuple[str, ...],
    remove_stored_summary: bool,
    error: type[Exception],
) -> None:
    run_dir = tmp_path / "run"
    shutil.copytree(FIXTURE, run_dir)
    for name in deleted_from:
        lines = (run_dir / name).read_bytes().splitlines(keepends=True)
        (run_dir / name).write_bytes(b"".join(lines[1:]))
    if remove_stored_summary:
        (run_dir / "study-summary.json").unlink()

    with pytest.raises(error):
        assert_report_totals_match_records(run_dir, tmp_path / "report")


def test_v1_smoke_fixture_is_small_and_has_no_credentials() -> None:
    files = [path for path in FIXTURE.rglob("*") if path.is_file()]
    assert sum(path.stat().st_size for path in files) < 1_000_000
    for path in files:
        match = SECRET_PATTERN.search(path.read_text(encoding="utf-8"))
        assert match is None, f"{path.name}: {match.group(0) if match else ''}"
