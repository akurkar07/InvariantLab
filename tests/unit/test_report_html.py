"""Tests for deterministic, self-contained HTML reports."""

import csv
import hashlib
import json
import shutil
from html.parser import HTMLParser
from pathlib import Path

import pytest
from typer.testing import CliRunner

from invariantlab.cli import app
from invariantlab.reporting import build_report

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests/fixtures/runs/repair-mini"
EXPERIMENT = FIXTURE / "experiment.yaml"


class ReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sample_ids: set[str] = set()
        self.condition_rows: dict[str, dict[str, str]] = {}
        self.condition_links: dict[str, set[str]] = {}
        self._condition: str | None = None
        self._field: str | None = None
        self._row: dict[str, str] = {}
        self._links: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        anchor = attributes.get("id")
        if anchor and anchor.startswith("sample-"):
            self.sample_ids.add(anchor)
        if tag == "tr" and attributes.get("data-condition") is not None:
            self._condition = attributes["data-condition"]
            self._row = {}
            self._links = set()
        if tag == "td" and attributes.get("data-field") is not None:
            self._field = attributes["data-field"]
            self._row[self._field] = ""
        if tag == "a" and self._condition is not None:
            href = attributes.get("href")
            if href and href.startswith("#sample-"):
                self._links.add(href[1:])

    def handle_data(self, data: str) -> None:
        if self._field is not None:
            self._row[self._field] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "td":
            self._field = None
        if tag == "tr" and self._condition is not None:
            self.condition_rows[self._condition] = self._row
            self.condition_links[self._condition] = self._links
            self._condition = None


@pytest.fixture(autouse=True)
def repo_working_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)


def _copy_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    shutil.copytree(FIXTURE, run_dir)
    return run_dir


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_report(path: Path) -> tuple[str, ReportParser]:
    document = path.read_text(encoding="utf-8")
    parser = ReportParser()
    parser.feed(document)
    return document, parser


def test_html_contains_samples_and_condition_values_match_csv_and_summary(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    build_report(EXPERIMENT, FIXTURE, output_dir, html=True)
    _, parser = _parse_report(output_dir / "report.html")
    samples = _read_csv(output_dir / "samples.csv")
    condition_csv = _read_csv(output_dir / "by_condition.csv")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    expected_ids = {
        f"sample-{row['condition']}-{row['trial']}" for row in samples
    }
    assert len(parser.sample_ids) == len(samples)
    assert parser.sample_ids == expected_ids

    expected_fields = {
        "scientific_passes": lambda stats: str(int(stats["scientific_passes"])),
        "n": lambda stats: str(int(stats["completed"])),
        "scientific_pass_rate": lambda stats: f"{stats['scientific_pass_rate']:.6f}",
        "wilson_low": lambda stats: f"{stats['scientific_pass_rate_wilson95'][0]:.6f}",
        "wilson_high": lambda stats: f"{stats['scientific_pass_rate_wilson95'][1]:.6f}",
        "scientific_regressions": lambda stats: str(
            int(stats["scientific_regressions"])
        ),
        "pass_rate_difference_vs_weak": lambda stats: (
            f"{stats['pass_rate_difference_vs_weak']:.6f}"
        ),
    }
    for csv_row in condition_csv:
        condition = csv_row["condition"]
        parsed_row = parser.condition_rows[condition]
        stats = summary["by_condition"][condition]
        for field, expected_value in expected_fields.items():
            assert parsed_row[field] == csv_row[field]
            assert parsed_row[field] == expected_value(stats)
        assert parsed_row["condition"] == csv_row["condition"]
        assert parser.condition_links[condition] == {
            anchor
            for anchor in expected_ids
            if anchor.startswith(f"sample-{condition}-")
        }


def test_html_escapes_untrusted_candidate_content(tmp_path: Path) -> None:
    run_dir = _copy_run(tmp_path)
    events_path = run_dir / "events.jsonl"
    records = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["candidate_source"] = "<script>alert('x')</script>\n"
    records[0]["candidate_error"] = "<b>boom</b>"
    with events_path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    output_dir = tmp_path / "out"
    build_report(EXPERIMENT, run_dir, output_dir, html=True)
    document, _ = _parse_report(output_dir / "report.html")

    assert "<script>" not in document
    assert "<b>boom" not in document
    assert "&lt;script&gt;" in document
    assert "&lt;b&gt;boom" in document


def test_html_is_self_contained_and_reports_baseline_source(tmp_path: Path) -> None:
    output_dir = tmp_path / "fallback"
    build_report(EXPERIMENT, FIXTURE, output_dir, html=True)
    fallback, _ = _parse_report(output_dir / "report.html")
    assert "mutation source tasks/oscillator/mutations/update-order/solver.py" in fallback

    run_dir = _copy_run(tmp_path)
    baseline_path = run_dir / "baseline_solver.py"
    with baseline_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# copied baseline\nbaseline_value = 1\n")
    copied_output = tmp_path / "copied-baseline"
    build_report(EXPERIMENT, run_dir, copied_output, html=True)
    copied, _ = _parse_report(copied_output / "report.html")

    assert "run_dir/baseline_solver.py" in copied
    assert "-# copied baseline" in copied
    for forbidden in ("<script", "<link", " src=", "http://", "https://"):
        assert forbidden not in copied
    assert "<style>" in copied


def test_html_provenance_and_output_are_deterministic(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    result = build_report(EXPERIMENT, FIXTURE, output_dir, html=True)
    report_path = Path(result["paths"]["html"])
    first = report_path.read_bytes()
    document = first.decode("utf-8")
    command = (
        "invariantlab report "
        f"--experiment {EXPERIMENT.as_posix()} "
        f"--run-dir {FIXTURE.as_posix()} "
        f"--output {output_dir.as_posix()} --html"
    )

    assert "repair-mini" in document
    assert "fixture/replay-mini" in document
    assert "<dt>Seed</dt><dd>7</dd>" in document
    assert hashlib.sha256((FIXTURE / "events.jsonl").read_bytes()).hexdigest() in document
    assert command in document

    build_report(EXPERIMENT, FIXTURE, output_dir, html=True)
    assert report_path.read_bytes() == first
    assert b"\r\n" not in first


def test_report_cli_writes_html_only_when_requested(tmp_path: Path) -> None:
    runner = CliRunner()
    html_output = tmp_path / "with-html"
    result = runner.invoke(
        app,
        [
            "report",
            "--experiment",
            str(EXPERIMENT),
            "--run-dir",
            str(FIXTURE),
            "--output",
            str(html_output),
            "--html",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (html_output / "report.html").exists()

    without_html_output = tmp_path / "without-html"
    result_without_html = runner.invoke(
        app,
        [
            "report",
            "--experiment",
            str(EXPERIMENT),
            "--run-dir",
            str(FIXTURE),
            "--output",
            str(without_html_output),
        ],
    )
    assert result_without_html.exit_code == 0, result_without_html.output
    assert not (without_html_output / "report.html").exists()
