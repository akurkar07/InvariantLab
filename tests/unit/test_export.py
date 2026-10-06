"""Tests for exporting repair runs as local Hugging Face datasets."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from invariantlab.cli import app
from invariantlab.reporting import (
    EXPORT_FIELDS,
    CredentialLeakError,
    build_report,
    export_hf_dataset,
)
from invariantlab.reporting.export import PROVENANCE_FIELDS

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests/fixtures/runs/repair-mini"
EXPERIMENT = FIXTURE / "experiment.yaml"


@pytest.fixture(autouse=True)
def repo_working_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)


def _copy_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    shutil.copytree(FIXTURE, run_dir)
    return run_dir


def _read_rows(output_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (output_dir / "data" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _rewrite_event(run_dir: Path, index: int, field: str, value: str) -> None:
    events_path = run_dir / "events.jsonl"
    lines = events_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[index])
    record[field] = value
    lines[index] = json.dumps(record)
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_export_row_count_and_schema(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    report = build_report(EXPERIMENT, FIXTURE, report_dir)
    output_dir = tmp_path / "out"
    result = export_hf_dataset(EXPERIMENT, FIXTURE, output_dir)

    rows = _read_rows(output_dir)
    assert result["rows"] == report["summary"]["source_records"] == 4
    assert len(rows) == 4
    for row in rows:
        assert list(row) == EXPORT_FIELDS
        assert list(row["provenance"]) == PROVENANCE_FIELDS
        provenance = row["provenance"]
        assert provenance["model_id"] == "fixture/replay-mini"
        assert provenance["adapter"] == "replay"
        assert provenance["container_image"] == "python:3.12-slim"
        assert provenance["git_sha"] is None
        assert provenance["package_version"] is None
        assert provenance["image_digest"] is None
        assert (
            provenance["events_sha256"]
            == hashlib.sha256((FIXTURE / "events.jsonl").read_bytes()).hexdigest()
        )

    contract_sha256 = hashlib.sha256(
        (REPO_ROOT / "tasks/oscillator/contract.yaml").read_bytes()
    ).hexdigest()
    assert all(row["contract_sha256"] == contract_sha256 for row in rows)
    assert all(row["candidate_diff"].startswith("--- baseline_solver.py") for row in rows)
    assert [(row["condition"], row["trial"]) for row in rows] == [
        ("weak", 1),
        ("weak", 2),
        ("metrics", 1),
        ("metrics", 2),
    ]


def test_export_provenance_from_manifest(tmp_path: Path) -> None:
    run_dir = _copy_run(tmp_path)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "git_commit": "abc123",
                "package_version": "0.1.0",
                "container_image": "python:3.12-slim",
                "image_digest": "python@sha256:" + "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    export_hf_dataset(EXPERIMENT, run_dir, output_dir)

    provenance = _read_rows(output_dir)[0]["provenance"]
    assert provenance["git_sha"] == "abc123"
    assert provenance["package_version"] == "0.1.0"
    assert provenance["container_image"] == "python:3.12-slim"
    assert provenance["image_digest"] == "python@sha256:" + "0" * 64


def test_export_dataset_card(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    export_hf_dataset(EXPERIMENT, FIXTURE, output_dir)
    card = (output_dir / "README.md").read_text(encoding="utf-8")

    front_matter = card.split("---")[1]
    metadata = yaml.safe_load(front_matter)
    assert metadata == {
        "configs": [{"config_name": "default", "data_files": "data/samples.jsonl"}],
        "license": "mit",
    }
    assert "4 rows" in card
    assert "weak" in card
    assert "metrics" in card
    assert "repair-mini" in card


def test_export_baseline_solver_source(tmp_path: Path) -> None:
    from invariantlab.config import load_experiment_config
    from invariantlab.experiments import repair

    experiment = load_experiment_config(EXPERIMENT)
    _, _, _, _, mutation_source, _ = repair._resolve_assets(experiment)

    output_dir = tmp_path / "out"
    export_hf_dataset(EXPERIMENT, FIXTURE, output_dir)
    assert (output_dir / "baseline_solver.py").read_text(encoding="utf-8") == mutation_source

    run_dir = _copy_run(tmp_path)
    (run_dir / "baseline_solver.py").write_text("# stored baseline\n", encoding="utf-8")
    output_dir2 = tmp_path / "out2"
    export_hf_dataset(EXPERIMENT, run_dir, output_dir2)
    assert (output_dir2 / "baseline_solver.py").read_text(encoding="utf-8") == "# stored baseline\n"


def test_export_refuses_env_secret_in_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canary = "canary-value-1234567890"
    monkeypatch.setenv("INVARIANTLAB_CANARY_API_KEY", canary)
    run_dir = _copy_run(tmp_path)
    _rewrite_event(run_dir, 0, "response", f"leak {canary} here")

    output_dir = tmp_path / "out"
    with pytest.raises(CredentialLeakError):
        export_hf_dataset(EXPERIMENT, run_dir, output_dir)
    assert not output_dir.exists()


def test_export_refuses_token_shape_in_candidate_source(tmp_path: Path) -> None:
    run_dir = _copy_run(tmp_path)
    _rewrite_event(run_dir, 1, "candidate_source", "tok = 'ghp_" + "a1" * 18 + "'")

    with pytest.raises(CredentialLeakError):
        export_hf_dataset(EXPERIMENT, run_dir, tmp_path / "out")


def test_export_cli_success_and_canary_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export-hf",
            "--experiment",
            str(EXPERIMENT),
            "--run-dir",
            str(FIXTURE),
            "--output",
            str(tmp_path / "cli"),
        ],
    )
    assert result.exit_code == 0
    assert "OK" in result.output
    assert "4" in result.output

    monkeypatch.setenv("INVARIANTLAB_CANARY_API_KEY", "canary-value-1234567890")
    run_dir = _copy_run(tmp_path)
    _rewrite_event(run_dir, 0, "response", "leak canary-value-1234567890")
    failed = runner.invoke(
        app,
        [
            "export-hf",
            "--experiment",
            str(EXPERIMENT),
            "--run-dir",
            str(run_dir),
            "--output",
            str(tmp_path / "cli-canary"),
        ],
    )
    assert failed.exit_code == 1
    assert "FAIL" in failed.output
    assert not (tmp_path / "cli-canary").exists()


def test_export_is_deterministic_and_uses_lf(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    export_hf_dataset(EXPERIMENT, FIXTURE, first_dir)
    export_hf_dataset(EXPERIMENT, FIXTURE, second_dir)

    for relative in ("data/samples.jsonl", "README.md", "baseline_solver.py"):
        first = (first_dir / relative).read_bytes()
        second = (second_dir / relative).read_bytes()
        assert first == second
        assert b"\r\n" not in first


def test_export_loads_with_datasets(tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")
    output_dir = tmp_path / "out"
    export_hf_dataset(EXPERIMENT, FIXTURE, output_dir)
    dataset = datasets.load_dataset(
        "json",
        data_files=str(output_dir / "data" / "samples.jsonl"),
        split="train",
    )
    assert len(dataset) == 4


def test_fixture_itself_passes_credential_scan(tmp_path: Path) -> None:
    from invariantlab.reporting.export import scan_for_credentials

    findings = scan_for_credentials(
        {"events.jsonl": (FIXTURE / "events.jsonl").read_text(encoding="utf-8")},
        [],
    )
    assert findings == []
