"""Acceptance tests for deterministic replay through the real evaluator."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from invariantlab.config import load_experiment_config
from invariantlab.experiments.repair import (
    _build_schedule,
    audit_records,
    run_repair_experiment,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = sorted(
    path.parent for path in (REPO_ROOT / "tests/fixtures/replay").glob("*/experiment.yaml")
)
MANIFEST_FIELDS = (
    "config_sha256",
    "seed",
    "image_digest",
    "package_version",
)
MEASUREMENT_SIGNIFICANT_DIGITS = 12

assert FIXTURES
assert any(path.name == "oscillator-update-order" for path in FIXTURES)


def _require_docker() -> None:
    message = (
        "V1-AC4 requires a working Docker daemon to run the real evaluator; "
        "refusing to skip (a skip would make the release gate falsely green)"
    )
    if shutil.which("docker") is None:
        pytest.fail(message, pytrace=False)
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.fail(f"{message}: {exc}", pytrace=False)
    if result.returncode != 0:
        pytest.fail(message, pytrace=False)


def _verdict(verdict: dict[str, Any]) -> dict[str, Any]:
    return {
        "public_passed": verdict["public_passed"],
        "scientific_passed": verdict["scientific_passed"],
        "public": dict(verdict.get("public", {})),
        "scientific": dict(verdict.get("scientific", {})),
        "metrics": {
            key: float(f"{float(value):.{MEASUREMENT_SIGNIFICANT_DIGITS}g}")
            for key, value in verdict.get("metrics", {}).items()
        },
    }


def evaluator_outcomes(run_dir: Path, config_path: Path) -> dict[str, dict[str, Any]]:
    """Project a repair run onto its evaluator outcomes, one entry per scheduled cell.

    Kept: public_passed, scientific_passed, per-gate pass/fail (`public`/`scientific`),
    per-gate measured values (`metrics`, rounded to MEASUREMENT_SIGNIFICANT_DIGITS) for the
    baseline and the repaired candidate, and the candidate source SHA-256. Everything else
    (timestamps, latency, container ids, absolute paths, run ids) is excluded by construction.
    Canonical cells are selected with `repair.audit_records`.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    experiment = load_experiment_config(config_path)
    schedule = _build_schedule(
        manifest["conditions"],
        int(manifest["n_attempts"]),
        int(manifest["seed"]),
        bool(manifest["randomize_order"]),
    )
    canonical, audit = audit_records(records, schedule, experiment, records[0]["model"])
    assert audit["complete"], f"run records are incomplete or invalid: {audit}"
    return {
        f"{record['condition']}/{record['trial']}": {
            "baseline": _verdict(record["baseline"]),
            "repaired": _verdict(record["repaired"]),
            "candidate_sha256": hashlib.sha256(record["candidate_source"].encode()).hexdigest(),
        }
        for record in canonical
    }


def _run(config_path: Path, output: Path) -> Path:
    run_repair_experiment(config_path, output)
    status = json.loads((output / "run-status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete", (
        f"repair run did not complete: {status['status']}; reason: {status.get('reason', '')}"
    )
    return output


@pytest.mark.integration
@pytest.mark.v1_acceptance("V1-AC4")
@pytest.mark.parametrize("fixture_dir", FIXTURES, ids=lambda path: path.name)
def test_repeated_replay_yields_identical_evaluator_outcomes(
    fixture_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    _require_docker()
    config_path = fixture_dir / "experiment.yaml"
    run_a = _run(config_path, tmp_path / "run-a")
    run_b = _run(config_path, tmp_path / "run-b")

    outcomes_a = evaluator_outcomes(run_a, config_path)
    outcomes_b = evaluator_outcomes(run_b, config_path)
    assert outcomes_a
    assert outcomes_a == outcomes_b

    manifest_a = json.loads((run_a / "manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((run_b / "manifest.json").read_text(encoding="utf-8"))
    for field in MANIFEST_FIELDS:
        assert manifest_a[field] == manifest_b[field], f"manifest field differs: {field}"
    contract_hash_a = manifest_a["artifact_sha256"]["contract_yaml"]
    contract_hash_b = manifest_b["artifact_sha256"]["contract_yaml"]
    assert contract_hash_a == contract_hash_b
    assert contract_hash_a
    assert manifest_a["image_digest"]

    if fixture_dir.name == "oscillator-update-order":
        scientific_passes = {
            outcome["repaired"]["scientific_passed"] for outcome in outcomes_a.values()
        }
        assert scientific_passes == {False, True}


@pytest.mark.integration
@pytest.mark.v1_acceptance("V1-AC4")
def test_perturbed_replay_response_changes_evaluator_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    _require_docker()
    fixture_dir = REPO_ROOT / "tests/fixtures/replay/oscillator-update-order"
    config_path = fixture_dir / "experiment.yaml"
    events_path = tmp_path / "events.jsonl"
    shutil.copyfile(fixture_dir / "events.jsonl", events_path)
    records = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trial_one = next(
        record for record in records if record["condition"] == "weak" and record["trial"] == 1
    )
    trial_two = next(
        record for record in records if record["condition"] == "weak" and record["trial"] == 2
    )
    trial_two["response"] = trial_one["response"]
    events_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    model_config = fixture_dir / "model.yaml"
    perturbed_model = tmp_path / "model.yaml"
    model_text = model_config.read_text(encoding="utf-8")
    original_events_path = "tests/fixtures/replay/oscillator-update-order/events.jsonl"
    assert original_events_path in model_text
    perturbed_model.write_text(
        model_text.replace(original_events_path, json.dumps(str(events_path))),
        encoding="utf-8",
    )

    perturbed_config = tmp_path / "experiment.yaml"
    config_text = config_path.read_text(encoding="utf-8")
    original_model_path = "tests/fixtures/replay/oscillator-update-order/model.yaml"
    assert original_model_path in config_text
    perturbed_config.write_text(
        config_text.replace(original_model_path, json.dumps(str(perturbed_model))),
        encoding="utf-8",
    )

    original_run = _run(config_path, tmp_path / "original-run")
    perturbed_run = _run(perturbed_config, tmp_path / "perturbed-run")
    original_outcomes = evaluator_outcomes(original_run, config_path)
    perturbed_outcomes = evaluator_outcomes(perturbed_run, perturbed_config)
    assert original_outcomes != perturbed_outcomes
    assert original_outcomes["weak/2"] != perturbed_outcomes["weak/2"]
