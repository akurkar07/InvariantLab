"""V1-AC8: the exported dataset has full provenance and no credentials."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from invariantlab.reporting import export_hf_dataset
from invariantlab.reporting.export import (
    EXPORT_FIELDS,
    PROVENANCE_FIELDS,
    scan_for_credentials,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_REL = Path("tests/fixtures/runs/v1-smoke")

# Required non-empty fields of every exported sample, grouped by the V1-AC8
# category they prove. Dotted names are keys inside the row's provenance dict.
REQUIRED_SAMPLE_FIELDS: dict[str, tuple[str, ...]] = {
    "task metadata": ("task", "contract_sha256"),
    "trajectory": ("prompt", "response"),
    "patch": ("candidate_source",),
    "measurements": ("baseline", "repaired", "severity"),
    "provenance": (
        "seed",
        "provenance.git_sha",
        "provenance.package_version",
        "provenance.model_id",
        "provenance.adapter",
        "provenance.image_digest",
        "provenance.experiment_config_sha256",
    ),
}

ENV_NAME_PATTERNS = ("*KEY*", "*TOKEN*", "*SECRET*", "*PASSWORD*")
SENSITIVE_FIELDS = {"authorization", "api_key"}
TOKEN_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_"),
)
ENV_CANARY = "sk-test-canary"
# Token-shaped canaries are assembled at runtime so this file holds no token literal.
TOKEN_CANARIES = (
    "sk-" + "A" * 20,
    "hf_" + "B" * 24,
    "ghp_" + "C" * 24,
    "github_pat_" + "D" * 8,
)


@pytest.fixture(autouse=True)
def repo_working_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def exported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("INVARIANTLAB_API_KEY", ENV_CANARY)
    monkeypatch.setenv("HF_TOKEN", TOKEN_CANARIES[1])
    output_dir = tmp_path / "dataset"
    export_hf_dataset(FIXTURE_REL / "experiment.yaml", FIXTURE_REL, output_dir)
    return output_dir


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _lookup(row: dict[str, Any], dotted: str) -> Any:
    value: Any = row
    for part in dotted.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _has_sensitive_field(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).lower() in SENSITIVE_FIELDS or _has_sensitive_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_sensitive_field(item) for item in value)
    return False


def scan_export(output_dir: Path) -> list[str]:
    """Return credential findings for every file under ``output_dir``."""

    texts = {
        path.relative_to(output_dir).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    records: list[Any] = []
    for name, text in texts.items():
        if name.endswith(".jsonl"):
            records += [json.loads(line) for line in text.splitlines() if line.strip()]
        elif name.endswith(".json"):
            records.append(json.loads(text))

    env_values = [
        (name, value)
        for name, value in os.environ.items()
        if value and any(fnmatch.fnmatchcase(name.upper(), p) for p in ENV_NAME_PATTERNS)
    ]
    findings = scan_for_credentials(texts, records)
    for name, text in texts.items():
        findings += [
            f"{name}: value of env var {env}" for env, value in env_values if value in text
        ]
        findings += [f"{name}: {p.pattern}" for p in TOKEN_PATTERNS if p.search(text)]
    if _has_sensitive_field(records):
        findings.append("Authorization or api_key field")
    return findings


def test_required_fields_match_export_schema() -> None:
    for fields in REQUIRED_SAMPLE_FIELDS.values():
        for field in fields:
            top, _, nested = field.partition(".")
            assert top in EXPORT_FIELDS, field
            assert not nested or nested in PROVENANCE_FIELDS, field


@pytest.mark.v1_acceptance("V1-AC8")
def test_exported_records_have_full_provenance(exported: Path) -> None:
    samples = _read_jsonl(exported / "data/samples.jsonl")
    canonical = _read_jsonl(FIXTURE_REL / "events.canonical.jsonl")
    assert canonical
    assert len(samples) == len(canonical)

    for index, row in enumerate(samples):
        assert set(row) == set(EXPORT_FIELDS), index
        for category, fields in REQUIRED_SAMPLE_FIELDS.items():
            for field in fields:
                assert not _is_empty(_lookup(row, field)), (index, category, field)
        for outcome in ("baseline", "repaired"):
            assert row[outcome]["metrics"], (index, outcome)
            assert isinstance(row[outcome]["public_passed"], bool), (index, outcome)
            assert isinstance(row[outcome]["scientific_passed"], bool), (index, outcome)
        assert isinstance(row["successful_repair"], bool), index
        assert isinstance(row["scientific_regression"], bool), index


@pytest.mark.v1_acceptance("V1-AC8")
def test_exported_dataset_passes_credential_scan(exported: Path) -> None:
    files = [path for path in exported.rglob("*") if path.is_file()]
    assert {path.relative_to(exported).as_posix() for path in files} >= {
        "data/samples.jsonl",
        "README.md",
    }
    assert os.environ["INVARIANTLAB_API_KEY"] == ENV_CANARY
    assert scan_export(exported) == []


@pytest.mark.v1_acceptance("V1-AC8")
@pytest.mark.parametrize(
    "inject",
    [
        pytest.param(lambda row: row.update(response=row["response"] + ENV_CANARY), id="env"),
        pytest.param(lambda row: row["provenance"].update(Authorization="x"), id="authz"),
        pytest.param(lambda row: row["provenance"].update(api_key="x"), id="api_key"),
        *[
            pytest.param(
                lambda row, token=token: row.update(candidate_source=token),
                id=token.split("_")[0].split("-")[0],
            )
            for token in TOKEN_CANARIES
        ],
    ],
)
def test_credential_scan_fails_on_injected_canary(
    exported: Path, tmp_path: Path, inject: Any
) -> None:
    tampered = tmp_path / "tampered"
    shutil.copytree(exported, tampered)
    samples_path = tampered / "data/samples.jsonl"
    rows = _read_jsonl(samples_path)
    inject(rows[0])
    samples_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert scan_export(exported) == []
    assert scan_export(tampered)
