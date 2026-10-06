"""Export a repair run as a local Hugging Face-loadable dataset."""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from invariantlab.config import load_model_config
from invariantlab.experiments import repair
from invariantlab.reporting.report import _format_number, load_canonical_run


class ExportError(RuntimeError):
    """Raised when a run cannot be exported as a dataset."""


class CredentialLeakError(ExportError):
    """Raised when the export content matches a credential pattern."""


EXPORT_FIELDS = [
    "task",
    "contract_sha256",
    "mutation",
    "condition",
    "trial",
    "schedule_index",
    "seed",
    "prompt",
    "prompt_sha256",
    "response",
    "candidate_source",
    "candidate_diff",
    "baseline",
    "repaired",
    "severity",
    "successful_repair",
    "scientific_regression",
    "candidate_error",
    "provenance",
]
PROVENANCE_FIELDS = [
    "git_sha",
    "package_version",
    "model_id",
    "adapter",
    "container_image",
    "image_digest",
    "experiment_config_sha256",
    "events_sha256",
]

_ENV_NAME_PATTERNS = ["*KEY*", "*TOKEN*", "*SECRET*", "*PASSWORD*"]
_MIN_ENV_VALUE_LENGTH = 8
_SENSITIVE_KEYS = {"authorization", "api_key"}
_TEXT_PATTERNS = [
    ("authorization header", re.compile(r"authorization\s*:\s*\S", re.IGNORECASE)),
    ("api_key assignment", re.compile(r"api_key\s*[=:]\s*\S", re.IGNORECASE)),
]
_TOKEN_PATTERNS = [
    ("openai-style token", re.compile(r"sk-[A-Za-z0-9_-]{16,}")),
    ("hugging face token", re.compile(r"hf_[A-Za-z0-9]{16,}")),
    ("github token", re.compile(r"ghp_[A-Za-z0-9]{20,}")),
    ("github pat", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
]


def _env_secret_values() -> list[tuple[str, str]]:
    secrets: list[tuple[str, str]] = []
    for name, value in os.environ.items():
        upper = name.upper()
        if not any(fnmatch.fnmatchcase(upper, pattern) for pattern in _ENV_NAME_PATTERNS):
            continue
        if len(value) < _MIN_ENV_VALUE_LENGTH:
            continue
        secrets.append((name, value))
    return secrets


def _find_sensitive_keys(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _SENSITIVE_KEYS:
                return True
            if _find_sensitive_keys(item):
                return True
    elif isinstance(value, list):
        return any(_find_sensitive_keys(item) for item in value)
    return False


def scan_for_credentials(texts: dict[str, str], rows: list[dict]) -> list[str]:
    """Return credential findings as '<file>: <rule>' strings (no secret values)."""

    findings: list[str] = []
    env_secrets = _env_secret_values()
    for name, text in texts.items():
        for env_name, env_value in env_secrets:
            if env_value in text:
                findings.append(f"{name}: value of env var {env_name}")
        for rule, pattern in _TEXT_PATTERNS + _TOKEN_PATTERNS:
            if pattern.search(text):
                findings.append(f"{name}: {rule}")
    if _find_sensitive_keys(rows):
        findings.append("data/samples.jsonl: sensitive dict key")
    return findings


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_card(
    experiment_name: str,
    row_count: int,
    summary: dict[str, Any],
    conditions: list[str],
    task_id: str,
    mutation: str,
    contract_sha256: str,
    provenance: dict[str, Any],
) -> str:
    front_matter = yaml.safe_dump(
        {
            "configs": [{"config_name": "default", "data_files": "data/samples.jsonl"}],
            "license": "mit",
        },
        sort_keys=False,
    )
    lines = [
        "---",
        front_matter.rstrip("\n"),
        "---",
        "",
        f"# InvariantLab repair run: {experiment_name}",
        "",
        f"{row_count} rows in `data/samples.jsonl`.",
        "",
        "| condition | n | scientific_passes | scientific_pass_rate | "
        "wilson95_low | wilson95_high | scientific_regressions | "
        "median_worst_scientific_ratio |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for condition in conditions:
        stats = summary["by_condition"][condition]
        interval = stats["scientific_pass_rate_wilson95"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                condition,
                int(stats["completed"]),
                int(stats["scientific_passes"]),
                _format_number(stats["scientific_pass_rate"]),
                _format_number(interval[0] if interval is not None else None),
                _format_number(interval[1] if interval is not None else None),
                int(stats["scientific_regressions"]),
                _format_number(stats["median_worst_scientific_ratio"]),
            )
        )
    lines += [
        "",
        "## Provenance",
        "",
        f"- task: {task_id}",
        f"- mutation: {mutation}",
        f"- contract_sha256: {contract_sha256}",
    ]
    for field in PROVENANCE_FIELDS:
        value = provenance.get(field)
        lines.append(f"- {field}: {json.dumps(value)}")
    lines += [
        "",
        "## Files",
        "",
        "- `data/samples.jsonl` — one JSON object per canonical scheduled record",
        "- `baseline_solver.py` — baseline solver source used for candidate_diff",
        "",
        'Load with `datasets.load_dataset("json", data_files="data/samples.jsonl")`.',
        "",
    ]
    return "\n".join(lines)


def export_hf_dataset(
    experiment_config: Path,
    run_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Write a Hugging Face-loadable dataset directory for a repair run."""

    experiment, model_id, canonical, summary = load_canonical_run(experiment_config, run_dir)
    assets = repair._resolve_assets(experiment)
    mutation_source = assets.mutation_source
    contract_sha256 = hashlib.sha256((assets.task_dir / "contract.yaml").read_bytes()).hexdigest()

    baseline_path = run_dir / "baseline_solver.py"
    if baseline_path.exists():
        baseline = baseline_path.read_text(encoding="utf-8")
    else:
        baseline = mutation_source

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}

    adapter = load_model_config(Path(experiment.model)).adapter
    provenance = {
        "git_sha": manifest.get("git_commit"),
        "package_version": manifest.get("package_version"),
        "model_id": model_id,
        "adapter": adapter,
        "container_image": manifest.get("container_image") or experiment.container_image,
        "image_digest": manifest.get("image_digest"),
        "experiment_config_sha256": hashlib.sha256(
            Path(experiment_config).read_bytes()
        ).hexdigest(),
        "events_sha256": summary["events_sha256"],
    }

    condition_order = {condition: index for index, condition in enumerate(experiment.conditions)}
    ordered_records = sorted(
        canonical,
        key=lambda record: (
            condition_order[record["condition"]],
            int(record["trial"]),
        ),
    )

    rows: list[dict[str, Any]] = []
    for record in ordered_records:
        candidate_diff = "".join(
            difflib.unified_diff(
                baseline.splitlines(keepends=True),
                record["candidate_source"].splitlines(keepends=True),
                fromfile="baseline_solver.py",
                tofile="candidate_solver.py",
            )
        )
        values = {
            "contract_sha256": contract_sha256,
            "candidate_diff": candidate_diff,
            "candidate_error": record.get("candidate_error", ""),
            "provenance": provenance,
        }
        rows.append({field: values.get(field, record.get(field)) for field in EXPORT_FIELDS})

    task_id = ordered_records[0]["task"] if ordered_records else assets.contract.id
    card = _build_card(
        experiment.name,
        len(rows),
        summary,
        list(experiment.conditions),
        task_id,
        summary["mutation"],
        contract_sha256,
        provenance,
    )
    samples_text = "".join(json.dumps(row) + "\n" for row in rows)

    findings = scan_for_credentials(
        {
            "data/samples.jsonl": samples_text,
            "README.md": card,
            "baseline_solver.py": baseline,
        },
        rows,
    )
    if findings:
        raise CredentialLeakError("Credential scan found potential secrets: " + "; ".join(findings))

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "samples": data_dir / "samples.jsonl",
        "readme": output_dir / "README.md",
        "baseline": output_dir / "baseline_solver.py",
    }
    contents = {
        "samples": samples_text,
        "readme": card,
        "baseline": baseline,
    }
    for key, path in paths.items():
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(contents[key])

    return {"rows": len(rows), "paths": {k: str(p) for k, p in paths.items()}}
