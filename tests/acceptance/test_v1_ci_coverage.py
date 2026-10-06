"""V1-AC7: CI exercises task validation, a complete smoke run and report reconstruction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.v1_acceptance("V1-AC7")

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github/workflows"

REQUIRED = {
    "task validation": re.compile(r"scripts/validate_task\.py"),
    "replay smoke run": re.compile(
        r"scripts/reproduce_report\.py\s+--smoke(?:\s|$)|invariantlab\s+run\s+\S*replay-smoke"
    ),
    "report reconstruction": re.compile(r"scripts/reproduce_report\.py"),
    "V1 acceptance gate": re.compile(r"scripts/check_v1_acceptance\.py"),
}

STUB_MARKERS = ("not yet implemented", "TODO: implement")


def _load_workflows() -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted((*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")))
    return [(path, yaml.safe_load(path.read_text(encoding="utf-8"))) for path in paths]


def _triggers_on_pr_to_main(on: Any) -> bool:
    if isinstance(on, str):
        return on == "pull_request"
    if isinstance(on, list):
        return "pull_request" in on
    if not isinstance(on, dict) or "pull_request" not in on:
        return False

    pull_request = on["pull_request"]
    if pull_request is None:
        return True
    if not isinstance(pull_request, dict):
        return False

    branches = pull_request.get("branches")
    branches_ignore = pull_request.get("branches-ignore")
    if isinstance(branches_ignore, list) and "main" in branches_ignore:
        return False
    if branches is not None:
        return isinstance(branches, list) and "main" in branches
    if branches_ignore is not None:
        return isinstance(branches_ignore, list) and "main" not in branches_ignore
    return True


def _pr_run_commands() -> list[str]:
    commands: list[str] = []
    for _, workflow in _load_workflows():
        on = workflow.get("on", workflow.get(True))
        if not _triggers_on_pr_to_main(on):
            continue
        jobs = workflow.get("jobs", {})
        if not isinstance(jobs, dict):
            continue
        for job in jobs.values():
            if not isinstance(job, dict):
                continue
            for step in job.get("steps", []):
                if isinstance(step, dict) and isinstance(step.get("run"), str):
                    commands.append(step["run"])
    return commands


def _stub_markers(text: str) -> list[str]:
    lowered = text.lower()
    return [marker for marker in STUB_MARKERS if marker.lower() in lowered]


@pytest.mark.parametrize(
    ("label", "pattern"),
    REQUIRED.items(),
    ids=REQUIRED.keys(),
)
def test_pull_request_ci_runs_required_job(label: str, pattern: re.Pattern[str]) -> None:
    commands = _pr_run_commands()
    assert any(pattern.search(command) for command in commands), (
        f"PR-to-main workflows do not run {label}; commands: {commands}"
    )


def test_ci_scripts_are_not_stubs() -> None:
    commands = _pr_run_commands()
    scripts = {
        script for command in commands for script in re.findall(r"scripts/[\w./-]+\.py", command)
    }
    required_scripts = {
        "scripts/validate_task.py",
        "scripts/validate_mutants.py",
        "scripts/reproduce_report.py",
        "scripts/check_v1_acceptance.py",
    }
    assert required_scripts <= scripts, (
        f"missing CI script references: {required_scripts - scripts}"
    )

    for script in sorted(scripts):
        path = REPO_ROOT / script
        assert path.is_file(), f"CI references missing script: {script}"
        text = path.read_text(encoding="utf-8")
        assert _stub_markers(text) == [], f"CI script {script} contains stub markers"


def test_trigger_parser_rejects_non_pr_or_other_branch() -> None:
    assert not _triggers_on_pr_to_main({"push": {"branches": ["main"]}})
    assert not _triggers_on_pr_to_main({"pull_request": {"branches": ["develop"]}})
    assert not _triggers_on_pr_to_main({"pull_request": {"branches-ignore": ["main"]}})
    assert _triggers_on_pr_to_main("pull_request")
    assert _triggers_on_pr_to_main(["push", "pull_request"])
    assert _triggers_on_pr_to_main({"pull_request": None})
    assert _triggers_on_pr_to_main({"pull_request": {"branches": ["main"]}})


def test_stub_marker_detection() -> None:
    assert _stub_markers('print("Reproduce report: not yet implemented (coming in M6).")')
    assert _stub_markers("# TODO: Implement me")
    assert _stub_markers("The implementation is complete.") == []
