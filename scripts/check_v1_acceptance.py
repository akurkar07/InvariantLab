"""Fail-closed release-gate checker for the V1 acceptance criteria.

Runs pytest over the given paths, keeps only tests marked with
``@pytest.mark.v1_acceptance("<id>")`` and maps each collected test to one of
the eight V1 acceptance criteria (``V1-AC1`` .. ``V1-AC8``).

Per-test outcomes are ``passed``, ``failed``, ``skipped``, ``xfailed`` or
``not_run``. Per-criterion statuses are:

- ``passed``: every test mapped to the criterion passed.
- ``failed``: at least one mapped test failed or was never run.
- ``skipped``: no failures, but at least one test was skipped or xfailed.
- ``missing``: no collected test is marked for the criterion.

The gate is fail-closed: the script exits 0 only when all eight criteria are
``passed``, no unknown acceptance ids were seen, and pytest itself exited
cleanly. Skipped, xfailed, failed, missing criteria and unknown ids all fail
the gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

V1_ACCEPTANCE_IDS: tuple[str, ...] = tuple(f"V1-AC{i}" for i in range(1, 9))

_MARKER_DOC = (
    "v1_acceptance(id): marks a test as evidence for a V1 acceptance "
    "criterion (V1-AC1..V1-AC8), see docs/v1-acceptance.md"
)


class V1AcceptanceCollector:
    """Pytest plugin that records outcomes for v1_acceptance-marked tests."""

    def __init__(self) -> None:
        self.tests: dict[str, list[str]] = {ac_id: [] for ac_id in V1_ACCEPTANCE_IDS}
        self.outcomes: dict[str, str] = {}
        self.unknown: list[dict[str, str]] = []
        self.invalid: list[dict[str, str]] = []

    def pytest_configure(self, config: pytest.Config) -> None:
        config.addinivalue_line("markers", _MARKER_DOC)

    def pytest_collection_modifyitems(
        self,
        session: pytest.Session,
        config: pytest.Config,
        items: list[pytest.Item],
    ) -> None:
        selected: list[pytest.Item] = []
        deselected: list[pytest.Item] = []
        for item in items:
            markers = list(item.iter_markers("v1_acceptance"))
            if markers:
                selected.append(item)
                self._record_markers(item, markers)
            else:
                deselected.append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = selected

    def _record_markers(self, item: pytest.Item, markers: list[pytest.Mark]) -> None:
        nodeid = item.nodeid
        self.outcomes.setdefault(nodeid, "not_run")
        for marker in markers:
            if len(marker.args) == 1 and isinstance(marker.args[0], str):
                ac_id = marker.args[0]
                if ac_id in self.tests:
                    self.tests[ac_id].append(nodeid)
                else:
                    self.unknown.append({"id": ac_id, "nodeid": nodeid})
            else:
                self.invalid.append({"args": repr(marker.args), "nodeid": nodeid})

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        nodeid = report.nodeid
        if nodeid not in self.outcomes:
            return
        current = self.outcomes[nodeid]
        if report.failed:
            self.outcomes[nodeid] = "failed"
            return
        if current == "failed":
            return
        if hasattr(report, "wasxfail"):
            self.outcomes[nodeid] = "xfailed" if report.skipped else "failed"
            return
        if report.skipped:
            self.outcomes[nodeid] = "skipped"
            return
        if report.when == "call" and report.passed and current == "not_run":
            self.outcomes[nodeid] = "passed"


def _criterion_status(nodeids: list[str], outcomes: dict[str, str]) -> str:
    if not nodeids:
        return "missing"
    results = [outcomes.get(nodeid, "not_run") for nodeid in nodeids]
    if any(r in ("failed", "not_run") for r in results):
        return "failed"
    if any(r in ("skipped", "xfailed") for r in results):
        return "skipped"
    return "passed"


def run(paths: list[str], report_path: Path | None) -> int:
    collector = V1AcceptanceCollector()
    exit_code = int(
        pytest.main(
            [*paths, "-p", "no:cacheprovider", "-q", "-rA"],
            plugins=[collector],
        )
    )

    criteria: dict[str, dict[str, object]] = {}
    for ac_id in V1_ACCEPTANCE_IDS:
        nodeids = collector.tests[ac_id]
        criteria[ac_id] = {
            "status": _criterion_status(nodeids, collector.outcomes),
            "tests": [
                {"nodeid": n, "outcome": collector.outcomes.get(n, "not_run")} for n in nodeids
            ],
        }

    ok = (
        all(c["status"] == "passed" for c in criteria.values())
        and not collector.unknown
        and not collector.invalid
        and exit_code == 0
    )

    if report_path is not None:
        report = {
            "ok": ok,
            "pytest_exit_code": exit_code,
            "criteria": criteria,
            "unknown_ids": collector.unknown,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for ac_id in V1_ACCEPTANCE_IDS:
        status = criteria[ac_id]["status"]
        nodeids = collector.tests[ac_id]
        suffix = "" if status == "missing" else f" ({len(nodeids)} test(s))"
        print(f"{ac_id}: {status}{suffix}")
    for entry in collector.unknown:
        print(f"unknown V1 acceptance id '{entry['id']}' in {entry['nodeid']}")
    for entry in collector.invalid:
        print(f"invalid v1_acceptance marker args {entry['args']} in {entry['nodeid']}")
    print(f"V1 acceptance gate: {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=None, help="Write a JSON report to PATH.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=[str(Path(__file__).resolve().parents[1] / "tests")],
        help="Test paths to collect from (default: the repo tests/ tree).",
    )
    args = parser.parse_args(argv)
    return run(list(args.paths), args.report)


if __name__ == "__main__":
    raise SystemExit(main())
