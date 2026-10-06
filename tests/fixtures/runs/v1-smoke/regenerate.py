"""Regenerate the v1-smoke run fixture through the M5 replay path.

Run from the repository root: ``uv run python tests/fixtures/runs/v1-smoke/regenerate.py``.
Needs Docker only to resolve the pinned image digest; candidates are scored by a
deterministic content-based evaluator so the fixture is reproducible offline.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from invariantlab.experiments import repair
from invariantlab.reporting import build_report

FIXTURE = Path("tests/fixtures/runs/v1-smoke")
EXPERIMENT = FIXTURE / "experiment.yaml"
INPUTS = {"README.md", "experiment.yaml", "model.yaml", "regenerate.py", "replay-source.jsonl"}
# The mutation source is pinned by manifest.json and lives in tasks/; skip the copy so ruff stays clean.
SKIPPED = {repair.CHECKSUMS_FILE, "baseline_solver.py"}
BASELINE_RESULT: dict[str, Any] = {
    "public": {},
    "scientific": {},
    "public_passed": True,
    "scientific_passed": False,
    "metrics": {"max_state_relative_error": 0.02, "max_energy_relative_drift": 0.04},
}
REPAIRED_RESULT: dict[str, Any] = {
    "public": {},
    "scientific": {},
    "public_passed": True,
    "scientific_passed": True,
    "metrics": {"max_state_relative_error": 1e-6, "max_energy_relative_drift": 1e-6},
}


def _content_based_evaluator(source: str, *_args: Any) -> dict[str, Any]:
    if "v = v_half + 0.5 * dt * a_new" in source:
        return REPAIRED_RESULT
    return BASELINE_RESULT


def main() -> None:
    for path in FIXTURE.iterdir():
        if path.name not in INPUTS:
            shutil.rmtree(path) if path.is_dir() else path.unlink()

    repair._evaluate_source = _content_based_evaluator  # type: ignore[assignment]
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        repair.run_repair_experiment(EXPERIMENT, run_dir)
        repair.audit_repair_experiment(EXPERIMENT, run_dir, write_canonical=True)
        for path in sorted(run_dir.iterdir()):
            if path.name not in SKIPPED:
                (FIXTURE / path.name).write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    repair._write_checksums(FIXTURE)
    build_report(EXPERIMENT, FIXTURE, FIXTURE / "expected")


if __name__ == "__main__":
    main()
