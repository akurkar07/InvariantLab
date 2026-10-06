"""Docker-free integration tests for the repair runner loop."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from invariantlab.cli import app
from invariantlab.experiments import repair
from invariantlab.experiments.repair import ArtifactIntegrityError
from invariantlab.models.adapter import (
    _REFERENCE_STUB_SOLUTION,
    ModelConnectionError,
    ModelRateLimitError,
    ModelRequestError,
    ModelResponse,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_RESULT = {
    "public": {},
    "scientific": {},
    "public_passed": True,
    "scientific_passed": False,
    "metrics": {
        "max_state_relative_error": 0.02,
        "max_energy_relative_drift": 0.04,
    },
}
REPAIRED_RESULT = {
    "public": {},
    "scientific": {},
    "public_passed": True,
    "scientific_passed": True,
    "metrics": {
        "max_state_relative_error": 1e-6,
        "max_energy_relative_drift": 1e-6,
    },
}
GOOD_RESPONSE = _REFERENCE_STUB_SOLUTION
BAD_RESPONSE = """```python
def solve_oscillator_verlet(x0, v0, omega, dt, n_steps):
    v = v_half + 0.5 * dt * a
```"""
ALTERNATING_RESPONSES = [GOOD_RESPONSE, BAD_RESPONSE] * 2


def _content_based_evaluator(source: str, *_args: object) -> dict[str, object]:
    if "v = v_half + 0.5 * dt * a_new" in source:
        return REPAIRED_RESULT
    return BASELINE_RESULT


@pytest.fixture(autouse=True)
def _no_docker_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(repair, "_resolve_image_digest", lambda image: f"python@sha256:{'a' * 64}")
    monkeypatch.setattr(repair, "_git_state", lambda: ("0" * 40, False))
    monkeypatch.setattr(repair, "_evaluate_source", _content_based_evaluator)


@dataclass
class FakeAdapter:
    outcomes: list[str | Exception]
    model_id: str = "reference_stub/oscillator-verlet"
    calls: int = 0

    def complete(self, prompt: str) -> ModelResponse:
        del prompt
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return ModelResponse(text=outcome)


def _install_adapter(monkeypatch: pytest.MonkeyPatch, adapter: FakeAdapter) -> None:
    monkeypatch.setattr(repair, "build_adapter", lambda _config: adapter)


def _write_repair_config(tmp_path: Path) -> Path:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join(
            [
                "name: integration-repair",
                f"model: {(REPO_ROOT / 'configs/models/reference-stub-oscillator.yaml').as_posix()}",
                "runner: repair",
                f"task: {(REPO_ROOT / 'tasks/oscillator').as_posix()}",
                f"mutation: {(REPO_ROOT / 'tasks/oscillator/mutations/update-order').as_posix()}",
                "conditions: [weak, metrics]",
                "n_attempts: 2",
                "seed: 1729",
                "randomize_order: false",
                "container_image: python:3.12-slim",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config


def _events(output: Path) -> list[dict[str, object]]:
    path = output / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _status(output: Path) -> dict[str, object]:
    return json.loads((output / "run-status.json").read_text(encoding="utf-8"))


def _cell_keys(events: list[dict[str, object]]) -> list[tuple[object, object]]:
    return [(event["condition"], event["trial"]) for event in events]


def _run_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[str | Exception] | None = None,
) -> tuple[Path, Path, FakeAdapter]:
    config = _write_repair_config(tmp_path)
    output = tmp_path / "run"
    adapter = FakeAdapter(outcomes or list(ALTERNATING_RESPONSES))
    _install_adapter(monkeypatch, adapter)
    repair.run_repair_experiment(config, output)
    return config, output, adapter


def test_full_run_writes_complete_schedule_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output, _adapter = _run_complete(tmp_path, monkeypatch)

    status = _status(output)
    assert status["status"] == "complete"
    assert status["completed_cells"] == status["target_cells"] == 4

    events = _events(output)
    expected_cells = [("weak", 1), ("weak", 2), ("metrics", 1), ("metrics", 2)]
    assert _cell_keys(events) == expected_cells
    assert len(set(_cell_keys(events))) == 4

    summary = json.loads((output / "study-summary.json").read_text(encoding="utf-8"))
    assert summary["target_cells"] == summary["completed_cells"] == 4
    assert summary["complete"] is True
    assert (
        sum(summary["by_condition"][condition]["completed"] for condition in ("weak", "metrics"))
        == 4
    )
    for condition in ("weak", "metrics"):
        assert summary["by_condition"][condition]["scientific_passes"] == sum(
            bool(event["successful_repair"]) for event in events if event["condition"] == condition
        )
    assert (output / "checksums.sha256").is_file()
    assert repair.audit_repair_experiment(config, output)["integrity_ok"] is True


def test_batch_resume_skips_completed_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_repair_config(tmp_path)
    output = tmp_path / "run"
    adapter = FakeAdapter(list(ALTERNATING_RESPONSES))
    _install_adapter(monkeypatch, adapter)

    repair.run_repair_experiment(config, output, max_new_attempts=2)

    assert _status(output)["status"] == "batch_complete"
    assert len(_events(output)) == 2
    repair.run_repair_experiment(config, output)

    events = _events(output)
    assert _status(output)["status"] == "complete"
    assert len(events) == 4
    assert len(set(_cell_keys(events))) == 4
    assert adapter.calls == 4


@pytest.mark.parametrize(
    ("error_type", "expected_status"),
    [
        (ModelRateLimitError, "paused_rate_limit"),
        (ModelConnectionError, "paused_connection"),
        (ModelRequestError, "paused_provider_error"),
    ],
)
def test_provider_pause_resumes_without_duplicate_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
    expected_status: str,
) -> None:
    config = _write_repair_config(tmp_path)
    output = tmp_path / "run"
    interrupted = FakeAdapter([GOOD_RESPONSE, error_type("provider unavailable")])
    _install_adapter(monkeypatch, interrupted)

    repair.run_repair_experiment(config, output)

    events = _events(output)
    assert _status(output)["status"] == expected_status
    assert len(events) == 1
    assert ("weak", 2) not in _cell_keys(events)

    resumed = FakeAdapter([BAD_RESPONSE, GOOD_RESPONSE, BAD_RESPONSE])
    _install_adapter(monkeypatch, resumed)
    repair.run_repair_experiment(config, output)

    events = _events(output)
    assert _status(output)["status"] == "complete"
    assert len(events) == 4
    assert len(set(_cell_keys(events))) == 4


def test_duplicate_event_is_rejected_and_audited_canonically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output, _adapter = _run_complete(tmp_path, monkeypatch)
    events_path = output / "events.jsonl"
    first_line = events_path.read_text(encoding="utf-8").splitlines()[0]
    with events_path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(first_line + "\n")

    with pytest.raises(ArtifactIntegrityError):
        repair.run_repair_experiment(config, output)

    assert _status(output)["status"] == "invalid_artifact"
    audit = repair.audit_repair_experiment(config, output, write_canonical=True)
    canonical = (output / "events.canonical.jsonl").read_text(encoding="utf-8").splitlines()
    assert audit["integrity_ok"] is False
    assert audit["duplicate_records"] == 1
    assert len(canonical) == 4
    assert len({_cell_keys([json.loads(line)])[0] for line in canonical}) == 4


def test_event_from_another_experiment_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output, _adapter = _run_complete(tmp_path, monkeypatch)
    events_path = output / "events.jsonl"
    foreign_record = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])
    foreign_record["experiment"] = "some-other-experiment"
    with events_path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(foreign_record) + "\n")

    with pytest.raises(ArtifactIntegrityError):
        repair.run_repair_experiment(config, output)

    assert _status(output)["status"] == "invalid_artifact"


def test_cli_runs_repair_experiment_with_scripted_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_repair_config(tmp_path)
    output = tmp_path / "run"
    adapter = FakeAdapter(list(ALTERNATING_RESPONSES))
    _install_adapter(monkeypatch, adapter)

    result = CliRunner().invoke(
        app,
        ["run", "--experiment", str(config), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert "Run complete" in result.output
    assert "4/4" in result.output
    assert _status(output)["status"] == "complete"
