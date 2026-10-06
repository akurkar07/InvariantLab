"""Unit tests for repair-run provenance: manifest, resume checks, images, checksums."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from invariantlab.cli import app
from invariantlab.config import ExperimentConfig, ModelConfig
from invariantlab.experiments import repair
from invariantlab.models import ModelResponse

REPO_ROOT = Path(__file__).resolve().parents[2]
DIGEST = f"python@sha256:{'a' * 64}"
REAL_RESOLVE_IMAGE_DIGEST = repair._resolve_image_digest


class FakeAdapter:
    model_id = "replay/oscillator-reference"

    def __init__(self, output: Path):
        self.output = output
        self.calls = 0

    def generate(self, prompt: str) -> str:
        return self.complete(prompt).text

    def complete(self, prompt: str) -> ModelResponse:
        del prompt
        assert (self.output / "manifest.json").exists()
        self.calls += 1
        return ModelResponse(text="```python\nprint('fixed')\n```")


def _evaluation(*args):
    del args
    return {
        "public_passed": True,
        "scientific_passed": False,
        "metrics": {"max_state_relative_error": 0.02, "max_energy_relative_drift": 0.04},
    }


def _write_config(tmp_path: Path, image: str = "python:3.12-slim") -> Path:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join(
            [
                "name: manifest-test",
                f"task_suite: {(REPO_ROOT / 'configs/task-suites/v1-smoke.yaml').as_posix()}",
                f"model: {(REPO_ROOT / 'configs/models/replay-first-model.yaml').as_posix()}",
                "runner: repair",
                f"task: {(REPO_ROOT / 'tasks/oscillator').as_posix()}",
                "mutation: "
                f"{(REPO_ROOT / 'tasks/oscillator/mutations/update-order').as_posix()}",
                "conditions: [weak, metrics]",
                "n_attempts: 1",
                "seed: 1729",
                f"container_image: {image}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config


@pytest.fixture
def run_env(tmp_path, monkeypatch):
    output = tmp_path / "run"
    state = {"digest": DIGEST, "git": ("1" * 40, False)}
    adapter = FakeAdapter(output)
    monkeypatch.setattr(repair, "build_adapter", lambda config: adapter)
    monkeypatch.setattr(repair, "_evaluate_source", _evaluation)
    monkeypatch.setattr(repair, "_resolve_image_digest", lambda image: state["digest"])
    monkeypatch.setattr(repair, "_git_state", lambda: state["git"])
    return output, state, adapter


def _manifest(output: Path) -> dict:
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def _status(output: Path) -> dict:
    return json.loads((output / "run-status.json").read_text(encoding="utf-8"))


def test_manifest_records_provenance_before_first_model_call(tmp_path, run_env):
    output, _state, adapter = run_env
    config = _write_config(tmp_path)

    repair.run_repair_experiment(config, output)

    assert adapter.calls == 2
    manifest = _manifest(output)
    assert manifest["experiment"] == "manifest-test"
    assert len(manifest["run_id"]) == 32
    assert manifest["config_path"] == config.as_posix()
    assert manifest["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert manifest["git_commit"] == "1" * 40
    assert manifest["git_dirty"] is False
    assert manifest["python_version"]
    assert manifest["platform"]
    assert manifest["task_id"] == "oscillator_verlet"
    assert manifest["mutation_id"] == "update-order"
    verifier = REPO_ROOT / "tasks/oscillator/verifier.py"
    assert manifest["artifact_sha256"]["verifier"] == hashlib.sha256(
        verifier.read_bytes()
    ).hexdigest()
    assert {
        "task_yaml",
        "contract_yaml",
        "verifier",
        "prompt_template",
        "mutation_source",
    } <= set(manifest["artifact_sha256"])
    assert manifest["model"] == {
        "adapter": "replay",
        "model_id": "replay/oscillator-reference",
        "temperature": 0.0,
        "max_tokens": 4096,
        "extra": {},
    }
    assert len(manifest["model_sha256"]) == 64
    assert manifest["seed"] == 1729
    assert manifest["conditions"] == ["weak", "metrics"]
    assert manifest["n_attempts"] == 1
    assert manifest["configured_image"] == "python:3.12-slim"
    assert manifest["image_override"] is None
    assert manifest["container_image"] == "python:3.12-slim"
    assert manifest["image_digest"] == DIGEST


def test_model_manifest_never_records_secret_values():
    config = ModelConfig(
        adapter="openai_compatible",
        model_id="m",
        extra={"api_key_env": "MY_KEY", "api_key": "sk-secret", "base_url": "http://x"},
    )

    extra = repair._model_manifest(config)["extra"]

    assert extra == {
        "api_key_env": "MY_KEY",
        "api_key": "<redacted>",
        "base_url": "http://x",
    }


def test_matching_manifest_is_left_untouched_on_resume(tmp_path, run_env):
    output, _state, adapter = run_env
    config = _write_config(tmp_path)

    repair.run_repair_experiment(config, output, max_new_attempts=1)
    assert _status(output)["status"] == "batch_complete"
    before = (output / "manifest.json").read_bytes()

    repair.run_repair_experiment(config, output)

    assert _status(output)["status"] == "complete"
    assert adapter.calls == 2
    assert (output / "manifest.json").read_bytes() == before


@pytest.mark.parametrize(
    ("change", "field"),
    [
        (lambda state: state.update(digest=f"python@sha256:{'b' * 64}"), "image_digest"),
        (lambda state: state.update(git=("2" * 40, False)), "git_commit"),
    ],
)
def test_manifest_mismatch_on_resume_stops_run(tmp_path, run_env, change, field):
    output, state, adapter = run_env
    config = _write_config(tmp_path)
    repair.run_repair_experiment(config, output, max_new_attempts=1)
    change(state)

    with pytest.raises(repair.ArtifactIntegrityError, match=field):
        repair.run_repair_experiment(config, output)

    status = _status(output)
    assert status["status"] == "invalid_artifact"
    assert field in status["reason"]
    assert adapter.calls == 1


def test_model_config_change_on_resume_stops_run(tmp_path, run_env):
    output, _state, adapter = run_env
    config = _write_config(tmp_path)
    repair.run_repair_experiment(config, output, max_new_attempts=1)
    manifest = _manifest(output)
    manifest["model"]["temperature"] = 0.7
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(repair.ArtifactIntegrityError, match="model"):
        repair.run_repair_experiment(config, output)
    assert adapter.calls == 1


def test_allow_code_change_permits_new_git_commit(tmp_path, run_env):
    output, state, adapter = run_env
    config = _write_config(tmp_path)
    repair.run_repair_experiment(config, output, max_new_attempts=1)
    before = (output / "manifest.json").read_bytes()
    state["git"] = ("2" * 40, True)

    repair.run_repair_experiment(config, output, allow_code_change=True)

    assert _status(output)["status"] == "complete"
    assert adapter.calls == 2
    assert (output / "manifest.json").read_bytes() == before


def test_image_override_is_used_and_recorded(tmp_path, run_env):
    output, _state, _adapter = run_env

    repair.run_repair_experiment(
        _write_config(tmp_path),
        output,
        image="mirror.gcr.io/library/python:3.12-slim",
    )

    manifest = _manifest(output)
    assert manifest["configured_image"] == "python:3.12-slim"
    assert manifest["image_override"] == "mirror.gcr.io/library/python:3.12-slim"
    assert manifest["container_image"] == "mirror.gcr.io/library/python:3.12-slim"


def test_placeholder_images_are_rejected(tmp_path, run_env):
    output, _state, adapter = run_env
    with pytest.raises(ValidationError, match="placeholder"):
        ExperimentConfig(
            name="x",
            task_suite="s",
            model="m",
            container_image="python:3.12-slim@sha256:placeholder",
        )
    with pytest.raises(ValueError, match="placeholder"):
        repair.run_repair_experiment(
            _write_config(tmp_path), output, image="python@sha256:placeholder"
        )
    assert adapter.calls == 0


@pytest.mark.parametrize("extra_args", [["--dry-run"], []])
def test_cli_run_rejects_placeholder_images(tmp_path, extra_args):
    placeholder = _write_config(tmp_path, image="python:3.12-slim@sha256:placeholder")
    result = CliRunner().invoke(
        app, ["run", "--experiment", str(placeholder), *extra_args]
    )
    assert result.exit_code == 1
    assert "placeholder" in result.output

    valid = _write_config(tmp_path)
    result = CliRunner().invoke(
        app,
        ["run", "--experiment", str(valid), "--image", "x@sha256:placeholder", *extra_args],
    )
    assert result.exit_code == 1
    assert "placeholder" in result.output


def test_checksums_cover_every_run_file_on_complete(tmp_path, run_env):
    output, _state, _adapter = run_env

    repair.run_repair_experiment(_write_config(tmp_path), output)

    lines = (output / "checksums.sha256").read_bytes().decode().splitlines()
    listed = {}
    for line in lines:
        digest, name = line.split("  ", 1)
        listed[name] = digest
    files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert set(listed) == files - {"checksums.sha256"}
    assert {"manifest.json", "events.jsonl", "run-status.json"} <= set(listed)
    for name, digest in listed.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_no_checksums_until_complete(tmp_path, run_env):
    output, _state, _adapter = run_env
    repair.run_repair_experiment(_write_config(tmp_path), output, max_new_attempts=1)
    assert not (output / "checksums.sha256").exists()


def test_digest_resolver_pulls_missing_image(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        del kwargs
        calls.append(args[:3])
        if args[:3] == ["docker", "image", "inspect"]:
            if len(calls) == 1:
                return subprocess.CompletedProcess(args, 1, "", "No such image")
            return subprocess.CompletedProcess(args, 0, DIGEST + "\n", "")
        return subprocess.CompletedProcess(args, 0, "pulled", "")

    monkeypatch.setattr(repair.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(repair.subprocess, "run", fake_run)

    assert REAL_RESOLVE_IMAGE_DIGEST("python:3.12-slim") == DIGEST
    assert calls == [
        ["docker", "image", "inspect"],
        ["docker", "pull", "python:3.12-slim"],
        ["docker", "image", "inspect"],
    ]


def test_digest_resolver_pull_failure_is_infrastructure_error(monkeypatch):
    def fake_run(args, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(args, 1, "", "toomanyrequests")

    monkeypatch.setattr(repair.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(repair.subprocess, "run", fake_run)

    with pytest.raises(repair.InfrastructureError, match="toomanyrequests"):
        REAL_RESOLVE_IMAGE_DIGEST("python:3.12-slim")
