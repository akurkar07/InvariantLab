from pathlib import Path

import pytest
from typer.testing import CliRunner

from invariantlab.cli import app
from invariantlab.schema import load_mutation_definition, load_task_definition

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = CliRunner()
DEFAULT_MODEL = "adapter: replay\nmodel_id: replay/test\n"
DEFAULT_TASK = REPO_ROOT / "tasks/oscillator"
DEFAULT_MUTATION = REPO_ROOT / "tasks/oscillator/mutations/update-order"


def _write_experiment(
    tmp_path: Path,
    *,
    model_content: str = DEFAULT_MODEL,
    runner: str = "repair",
    task: Path = DEFAULT_TASK,
    mutation: Path = DEFAULT_MUTATION,
    container_image: str = "python:3.12-slim",
    task_suite: str | None = None,
) -> Path:
    model_path = tmp_path / "model.yaml"
    model_path.write_text(model_content, encoding="utf-8")

    lines = [
        "name: dry-run-test",
        f"model: {model_path.as_posix()}",
        f"runner: {runner}",
        f"task: {task.as_posix()}",
        f"mutation: {mutation.as_posix()}",
        "conditions:",
        "  - weak",
        "n_attempts: 1",
        "randomize_order: false",
        f"container_image: {container_image}",
    ]
    if task_suite is not None:
        lines.append(f"task_suite: {task_suite}")

    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def _dry_run(config_path: Path):
    return RUNNER.invoke(
        app,
        ["run", "--experiment", str(config_path), "--dry-run"],
    )


def test_valid_replay_config_passes_dry_run(tmp_path):
    result = _dry_run(_write_experiment(tmp_path))

    assert result.exit_code == 0
    assert "is valid" in result.output


def test_unsupported_model_adapter_fails_dry_run(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        model_content="adapter: bogus\nmodel_id: test\n",
    )

    result = _dry_run(config_path)

    assert result.exit_code == 1
    assert "Unsupported model adapter" in result.output


def test_openai_compatible_model_requires_base_url(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        model_content=(
            "adapter: openai_compatible\n"
            "model_id: test\n"
            "extra: {}\n"
        ),
    )

    result = _dry_run(config_path)

    assert result.exit_code == 1
    assert "extra.base_url" in result.output


def test_nonexistent_task_directory_fails_dry_run(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        task=tmp_path / "missing-task",
    )

    result = _dry_run(config_path)

    assert result.exit_code == 1


def test_nonexistent_mutation_directory_fails_dry_run(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        mutation=tmp_path / "missing-mutation",
    )

    result = _dry_run(config_path)

    assert result.exit_code == 1


def test_missing_mutation_source_fails_dry_run(tmp_path):
    mutation_dir = tmp_path / "missing-source-mutation"
    mutation_dir.mkdir()
    (mutation_dir / "mutation.yaml").write_text(
        "id: missing-source\n"
        "task_id: oscillator_verlet\n"
        "family: sign_error\n"
        "source: missing.py\n",
        encoding="utf-8",
    )
    config_path = _write_experiment(tmp_path, mutation=mutation_dir)

    result = _dry_run(config_path)

    assert result.exit_code == 1
    assert "Mutation source not found" in result.output


def test_placeholder_container_image_fails_dry_run(tmp_path):
    config_path = _write_experiment(
        tmp_path,
        container_image="python:3.12-slim@sha256:placeholder",
    )

    result = _dry_run(config_path)

    assert result.exit_code == 1
    assert "placeholder" in result.output


def test_first_model_runner_is_unsupported(tmp_path):
    config_path = _write_experiment(tmp_path, runner="first_model")

    result = _dry_run(config_path)

    assert result.exit_code == 1
    assert "Unsupported experiment runner" in result.output


def test_task_suite_is_ignored_with_warning(tmp_path):
    config_path = _write_experiment(tmp_path, task_suite="anything")

    result = _dry_run(config_path)

    assert result.exit_code == 0
    assert "task_suite is ignored" in result.output


@pytest.mark.parametrize(
    "config_path",
    sorted((REPO_ROOT / "configs/experiments").glob("*.yaml")),
)
def test_repository_experiment_configs_pass_dry_run(config_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)

    result = _dry_run(config_path)

    assert result.exit_code == 0, result.output


def test_sign_error_mutation_loads_for_oscillator_task():
    mutation_dir = REPO_ROOT / "tasks/oscillator/mutations/sign-error"
    task = load_task_definition(REPO_ROOT / "tasks/oscillator")
    mutation = load_mutation_definition(mutation_dir)
    source = (mutation_dir / mutation.source).read_text(encoding="utf-8")

    assert mutation.family == "sign_error"
    assert mutation.task_id == task.id
    assert "a = omega2 * x" in source
