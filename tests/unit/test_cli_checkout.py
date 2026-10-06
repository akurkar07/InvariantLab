from pathlib import Path

import pytest
from typer.testing import CliRunner

from invariantlab.cli import app

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = CliRunner()
EXPERIMENT = REPO_ROOT / "configs/experiments/first-model-oscillator.yaml"


@pytest.mark.parametrize(
    ("dirs", "missing"),
    [
        ([], "tasks/"),
        (["tasks"], "configs/"),
    ],
)
def test_run_requires_source_checkout(tmp_path, monkeypatch, dirs: list[str], missing: str) -> None:
    for name in dirs:
        (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path)

    result = RUNNER.invoke(
        app,
        ["run", "--experiment", str(EXPERIMENT), "--dry-run"],
    )

    assert result.exit_code == 1
    assert f"run InvariantLab from a source checkout ({missing} not found)" in result.output
    assert type(result.exception) is SystemExit
