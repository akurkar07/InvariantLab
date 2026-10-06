"""CLI output must survive non-UTF-8 (e.g. Windows cp1252) consoles."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from invariantlab import cli

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("args", "exit_code", "marker"),
    [
        (["version"], 0, "InvariantLab"),
        (["--help"], 0, "physics-grounded"),
        (
            ["run", "--experiment", "configs/experiments/first-model-oscillator.yaml", "--dry-run"],
            0,
            "OK",
        ),
        (["validate-task", "--task-dir", "tasks/oscillator"], 0, "OK"),
        (["model-check", "--model", "configs/models/does-not-exist.yaml"], 1, "FAIL"),
    ],
)
def test_cli_commands_run_on_cp1252_console(
    monkeypatch: pytest.MonkeyPatch, args: list[str], exit_code: int, marker: str
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(cli, "console", Console(file=stream, width=200, no_color=True))

    result = CliRunner(charset="cp1252").invoke(cli.app, args)

    assert result.exception is None or isinstance(result.exception, SystemExit), repr(
        result.exception
    )
    assert result.exit_code == exit_code
    stream.flush()
    output = raw.getvalue().decode("cp1252") + result.stdout
    assert marker in output
