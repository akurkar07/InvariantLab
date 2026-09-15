"""CLI entrypoint for InvariantLab."""

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="InvariantLab — Physics-grounded evaluation for AI-generated scientific software.",
)
console = Console()


@app.command()
def version() -> None:
    """Print the current InvariantLab version."""
    try:
        from importlib.metadata import version as meta_version

        console.print(f"InvariantLab [bold cyan]{meta_version('invariantlab')}[/bold cyan]")
    except Exception:
        console.print("InvariantLab [bold cyan]development[/bold cyan]")


@app.command()
def validate_task(
    task_dir: str = typer.Option(..., "--task-dir", help="Path to task directory."),
) -> None:
    """Validate a task contract against the schema."""
    from invariantlab.tasks import load_task_contract

    try:
        contract = load_task_contract(task_dir)
        console.print(f"[green]✓[/green] Task [bold]{contract.id}[/bold] is valid.")
    except Exception as e:
        console.print(f"[red]✗[/red] Validation failed: {e}")
        raise typer.Exit(code=1) from e


@app.command("model-check")
def model_check(
    model: str = typer.Option(..., "--model", help="Path to model config."),
) -> None:
    """Send one small request to verify a configured model endpoint."""
    from invariantlab.config import load_model_config
    from invariantlab.models import build_adapter

    try:
        config = load_model_config(Path(model))
        adapter = build_adapter(config)
        response = adapter.generate(
            "Connectivity check. Return a tiny Python file containing "
            "def ping(): return 'ok'."
        )
        preview = response.replace("\n", " ")[:160]
        console.print(
            f"[green]✓[/green] Model [bold]{adapter.model_id}[/bold] responded: {preview}"
        )
    except Exception as e:
        console.print(f"[red]✗[/red] Model check failed: {e}")
        raise typer.Exit(code=1) from e


@app.command()
def run(
    experiment: str = typer.Option(..., "--experiment", help="Path to experiment config."),
    output: str | None = typer.Option(None, "--output", help="Optional run output directory."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate configuration only."),
    max_new_attempts: int | None = typer.Option(
        None,
        "--max-new-attempts",
        help="Stop cleanly after this many new cells; rerun to resume.",
    ),
) -> None:
    """Run an evaluation experiment."""
    from invariantlab.config import load_experiment_config, load_model_config

    config_path = Path(experiment)
    try:
        config = load_experiment_config(config_path)
        load_model_config(Path(config.model))
        if max_new_attempts is not None and max_new_attempts < 1:
            raise ValueError("--max-new-attempts must be at least 1")
        if dry_run:
            console.print(f"[green]✓[/green] Experiment [bold]{config.name}[/bold] is valid.")
            return

        output_path = Path(output) if output is not None else None
        if config.runner == "feedback_replication":
            from invariantlab.experiments import run_feedback_replication

            result_dir = run_feedback_replication(
                config_path,
                output_path,
                max_new_attempts=max_new_attempts,
            )
        elif config.runner == "first_model":
            if max_new_attempts is not None:
                raise ValueError(
                    "--max-new-attempts is only supported by resumable experiment runners"
                )
            from invariantlab.experiments import run_first_model_experiment

            result_dir = run_first_model_experiment(config_path, output_path)
        else:
            raise ValueError(f"Unsupported experiment runner: {config.runner}")

        status_path = result_dir / "run-status.json"
        if not status_path.exists():
            console.print(f"[green]✓[/green] Run complete: [bold]{result_dir}[/bold]")
            return

        status = json.loads(status_path.read_text(encoding="utf-8"))
        state = str(status.get("status", "unknown"))
        completed = int(status.get("completed_cells", 0))
        target = int(status.get("target_cells", 0))
        reason = str(status.get("reason", ""))
        if state == "complete":
            console.print(
                f"[green]✓[/green] Run complete: [bold]{completed}/{target}[/bold] cells"
            )
        else:
            suffix = f" — {reason}" if reason else ""
            console.print(
                f"[yellow]•[/yellow] Run stopped safely at "
                f"[bold]{completed}/{target}[/bold] cells ({state}){suffix}"
            )
            console.print(f"Resume by running the same command. Evidence: [bold]{result_dir}[/bold]")
    except Exception as e:
        console.print(f"[red]✗[/red] Run failed: {e}")
        raise typer.Exit(code=1) from e


@app.command()
def report(
    run_id: str = typer.Option(..., "--run-id", help="Run ID to report on."),
) -> None:
    """Generate a report for a completed run."""
    console.print(f"Generating report for run: [bold]{run_id}[/bold]")
    console.print("[yellow]report command not yet implemented — coming in M6.[/yellow]")


if __name__ == "__main__":
    app()
