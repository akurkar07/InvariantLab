"""CLI entrypoint for InvariantLab."""

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


@app.command()
def run(
    experiment: str = typer.Option(..., "--experiment", help="Path to experiment config."),
    output: str | None = typer.Option(None, "--output", help="Optional run output directory."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate configuration only."),
) -> None:
    """Run an evaluation experiment."""
    from invariantlab.config import load_experiment_config, load_model_config

    config_path = Path(experiment)
    try:
        config = load_experiment_config(config_path)
        load_model_config(Path(config.model))
        if dry_run:
            console.print(f"[green]✓[/green] Experiment [bold]{config.name}[/bold] is valid.")
            return

        from invariantlab.experiments import run_first_model_experiment

        result_dir = run_first_model_experiment(
            config_path,
            Path(output) if output is not None else None,
        )
        console.print(f"[green]✓[/green] Run complete: [bold]{result_dir}[/bold]")
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
