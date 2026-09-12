"""CLI entrypoint for InvariantLab."""

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
    experiment: str = typer.Option(..., "--experiment", help="Experiment config name."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without executing."),
) -> None:
    """Run an evaluation experiment."""
    console.print(f"Running experiment: [bold]{experiment}[/bold]")
    # TODO: implement in M5
    console.print("[yellow]run command not yet implemented — coming in M5.[/yellow]")


@app.command()
def report(
    run_id: str = typer.Option(..., "--run-id", help="Run ID to report on."),
) -> None:
    """Generate a report for a completed run."""
    console.print(f"Generating report for run: [bold]{run_id}[/bold]")
    # TODO: implement in M6
    console.print("[yellow]report command not yet implemented — coming in M6.[/yellow]")


if __name__ == "__main__":
    app()
