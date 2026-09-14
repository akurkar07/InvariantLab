"""CLI entrypoint for the implemented InvariantLab benchmark utilities."""

import typer
from rich.console import Console

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="InvariantLab — physics-grounded evaluation for numerical software.",
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
    task_dir: str = typer.Option(..., "--task-dir", help="Path to one task directory."),
) -> None:
    """Validate one task contract and its declared package artifacts."""
    from invariantlab.schema import load_task_contract
    from invariantlab.tasks.validation import validate_task_artifacts

    try:
        contract = load_task_contract(task_dir)
        errors = validate_task_artifacts(task_dir, contract)
        if errors:
            for error in errors:
                console.print(f"[red]✗[/red] {error}")
            raise typer.Exit(code=1)
        console.print(f"[green]✓[/green] Task [bold]{contract.id}[/bold] is valid.")
    except typer.Exit:
        raise
    except Exception as error:
        console.print(f"[red]✗[/red] Validation failed: {error}")
        raise typer.Exit(code=1) from error


if __name__ == "__main__":
    app()
