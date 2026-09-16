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


@app.command("audit-run")
def audit_run(
    experiment: str = typer.Option(..., "--experiment", help="Path to experiment config."),
    run_dir: str = typer.Option(..., "--run-dir", help="Existing run directory."),
    write_canonical: bool = typer.Option(
        False,
        "--write-canonical",
        help="Write a non-destructive events.canonical.jsonl projection.",
    ),
) -> None:
    """Audit raw experiment records against the configured cell schedule."""
    from invariantlab.experiments.feedback_replication import audit_feedback_replication

    try:
        audit = audit_feedback_replication(
            Path(experiment), Path(run_dir), write_canonical=write_canonical
        )
        expected = int(audit["expected_cells"])
        raw = int(audit["raw_records"])
        canonical = int(audit["canonical_cells"])
        console.print(
            f"Raw records: [bold]{raw}[/bold] | "
            f"Canonical scheduled cells: [bold]{canonical}/{expected}[/bold]"
        )
        if audit["integrity_ok"]:
            console.print("[green]✓[/green] Raw artifact integrity is valid.")
        else:
            console.print("[red]✗[/red] Raw artifact integrity is invalid.")
        if write_canonical:
            console.print(f"Canonical projection: [bold]{audit['canonical_events_path']}[/bold]")
        console.print(
            f"Integrity report: [bold]{Path(run_dir) / 'artifact-integrity.json'}[/bold]"
        )
    except Exception as e:
        console.print(f"[red]✗[/red] Audit failed: {e}")
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
    from dotenv import load_dotenv

    load_dotenv()
    from invariantlab.config import (
        load_experiment_config,
        load_model_config,
        load_repair_mutation_config,
        load_repair_task_config,
    )

    config_path = Path(experiment)
    try:
        config = load_experiment_config(config_path)
        load_model_config(Path(config.model))
        if config.runner == "generic_repair":
            if config.task is None or config.mutation_config is None:
                raise ValueError("generic_repair requires task and mutation_config")
            load_repair_task_config(Path(config.task))
            load_repair_mutation_config(Path(config.mutation_config))
        if max_new_attempts is not None and max_new_attempts < 1:
            raise ValueError("--max-new-attempts must be at least 1")
        if dry_run:
            console.print(f"[green]✓[/green] Experiment [bold]{config.name}[/bold] is valid.")
            return

        output_path = Path(output) if output is not None else None
        if config.runner == "generic_repair":
            from invariantlab.experiments import run_generic_repair

            result_dir = run_generic_repair(
                config_path,
                output_path,
                max_new_attempts=max_new_attempts,
            )
        elif config.runner == "feedback_replication":
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
            console.print(
                "Resume by running the same command. "
                f"Evidence: [bold]{result_dir}[/bold]"
            )
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
