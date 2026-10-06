"""CLI entrypoint for the implemented InvariantLab benchmark utilities."""

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="InvariantLab - physics-grounded evaluation for numerical software.",
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
                console.print(f"[red]FAIL[/red] {error}")
            raise typer.Exit(code=1)
        console.print(f"[green]OK[/green] Task [bold]{contract.id}[/bold] is valid.")
    except typer.Exit:
        raise
    except Exception as error:
        console.print(f"[red]FAIL[/red] Validation failed: {error}")
        raise typer.Exit(code=1) from error


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
            f"[green]OK[/green] Model [bold]{adapter.model_id}[/bold] responded: {preview}"
        )
    except Exception as e:
        console.print(f"[red]FAIL[/red] Model check failed: {e}")
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
    from invariantlab.experiments import audit_repair_experiment

    try:
        audit = audit_repair_experiment(
            Path(experiment),
            Path(run_dir),
            write_canonical=write_canonical,
        )
        expected = int(audit["expected_cells"])
        raw = int(audit["raw_records"])
        canonical = int(audit["canonical_cells"])
        console.print(
            f"Raw records: [bold]{raw}[/bold] | "
            f"Canonical scheduled cells: [bold]{canonical}/{expected}[/bold]"
        )

        if audit["integrity_ok"]:
            console.print("[green]OK[/green] Raw artifact integrity is valid.")
        else:
            console.print("[red]FAIL[/red] Raw artifact integrity is invalid.")
            console.print(
                "Duplicates: "
                f"{audit['duplicate_records']} | "
                "out-of-schedule: "
                f"{len(audit['unexpected_records'])} | "
                "metadata mismatches: "
                f"{len(audit['metadata_mismatches'])} | "
                "malformed: "
                f"{len(audit['malformed_records'])}"
            )

        if write_canonical:
            console.print(
                "Canonical projection: "
                f"[bold]{audit['canonical_events_path']}[/bold]"
            )
        console.print(
            f"Integrity report: [bold]{Path(run_dir) / 'artifact-integrity.json'}[/bold]"
        )
    except Exception as e:
        console.print(f"[red]FAIL[/red] Audit failed: {e}")
        raise typer.Exit(code=1) from e


@app.command()
def report(
    experiment: str = typer.Option(..., "--experiment", help="Path to experiment config."),
    run_dir: str = typer.Option(..., "--run-dir", help="Existing run directory."),
    output: str = typer.Option(..., "--output", help="Directory for rebuilt report files."),
) -> None:
    """Rebuild report tables from a repair run's events."""
    from invariantlab.reporting import build_report

    try:
        result = build_report(Path(experiment), Path(run_dir), Path(output))
        count = result["summary"]["source_records"]
        console.print(
            f"[green]OK[/green] Report rebuilt from {count} canonical records."
        )
        for name, path in result["paths"].items():
            console.print(f"{name}: [bold]{path}[/bold]", soft_wrap=True)
    except Exception as e:
        console.print(f"[red]FAIL[/red] Report failed: {e}")
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
    image: str | None = typer.Option(
        None,
        "--image",
        help="Override the configured container image (recorded in manifest.json).",
    ),
    allow_code_change: bool = typer.Option(
        False,
        "--allow-code-change",
        help="Resume even if the git commit or package version differs from manifest.json.",
    ),
) -> None:
    """Run an evaluation experiment."""
    from dotenv import load_dotenv

    load_dotenv()
    from invariantlab.config import (
        load_experiment_config,
        load_model_config,
        validate_container_image,
    )

    config_path = Path(experiment)
    try:
        config = load_experiment_config(config_path)
        model_config = load_model_config(Path(config.model))
        if image is not None:
            validate_container_image(image)
        if max_new_attempts is not None and max_new_attempts < 1:
            raise ValueError("--max-new-attempts must be at least 1")
        if config.runner not in {"repair", "feedback_replication"}:
            raise ValueError(f"Unsupported experiment runner: {config.runner}")
        from invariantlab.experiments.repair import (
            _resolve_assets,
            _validate_experiment,
        )

        _validate_experiment(config)
        _resolve_assets(config)
        from invariantlab.models import build_adapter

        build_adapter(model_config)
        if config.task_suite is not None:
            console.print(
                "[yellow]Warning:[/yellow] task_suite is ignored until suite "
                "evaluation (#120) lands."
            )
        if dry_run:
            console.print(f"[green]OK[/green] Experiment [bold]{config.name}[/bold] is valid.")
            return

        output_path = Path(output) if output is not None else None
        if config.runner == "repair":
            from invariantlab.experiments import run_repair_experiment

            result_dir = run_repair_experiment(
                config_path,
                output_path,
                max_new_attempts=max_new_attempts,
                image=image,
                allow_code_change=allow_code_change,
            )
        elif config.runner == "feedback_replication":
            from invariantlab.experiments import run_feedback_replication

            result_dir = run_feedback_replication(
                config_path,
                output_path,
                max_new_attempts=max_new_attempts,
                image=image,
                allow_code_change=allow_code_change,
            )
        else:
            raise ValueError(f"Unsupported experiment runner: {config.runner}")

        status_path = result_dir / "run-status.json"
        if not status_path.exists():
            console.print(f"[green]OK[/green] Run complete: [bold]{result_dir}[/bold]")
            return

        status = json.loads(status_path.read_text(encoding="utf-8"))
        state = str(status.get("status", "unknown"))
        completed = int(status.get("completed_cells", 0))
        target = int(status.get("target_cells", 0))
        reason = str(status.get("reason", ""))
        if state == "complete":
            console.print(
                f"[green]OK[/green] Run complete: [bold]{completed}/{target}[/bold] cells"
            )
        else:
            suffix = f" - {reason}" if reason else ""
            console.print(
                f"[yellow]STOPPED[/yellow] Run stopped safely at "
                f"[bold]{completed}/{target}[/bold] cells ({state}){suffix}"
            )
            console.print(
                "Resume by running the same command. "
                f"Evidence: [bold]{result_dir}[/bold]"
            )
    except Exception as e:
        console.print(f"[red]FAIL[/red] Run failed: {e}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
