"""Task contract loading and reference implementations.

V1 families: oscillator, kepler 2-body, heat 1d, wave 1d.
TODO: implement reference solvers in M2.
"""

from __future__ import annotations

from pathlib import Path

from invariantlab.schema import TaskContract, load_task_contract  # noqa: F401


def load_task_contract(task_dir: str | Path) -> TaskContract:
    """Load a task contract from a directory containing contract.yaml."""
    task_path = Path(task_dir) if isinstance(task_dir, str) else task_dir
    contract_file = task_path / "contract.yaml"
    if not contract_file.exists():
        raise FileNotFoundError(f"No contract.yaml in {task_path}")
    import yaml

    with contract_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return TaskContract(**data)


__all__ = ["load_task_contract", "TaskContract"]
