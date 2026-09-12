"""Task contract loading and reference implementations.

V1 families: oscillator, kepler 2-body, heat 1d, wave 1d.
TODO: implement reference solvers in M2.
"""

from __future__ import annotations

from invariantlab.schema import TaskContract, load_task_contract

__all__ = ["TaskContract", "load_task_contract"]
