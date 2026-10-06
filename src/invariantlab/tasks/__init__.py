"""Trusted task-contract discovery API.

Executable agent-facing task packages live under the repository-level ``tasks/``
directory. Trusted reference solvers and scientific verification remain under
``invariantlab.verification``; this package only exposes contract loading.
"""

from __future__ import annotations

from invariantlab.schema import TaskContract, load_task_contract

__all__ = ["TaskContract", "load_task_contract"]
