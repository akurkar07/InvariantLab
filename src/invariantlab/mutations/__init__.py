"""Discover curated mutant directories under tasks/<task>/mutations/<id>/{mutation.yaml, solver.py}.

The registry covers the 10 MutationFamily defect families and is trusted
evaluator-only; tasks/*/src must never import it.
"""

from invariantlab.mutations.registry import (
    MutationRegistryError,
    RegisteredMutant,
    discover_mutants,
)

__all__ = ["MutationRegistryError", "RegisteredMutant", "discover_mutants"]
