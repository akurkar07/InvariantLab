"""Discover curated mutant directories under tasks/<task>/mutations/<id>/{mutation.yaml, solver.py}.

The registry covers the 10 MutationFamily defect families and is trusted
evaluator-only; tasks/*/src must never import it.
"""

from invariantlab.mutations.registry import (
    MutationRegistryError,
    RegisteredMutant,
    discover_mutants,
)
from invariantlab.mutations.validation import (
    MutantValidation,
    ReferenceValidation,
    validate_mutant,
    validate_reference,
)

__all__ = [
    "MutantValidation",
    "MutationRegistryError",
    "ReferenceValidation",
    "RegisteredMutant",
    "discover_mutants",
    "validate_mutant",
    "validate_reference",
]
