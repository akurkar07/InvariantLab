"""Trusted reference code and the Layers 0-6 (seven layers) verification gates.

Layer 0 is ``execution.run_task``; Layer 1 is the task's public tests
(``tasks/*/tests/public``), run by ``verify``; Layers 2-6 are ``oracles``,
``invariants``, ``convergence``, ``metamorphic`` and ``robustness``. The package also
holds analytical solutions, the high-accuracy Kepler oracle and numerical reference
solvers. ``verify.verify_candidate`` composes the seven layers into one
``VerificationResult`` (also exposed as ``invariantlab verify``). See
``docs/verification.md``.
"""
