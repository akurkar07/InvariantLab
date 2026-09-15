"""Unit tests for the first executable model experiment."""

from invariantlab.experiments.first_model import BROKEN_SOLVER, _extract_python


def test_broken_solver_contains_expected_entrypoint():
    assert "def solve_oscillator_verlet" in BROKEN_SOLVER


def test_extract_python_fence():
    response = """```python
def solve_oscillator_verlet(x0, v0, omega, dt, n_steps):
    return []
```"""
    source = _extract_python(response)
    assert source.startswith("def solve_oscillator_verlet")
