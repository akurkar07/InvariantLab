"""Public interface tests for the heat FTCS task package.

These tests are agent-visible and check ordinary interface behaviour, boundaries,
and a stable example. Scientific certification stays in ``tests/scientific``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
import solver
from conftest import make_input, write_input

if TYPE_CHECKING:
    from pathlib import Path


def test_solve_returns_finite_float64_state_with_exact_boundaries() -> None:
    x, state = solver.solve(nx=41, nt=100, alpha=0.1, length=1.2, t_final=0.05)

    assert x.shape == (41,)
    assert state.shape == (41,)
    assert x.dtype == np.dtype(np.float64)
    assert state.dtype == np.dtype(np.float64)
    assert np.isfinite(x).all()
    assert np.isfinite(state).all()
    np.testing.assert_array_equal(state[[0, -1]], np.zeros(2))


def test_ordinary_stable_case_decays_toward_the_manufactured_mode() -> None:
    alpha, length, t_final = 0.1, 1.2, 0.05
    x, state = solver.solve(nx=41, nt=100, alpha=alpha, length=length, t_final=t_final)
    expected = np.sin(np.pi * x / length) * math.exp(-alpha * (np.pi / length) ** 2 * t_final)
    expected[[0, -1]] = 0.0

    assert np.max(np.abs(state)) < 1.0
    np.testing.assert_allclose(state, expected, atol=2e-3, rtol=0.0)


def test_run_writes_exactly_the_declared_arrays(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.json", make_input(41, 100, 0.1, 1.2, 0.05))
    output_path = tmp_path / "result.npz"

    solver.run(input_path, output_path)

    with np.load(output_path) as archive:
        assert sorted(archive.files) == ["state", "x"]
        assert archive["x"].shape == (41,)
        assert archive["state"].shape == (41,)
        assert archive["x"].dtype == np.dtype(np.float64)
        assert archive["state"].dtype == np.dtype(np.float64)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["input.json", "result.npz"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.__setitem__("task_id", "wave_leapfrog"), "task_id"),
        (lambda p: p["numerics"].__setitem__("dtype", "float32"), "numerics.dtype"),
        (lambda p: p["numerics"].__setitem__("seed", 7), "numerics.seed"),
        (lambda p: p["parameters"].pop("alpha"), "missing required"),
        (lambda p: p["parameters"].__setitem__("method", "cn"), "unknown fields"),
        (lambda p: p["parameters"].__setitem__("nx", 2), "nx"),
        (lambda p: p["parameters"].__setitem__("nt", 0), "nt"),
        (lambda p: p["parameters"].__setitem__("alpha", -0.1), "alpha"),
        (lambda p: p["parameters"].__setitem__("length", math.inf), "length"),
        (lambda p: p["parameters"].__setitem__("t_final", 0.0), "t_final"),
        (lambda p: p["parameters"].__setitem__("nx", 11), "FTCS stability"),
    ],
)
def test_parse_input_rejects_invalid_or_unstable_envelopes(mutate, message: str) -> None:
    payload = make_input(41, 100, 0.1, 1.2, 0.05)
    if message == "FTCS stability":
        payload["parameters"].update({"nt": 1, "alpha": 1.0, "length": 1.0, "t_final": 1.0})
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        solver.parse_input(payload)


def test_main_exits_non_zero_and_writes_nothing_on_invalid_input(tmp_path: Path) -> None:
    payload = make_input(41, 100, 0.1, 1.2, 0.05)
    payload["parameters"]["alpha"] = 1.0
    payload["parameters"]["nt"] = 1
    input_path = write_input(tmp_path / "input.json", payload)
    output_path = tmp_path / "result.npz"

    assert solver.main(["--input", str(input_path), "--output", str(output_path)]) == 1
    assert not output_path.exists()
