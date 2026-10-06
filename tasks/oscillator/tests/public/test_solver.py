"""Public interface tests for the oscillator task package.

These tests are visible to the agent. They check the invocation protocol and one
small ordinary example; scientific certification lives in ``tests/scientific``.
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


def test_solve_returns_state_rows_for_every_time_level() -> None:
    state = solver.solve(x0=0.5, v0=0.2, omega=2.0, dt=0.05, n_steps=40)

    assert state.shape == (41, 2)
    assert state.dtype == np.dtype(np.float64)
    assert np.isfinite(state).all()
    np.testing.assert_array_equal(state[0], [0.5, 0.2])


def test_small_example_stays_close_to_exact_solution() -> None:
    x0, v0, omega, dt, n_steps = 0.5, 0.2, 2.0, 0.01, 50
    state = solver.solve(x0, v0, omega, dt, n_steps)
    t = dt * n_steps
    x_exact = x0 * math.cos(omega * t) + v0 / omega * math.sin(omega * t)
    v_exact = -x0 * omega * math.sin(omega * t) + v0 * math.cos(omega * t)

    assert state[-1] == pytest.approx([x_exact, v_exact], abs=1e-4)


def test_run_writes_exactly_the_declared_arrays(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.json", make_input(0.5, 0.2, 2.0, 0.05, 40))
    output_path = tmp_path / "result.npz"

    solver.run(input_path, output_path)

    with np.load(output_path) as archive:
        assert sorted(archive.files) == ["state", "time"]
        assert archive["time"].shape == (41,)
        assert archive["state"].shape == (41, 2)
        assert archive["time"].dtype == np.dtype(np.float64)
        assert archive["state"].dtype == np.dtype(np.float64)
        assert archive["time"][-1] == pytest.approx(2.0)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["input.json", "result.npz"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.__setitem__("task_id", "kepler_verlet"), "task_id"),
        (lambda p: p["numerics"].__setitem__("dtype", "float32"), "numerics.dtype"),
        (lambda p: p["numerics"].__setitem__("seed", 7), "numerics.seed"),
        (lambda p: p["parameters"].pop("omega"), "missing required"),
        (lambda p: p["parameters"].__setitem__("mass", 1.0), "unknown fields"),
        (lambda p: p["parameters"].__setitem__("omega", -2.0), "omega"),
        (lambda p: p["parameters"].__setitem__("dt", 0.0), "dt"),
        (lambda p: p["parameters"].__setitem__("n_steps", 2.5), "n_steps"),
        (lambda p: p["parameters"].__setitem__("n_steps", 0), "n_steps"),
        (lambda p: p["parameters"].__setitem__("x0", "1.0"), "x0"),
    ],
)
def test_parse_input_rejects_invalid_envelopes(mutate, message: str) -> None:
    payload = make_input(0.5, 0.2, 2.0, 0.05, 40)
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        solver.parse_input(payload)


def test_main_exits_non_zero_and_writes_nothing_on_invalid_input(tmp_path: Path) -> None:
    payload = make_input(0.5, 0.2, 2.0, 0.05, 40)
    payload["task_id"] = "wrong"
    input_path = write_input(tmp_path / "input.json", payload)
    output_path = tmp_path / "result.npz"

    assert solver.main(["--input", str(input_path), "--output", str(output_path)]) == 1
    assert not output_path.exists()
