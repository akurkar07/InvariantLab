"""Public interface tests for the Kepler task package.

These tests are agent-visible and cover ordinary interface behaviour plus one
simple circular orbit; scientific certification remains in ``tests/scientific``.
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
    state = solver.solve(rx=1.5, ry=0.2, vx=-0.1, vy=0.9, mu=2.0, dt=0.02, n_steps=40)

    assert state.shape == (41, 4)
    assert state.dtype == np.dtype(np.float64)
    assert np.isfinite(state).all()
    np.testing.assert_array_equal(state[0], [1.5, 0.2, -0.1, 0.9])


def test_simple_circular_orbit_stays_close_to_closed_form() -> None:
    mu, radius, phase, dt, n_steps = 4.0, 2.0, 0.3, 0.01, 40
    angular_rate = math.sqrt(mu / radius**3)
    speed = math.sqrt(mu / radius)
    initial = (
        radius * math.cos(phase),
        radius * math.sin(phase),
        -speed * math.sin(phase),
        speed * math.cos(phase),
    )
    state = solver.solve(*initial, mu, dt, n_steps)
    theta = phase + angular_rate * dt * n_steps
    expected = [
        radius * math.cos(theta),
        radius * math.sin(theta),
        -speed * math.sin(theta),
        speed * math.cos(theta),
    ]

    assert state[-1] == pytest.approx(expected, abs=2e-4)


def test_run_writes_exactly_the_declared_arrays(tmp_path: Path) -> None:
    input_path = write_input(
        tmp_path / "input.json", make_input(1.5, 0.2, -0.1, 0.9, 2.0, 0.02, 40)
    )
    output_path = tmp_path / "result.npz"

    solver.run(input_path, output_path)

    with np.load(output_path) as archive:
        assert sorted(archive.files) == ["state", "time"]
        assert archive["time"].shape == (41,)
        assert archive["state"].shape == (41, 4)
        assert archive["time"].dtype == np.dtype(np.float64)
        assert archive["state"].dtype == np.dtype(np.float64)
        assert archive["time"][-1] == pytest.approx(0.8)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["input.json", "result.npz"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.__setitem__("task_id", "oscillator_verlet"), "task_id"),
        (lambda p: p["numerics"].__setitem__("dtype", "float32"), "numerics.dtype"),
        (lambda p: p["numerics"].__setitem__("seed", 42), "numerics.seed"),
        (lambda p: p["parameters"].pop("mu"), "missing required"),
        (lambda p: p["parameters"].__setitem__("mass", 1.0), "unknown fields"),
        (lambda p: p["parameters"].__setitem__("mu", -2.0), "mu"),
        (lambda p: p["parameters"].__setitem__("dt", 0.0), "dt"),
        (lambda p: p["parameters"].__setitem__("n_steps", 2.5), "n_steps"),
        (lambda p: p["parameters"].__setitem__("n_steps", 0), "n_steps"),
        (lambda p: p["parameters"].__setitem__("rx", math.inf), "rx"),
        (
            lambda p: (
                p["parameters"].__setitem__("rx", 0.0),
                p["parameters"].__setitem__("ry", 0.0),
            ),
            "radius",
        ),
    ],
)
def test_parse_input_rejects_invalid_envelopes(mutate, message: str) -> None:
    payload = make_input(1.5, 0.2, -0.1, 0.9, 2.0, 0.02, 40)
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        solver.parse_input(payload)


def test_main_exits_non_zero_and_writes_nothing_on_invalid_input(tmp_path: Path) -> None:
    payload = make_input(1.5, 0.2, -0.1, 0.9, 2.0, 0.02, 40)
    payload["task_id"] = "wrong"
    input_path = write_input(tmp_path / "input.json", payload)
    output_path = tmp_path / "result.npz"

    assert solver.main(["--input", str(input_path), "--output", str(output_path)]) == 1
    assert not output_path.exists()
