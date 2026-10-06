"""Public interface tests for the wave-equation task package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import solver
from conftest import make_input, write_input

if TYPE_CHECKING:
    from pathlib import Path


def test_solve_returns_stable_finite_float64_state_with_dirichlet_boundaries() -> None:
    x, state = solver.solve(nx=81, nt=160, c=0.7, length=1.3, t_final=0.25)

    assert x.shape == (81,)
    assert state.shape == (81,)
    assert x.dtype == np.dtype(np.float64)
    assert state.dtype == np.dtype(np.float64)
    assert np.isfinite(x).all()
    assert np.isfinite(state).all()
    np.testing.assert_array_equal(state[[0, -1]], np.zeros(2))


def test_run_writes_exactly_the_declared_arrays(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.json", make_input(81, 160, 0.7, 1.3, 0.25))
    output_path = tmp_path / "result.npz"

    solver.run(input_path, output_path)

    with np.load(output_path) as archive:
        assert sorted(archive.files) == ["state", "x"]
        assert archive["x"].shape == (81,)
        assert archive["state"].shape == (81,)
        assert archive["x"].dtype == np.dtype(np.float64)
        assert archive["state"].dtype == np.dtype(np.float64)
        assert np.isfinite(archive["x"]).all()
        assert np.isfinite(archive["state"]).all()
        np.testing.assert_array_equal(archive["state"][[0, -1]], np.zeros(2))
    assert sorted(path.name for path in tmp_path.iterdir()) == ["input.json", "result.npz"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.__setitem__("task_id", "oscillator_verlet"), "task_id"),
        (lambda payload: payload["numerics"].__setitem__("dtype", "float32"), "numerics.dtype"),
        (lambda payload: payload["numerics"].__setitem__("seed", 7), "numerics.seed"),
        (lambda payload: payload["parameters"].pop("c"), "missing required"),
        (lambda payload: payload["parameters"].__setitem__("extra", 1.0), "unknown fields"),
        (lambda payload: payload["parameters"].__setitem__("nx", 2), "nx"),
        (lambda payload: payload["parameters"].__setitem__("nt", 0), "nt"),
        (lambda payload: payload["parameters"].__setitem__("c", 0.0), "c"),
        (lambda payload: payload["parameters"].__setitem__("length", float("inf")), "length"),
        (lambda payload: payload["parameters"].__setitem__("t_final", -0.1), "t_final"),
        (lambda payload: payload["parameters"].__setitem__("c", "0.7"), "c"),
    ],
)
def test_parse_input_rejects_invalid_envelopes(mutate, message: str) -> None:
    payload = make_input(81, 160, 0.7, 1.3, 0.25)
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        solver.parse_input(payload)


def test_parse_input_rejects_unstable_or_inconsistent_derived_courant_number() -> None:
    unstable = make_input(11, 1, 2.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="Courant"):
        solver.parse_input(unstable)

    inconsistent = make_input(81, 160, 0.7, 1.3, 0.25, courant=0.5)
    with pytest.raises(ValueError, match="derived Courant"):
        solver.parse_input(inconsistent)


def test_main_exits_non_zero_and_writes_nothing_on_invalid_input(tmp_path: Path) -> None:
    payload = make_input(11, 1, 2.0, 1.0, 1.0)
    input_path = write_input(tmp_path / "input.json", payload)
    output_path = tmp_path / "result.npz"

    assert solver.main(["--input", str(input_path), "--output", str(output_path)]) == 1
    assert not output_path.exists()
