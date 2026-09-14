"""Scientific tests for the wave task's independent leapfrog implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import solver
from conftest import make_input, write_input

from invariantlab.verification.analytical import wave_standing_trajectory

if TYPE_CHECKING:
    from pathlib import Path

# For the two fine grids below, the second-order spatial/temporal dispersion
# error is below 2e-7 in relative L2.  This 5e-6 allowance leaves over a 25x
# round-off/model-error margin while rejecting fixed-C, ignored-c, and
# ignored-final-time implementations by orders of magnitude.
STATE_RELATIVE_L2 = 5.0e-6
MODE_AMPLITUDE_ABSOLUTE_ERROR = 5.0e-6

# With nx=5, C=0.9, and one step, the correct half-acceleration startup differs
# from the continuous fundamental mode by about 3e-3 due to the deliberately
# coarse second-order discretisation.  A 5e-3 bound admits that expected error,
# while u_prev=u produces about 3e-1 relative error and fails decisively.
STARTUP_RELATIVE_L2 = 5.0e-3

WAVE_CASES = (
    pytest.param(401, 400, 0.65, 1.3, 0.39, id="slow-wave-nonunit-domain"),
    pytest.param(401, 500, 1.15, 1.7, 0.319, id="fast-wave-nonunit-domain"),
)


def _candidate_state(
    tmp_path: Path, nx: int, nt: int, c: float, length: float, t_final: float
) -> tuple[np.ndarray, np.ndarray]:
    input_path = write_input(tmp_path / "input.json", make_input(nx, nt, c, length, t_final))
    output_path = tmp_path / "result.npz"
    solver.run(input_path, output_path)
    with np.load(output_path) as archive:
        return archive["x"], archive["state"]


@pytest.mark.parametrize(("nx", "nt", "c", "length", "t_final"), WAVE_CASES)
def test_non_special_standing_wave_phase_and_amplitude_match_oracle(
    tmp_path: Path, nx: int, nt: int, c: float, length: float, t_final: float
) -> None:
    x, state = _candidate_state(tmp_path, nx, nt, c, length, t_final)
    expected = wave_standing_trajectory(x, t_final, c, length=length)
    phase = np.pi * c * t_final / length
    mode = np.sin(np.pi * x / length)

    assert abs(np.cos(phase)) > 0.1
    assert abs(abs(np.cos(phase)) - 1.0) > 0.1
    state_relative_l2 = np.linalg.norm(state - expected) / np.linalg.norm(expected)
    amplitude_error = abs(np.dot(state, mode) / np.dot(mode, mode) - np.cos(phase))

    assert state_relative_l2 < STATE_RELATIVE_L2, f"state_relative_l2={state_relative_l2:.3e}"
    assert amplitude_error < MODE_AMPLITUDE_ABSOLUTE_ERROR, (
        f"mode_amplitude_error={amplitude_error:.3e}"
    )


def test_zero_velocity_startup_matches_non_special_standing_wave_oracle() -> None:
    nx, nt, c, length, t_final = 5, 1, 0.75, 1.7, 0.51
    x, state = solver.solve(nx, nt, c, length, t_final)
    expected = wave_standing_trajectory(x, t_final, c, length=length)
    phase = np.pi * c * t_final / length
    startup_relative_l2 = np.linalg.norm(state - expected) / np.linalg.norm(expected)

    assert abs(np.cos(phase)) > 0.1
    assert abs(abs(np.cos(phase)) - 1.0) > 0.1
    assert startup_relative_l2 < STARTUP_RELATIVE_L2, (
        f"startup_relative_l2={startup_relative_l2:.3e}"
    )


def test_different_wave_speeds_produce_distinct_correct_phases(tmp_path: Path) -> None:
    slow = WAVE_CASES[0].values
    fast = WAVE_CASES[1].values
    slow_x, slow_state = _candidate_state(tmp_path / "slow", *slow)
    fast_x, fast_state = _candidate_state(tmp_path / "fast", *fast)
    slow_expected = wave_standing_trajectory(slow_x, slow[4], slow[2], length=slow[3])
    fast_expected = wave_standing_trajectory(fast_x, fast[4], fast[2], length=fast[3])

    slow_phase = np.cos(np.pi * slow[2] * slow[4] / slow[3])
    fast_phase = np.cos(np.pi * fast[2] * fast[4] / fast[3])
    assert not np.isclose(slow_phase, fast_phase)
    np.testing.assert_allclose(
        slow_state, slow_expected, rtol=0.0, atol=MODE_AMPLITUDE_ABSOLUTE_ERROR
    )
    np.testing.assert_allclose(
        fast_state, fast_expected, rtol=0.0, atol=MODE_AMPLITUDE_ABSOLUTE_ERROR
    )
