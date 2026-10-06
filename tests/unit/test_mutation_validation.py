"""Tests for validating reference task suites and declared scientific mutants."""

from __future__ import annotations

import os
import time
from pathlib import Path

from invariantlab.mutations import validate_mutant, validate_reference
from invariantlab.mutations.registry import RegisteredMutant
from invariantlab.schema import (
    ExpectedFailure,
    MutationDefinition,
    MutationFamily,
    load_task_contract,
)
from invariantlab.tasks.boundary import forbidden_imports

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = REPO_ROOT / "tasks" / "oscillator"
STATE_TEST = "tests/scientific/test_reference.py::test_state_matches_analytical_solution"
ENERGY_TEST = "tests/scientific/test_reference.py::test_energy_drift_is_bounded_over_horizon"
STATE_EXPECTATION = ExpectedFailure(test=STATE_TEST, message="state_relative_l2=")
ENERGY_EXPECTATION = ExpectedFailure(test=ENERGY_TEST, message="energy_relative_drift=")


def _mutant(
    tmp_path: Path,
    source: str,
    *,
    mutant_id: str,
    family: MutationFamily,
    expected_failures: list[ExpectedFailure],
) -> RegisteredMutant:
    source_path = tmp_path / "solver.py"
    source_path.write_text(source, encoding="utf-8")
    contract = load_task_contract(TASK_DIR)
    definition = MutationDefinition(
        id=mutant_id,
        task_id="oscillator_verlet",
        family=family,
        expected_effect="The declared solver defect changes the scientific behavior.",
        expected_failures=expected_failures,
    )
    return RegisteredMutant(
        task_dir=TASK_DIR,
        contract=contract,
        definition=definition,
        mutation_dir=tmp_path,
        source_path=source_path,
    )


def _replace(source: str, old: str, new: str) -> str:
    assert old in source
    return source.replace(old, new)


def _python_process_is_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == 259
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.exists():
        state = stat_path.read_text(encoding="utf-8").split(") ", 1)[-1][:1]
        if state == "Z":
            return False
    return True


def test_reference_passes_public_and_scientific_suites() -> None:
    result = validate_reference(TASK_DIR)

    assert result.passed
    assert result.public_passed
    assert result.scientific_passed
    assert result.reasons == []


def test_damping_mutant_matches_both_scientific_failures(tmp_path: Path) -> None:
    source = (TASK_DIR / "src" / "solver.py").read_text(encoding="utf-8")
    source = _replace(
        source,
        "v = v_half - 0.5 * dt * omega2 * x",
        "v = (v_half - 0.5 * dt * omega2 * x) * (1.0 - 1e-6)",
    )
    mutant = _mutant(
        tmp_path,
        source,
        mutant_id="damping",
        family=MutationFamily.PRECISION_DEFECT,
        expected_failures=[STATE_EXPECTATION, ENERGY_EXPECTATION],
    )

    result = validate_mutant(mutant)

    assert result.passed
    assert result.public_passed
    assert result.expected_failures_matched
    assert result.controlled_defect
    assert result.reasons == []


def test_damping_mutant_requires_every_declared_failure(tmp_path: Path) -> None:
    source = (TASK_DIR / "src" / "solver.py").read_text(encoding="utf-8")
    source = _replace(
        source,
        "v = v_half - 0.5 * dt * omega2 * x",
        "v = (v_half - 0.5 * dt * omega2 * x) * (1.0 - 1e-6)",
    )
    mutant = _mutant(
        tmp_path,
        source,
        mutant_id="damping-state-only",
        family=MutationFamily.PRECISION_DEFECT,
        expected_failures=[STATE_EXPECTATION],
    )

    result = validate_mutant(mutant)

    assert not result.passed
    assert result.public_passed
    assert not result.expected_failures_matched
    assert result.controlled_defect
    assert any("test_energy_drift_is_bounded_over_horizon" in reason for reason in result.reasons)


def test_sign_error_mutant_fails_public_example(tmp_path: Path) -> None:
    source = (TASK_DIR / "src" / "solver.py").read_text(encoding="utf-8")
    source = _replace(
        source,
        "v_half = v - 0.5 * dt * omega2 * x",
        "v_half = v + 0.5 * dt * omega2 * x",
    )
    source = _replace(
        source,
        "v = v_half - 0.5 * dt * omega2 * x",
        "v = v_half + 0.5 * dt * omega2 * x",
    )
    mutant = _mutant(
        tmp_path,
        source,
        mutant_id="sign-error",
        family=MutationFamily.SIGN_ERROR,
        expected_failures=[STATE_EXPECTATION, ENERGY_EXPECTATION],
    )

    result = validate_mutant(mutant)

    assert not result.public_passed
    assert any(
        "test_small_example_stays_close_to_exact_solution" in reason for reason in result.reasons
    )


def test_identical_solver_is_not_a_controlled_defect(tmp_path: Path) -> None:
    source = (TASK_DIR / "src" / "solver.py").read_text(encoding="utf-8")
    mutant = _mutant(
        tmp_path,
        source,
        mutant_id="identical",
        family=MutationFamily.PRECISION_DEFECT,
        expected_failures=[STATE_EXPECTATION],
    )

    result = validate_mutant(mutant)

    assert not result.passed
    assert not result.controlled_defect


def test_timeout_kills_pytest_process_tree(tmp_path: Path) -> None:
    source = (TASK_DIR / "src" / "solver.py").read_text(encoding="utf-8")
    source = _replace(source, "for i in range(n_steps):", "for i in iter(int, 1):")
    pid_file = tmp_path / "pids.txt"
    pid_instrumentation = (
        "import numpy as np\n"
        "import os as _os\n"
        f"with open({str(pid_file)!r}, 'a', encoding='utf-8') as _pid_file:\n"
        '    _pid_file.write(f"{_os.getpid()}\\n")\n'
    )
    source = _replace(source, "import numpy as np\n", pid_instrumentation)
    mutant = _mutant(
        tmp_path,
        source,
        mutant_id="infinite-loop",
        family=MutationFamily.TERMINATION_DEFECT,
        expected_failures=[STATE_EXPECTATION],
    )

    result = validate_mutant(mutant, timeout=5)

    assert not result.passed
    assert any("timeout" in reason for reason in result.reasons)
    pids = [int(value) for value in pid_file.read_text(encoding="utf-8").splitlines()]
    assert pids
    deadline = time.monotonic() + 5
    while any(_python_process_is_alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not [pid for pid in pids if _python_process_is_alive(pid)]


def test_forbidden_imports_respects_module_prefix_boundaries() -> None:
    source = "import invariantlab\nimport invariantlab_extra"

    assert forbidden_imports(source, ("invariantlab",)) == [(1, "invariantlab")]
