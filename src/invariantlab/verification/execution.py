"""Layer 0 execution and output validation for task candidates."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from invariantlab.schema import GateResult, load_task_contract

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

GATE_NAMES = ("execution", "output_present", "archive_schema", "finite")


class CandidateExecutor(Protocol):
    def run_entrypoint(
        self,
        candidate_root: Path,
        entrypoint: str,
        input_path: Path,
        output_path: Path,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]: ...

    def run_public_tests(
        self,
        workspace: Path,
        public_tests: str,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True)
class LocalExecutor:
    def run_entrypoint(
        self,
        candidate_root: Path,
        entrypoint: str,
        input_path: Path,
        output_path: Path,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                entrypoint,
                "--input",
                str(input_path.resolve()),
                "--output",
                str(output_path.resolve()),
            ],
            cwd=candidate_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def run_public_tests(
        self,
        workspace: Path,
        public_tests: str,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                public_tests,
                "-q",
                "-p",
                "no:cacheprovider",
                f"--rootdir={workspace}",
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )


_DEFAULT_EXECUTOR = LocalExecutor()
_EXECUTOR: ContextVar[CandidateExecutor] = ContextVar(
    "candidate_executor", default=_DEFAULT_EXECUTOR
)


@contextmanager
def candidate_executor(executor: CandidateExecutor) -> Iterator[None]:
    token = _EXECUTOR.set(executor)
    try:
        yield
    finally:
        _EXECUTOR.reset(token)


def current_executor() -> CandidateExecutor:
    return _EXECUTOR.get()


@dataclass(frozen=True)
class TaskRun:
    gates: list[GateResult]
    arrays: dict[str, np.ndarray] | None
    returncode: int | None
    stderr: str

    @property
    def passed(self) -> bool:
        return len(self.gates) == len(GATE_NAMES) and all(gate.passed for gate in self.gates)


def _failed_run(failed_gate: str, detail: str, returncode: int | None, stderr: str) -> TaskRun:
    failed_index = GATE_NAMES.index(failed_gate)
    gates = [
        GateResult(
            name=name,
            passed=index < failed_index,
            detail=(
                detail
                if index == failed_index
                else f"skipped: {failed_gate} failed"
                if index > failed_index
                else ""
            ),
        )
        for index, name in enumerate(GATE_NAMES)
    ]
    return TaskRun(gates=gates, arrays=None, returncode=returncode, stderr=stderr)


def run_task(
    task_dir: Path,
    candidate_root: Path,
    parameters: dict[str, object],
    work_dir: Path,
    *,
    timeout_seconds: float | None = None,
) -> TaskRun:
    """Execute a candidate and validate its declared output archive."""
    contract = load_task_contract(task_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "task_id": contract.id,
                "parameters": parameters,
                "numerics": {
                    "dtype": contract.numerics.dtype,
                    "seed": contract.numerics.seed,
                },
            }
        ),
        encoding="utf-8",
    )
    output_path = work_dir / contract.output.path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        output_path.unlink()

    timeout = timeout_seconds or contract.budgets.wall_seconds
    try:
        completed = current_executor().run_entrypoint(
            candidate_root,
            contract.entrypoint,
            input_path,
            output_path,
            timeout,
        )
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return _failed_run(
            "execution", f"timed out after {timeout:g} s", returncode=None, stderr=stderr
        )

    returncode = completed.returncode
    stderr = completed.stderr
    if returncode != 0:
        return _failed_run("execution", f"exit code {returncode}", returncode, stderr)

    if not output_path.is_file():
        return _failed_run(
            "output_present", f"{contract.output.path} was not written", returncode, stderr
        )

    arrays: dict[str, np.ndarray] = {}
    expected_names = {spec.name for spec in contract.output.arrays}
    try:
        with np.load(output_path, allow_pickle=False) as archive:
            archive_names = set(archive.files)
            if archive_names != expected_names:
                missing = sorted(expected_names - archive_names)
                unexpected = sorted(archive_names - expected_names)
                return _failed_run(
                    "archive_schema",
                    f"archive keys mismatch (missing: {missing}; unexpected: {unexpected})",
                    returncode,
                    stderr,
                )

            for spec in contract.output.arrays:
                try:
                    array = archive[spec.name]
                except ValueError:
                    return _failed_run(
                        "archive_schema",
                        f"array '{spec.name}' cannot be loaded without pickle "
                        "(object arrays are rejected)",
                        returncode,
                        stderr,
                    )
                if array.ndim != len(spec.shape):
                    return _failed_run(
                        "archive_schema",
                        f"array '{spec.name}' has rank {array.ndim}; expected {len(spec.shape)}",
                        returncode,
                        stderr,
                    )
                for dimension, (actual, expected) in enumerate(
                    zip(array.shape, spec.shape, strict=True)
                ):
                    if expected is not None and actual != expected:
                        return _failed_run(
                            "archive_schema",
                            f"array '{spec.name}' has shape {array.shape}; "
                            f"expected dimension {dimension} to be {expected}",
                            returncode,
                            stderr,
                        )
                if array.dtype != np.dtype(spec.dtype):
                    return _failed_run(
                        "archive_schema",
                        f"array '{spec.name}' has dtype {array.dtype}; expected {spec.dtype}",
                        returncode,
                        stderr,
                    )
                arrays[spec.name] = array
    except (ValueError, OSError, EOFError, zipfile.BadZipFile):
        return _failed_run(
            "archive_schema", "archive could not be loaded as NPZ", returncode, stderr
        )

    non_finite = sorted(name for name, array in arrays.items() if not np.isfinite(array).all())
    if non_finite:
        return _failed_run(
            "finite", f"non-finite values in arrays: {non_finite}", returncode, stderr
        )

    return TaskRun(
        gates=[GateResult(name=name, passed=True) for name in GATE_NAMES],
        arrays=arrays,
        returncode=returncode,
        stderr=stderr,
    )
