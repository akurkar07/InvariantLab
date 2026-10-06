"""Validate reference tasks and mutants against their public and scientific suites.

References must pass every gate; mutants must pass public tests and fail their declared
scientific tests for the declared reasons while staying within a bounded source diff.
Validation runs on the repository virtual environment, not in Docker.
"""

from __future__ import annotations

import ast
import difflib
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from invariantlab.schema import load_task_contract
from invariantlab.tasks.boundary import forbidden_imports

if TYPE_CHECKING:
    from invariantlab.mutations.registry import RegisteredMutant


@dataclass(frozen=True)
class ReferenceValidation:
    task_dir: Path
    passed: bool
    public_passed: bool
    scientific_passed: bool
    reasons: list[str]


@dataclass(frozen=True)
class MutantValidation:
    task_id: str
    mutant_id: str
    passed: bool
    public_passed: bool
    expected_failures_matched: bool
    controlled_defect: bool
    reasons: list[str]


@dataclass(frozen=True)
class _TestCase:
    node_id: str
    outcome: str
    message: str


@dataclass(frozen=True)
class _SuiteResult:
    cases: tuple[_TestCase, ...]
    returncode: int
    timed_out: bool = False
    issue: str | None = None


def _copy_task(task_dir: Path, dest: Path) -> None:
    """Copy a task while excluding mutants and generated test caches."""
    task_root = task_dir.resolve()

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in {"__pycache__", ".pytest_cache"}}
        if Path(directory).resolve() == task_root and "mutations" in names:
            ignored.add("mutations")
        return ignored

    shutil.copytree(task_dir, dest, ignore=ignore)


def _node_id(copy_dir: Path, classname: str, name: str) -> str:
    parts = classname.split(".") if classname else []
    for length in range(len(parts), 0, -1):
        test_file = copy_dir.joinpath(*parts[:length]).with_suffix(".py")
        if test_file.is_file():
            relative = test_file.relative_to(copy_dir).as_posix()
            suffix = parts[length:] + ([name] if name else [])
            return f"{relative}::{ '::'.join(suffix) }" if suffix else relative
    return "::".join(part for part in (classname, name) if part) or "<unknown test>"


def _run_suite(copy_dir: Path, suite: str, timeout: float) -> _SuiteResult:
    """Run one task pytest suite in an isolated copy and parse its JUnit report."""
    temp_dir = copy_dir.parent
    xml_path = temp_dir / f"{Path(suite).name}-results.xml"
    log_path = temp_dir / f"{Path(suite).name}-pytest.log"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        f"--rootdir={copy_dir}",
        suite,
        f"--junitxml={xml_path}",
    ]
    with log_path.open("wb") as log_file:
        if sys.platform == "win32":
            process = subprocess.Popen(
                command,
                cwd=copy_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            process = subprocess.Popen(
                command,
                cwd=copy_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    capture_output=True,
                    check=False,
                )
            else:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return _SuiteResult((), process.returncode, timed_out=True)

    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError) as error:
        return _SuiteResult(
            (),
            returncode,
            issue=(
                f"pytest exited with return code {returncode}; "
                f"JUnit XML unavailable or invalid: {error}"
            ),
        )

    cases: list[_TestCase] = []
    for testcase in root.iter("testcase"):
        child = next(
            (
                testcase.find(tag)
                for tag in ("failure", "error", "skipped")
                if testcase.find(tag) is not None
            ),
            None,
        )
        if child is None:
            outcome = "passed"
            message = ""
        else:
            outcome = child.tag
            message = child.get("message", "") + "\n" + (child.text or "")
        cases.append(
            _TestCase(
                node_id=_node_id(
                    copy_dir,
                    testcase.get("classname", ""),
                    testcase.get("name", ""),
                ),
                outcome=outcome,
                message=message,
            )
        )
    issue = (
        f"pytest exited with return code {returncode}"
        if returncode not in {0, 1}
        else None
    )
    return _SuiteResult(tuple(cases), returncode, issue=issue)


def _suite_failure_reasons(suite: str, result: _SuiteResult, timeout: float) -> list[str]:
    if result.timed_out:
        return [f"timeout: {suite} suite exceeded {timeout:g} s"]
    if result.issue is not None:
        return [f"{suite}: {result.issue}"]
    return []


def _reference_suite_passed(
    suite: str, result: _SuiteResult, timeout: float, reasons: list[str]
) -> bool:
    suite_reasons = _suite_failure_reasons(suite, result, timeout)
    reasons.extend(suite_reasons)
    if suite_reasons:
        return False
    if not result.cases:
        reasons.append(f"{suite}: suite collected no test cases")
        return False
    passed = True
    for testcase in result.cases:
        if testcase.outcome != "passed":
            reasons.append(
                f"{suite}: {testcase.node_id} {testcase.outcome}: {testcase.message.strip()}"
            )
            passed = False
    return passed


def validate_reference(task_dir: Path, *, timeout: float | None = None) -> ReferenceValidation:
    """Require a task's reference implementation to pass its public and scientific suites."""
    contract = load_task_contract(task_dir)
    suite_timeout = float(contract.budgets.wall_seconds if timeout is None else timeout)
    reasons: list[str] = []
    with tempfile.TemporaryDirectory(
        prefix="invariantlab-validate-", ignore_cleanup_errors=True
    ) as temporary:
        temp_dir = Path(temporary)
        copy_dir = temp_dir / "task"
        _copy_task(task_dir, copy_dir)
        public_result = _run_suite(copy_dir, contract.public_tests, suite_timeout)
        public_passed = _reference_suite_passed(
            "public", public_result, suite_timeout, reasons
        )
        if public_result.timed_out:
            reasons.append("scientific: skipped because the public suite timed out")
            scientific_passed = False
        else:
            scientific_result = _run_suite(copy_dir, contract.scientific_tests, suite_timeout)
            scientific_passed = _reference_suite_passed(
                "scientific", scientific_result, suite_timeout, reasons
            )
    return ReferenceValidation(
        task_dir=task_dir,
        passed=public_passed and scientific_passed,
        public_passed=public_passed,
        scientific_passed=scientific_passed,
        reasons=reasons,
    )


def _controlled_defect(mutant: RegisteredMutant, reasons: list[str]) -> bool:
    """Check that the mutant is a bounded, parseable, evaluator-independent change."""
    try:
        baseline = (mutant.task_dir / mutant.contract.entrypoint).read_text(encoding="utf-8")
        source = mutant.source_path.read_text(encoding="utf-8")
    except OSError as error:
        reasons.append(f"controlled defect: unable to read solver source: {error}")
        return False

    valid = True
    baseline_lines = baseline.splitlines()
    mutant_lines = source.splitlines()
    if baseline_lines == mutant_lines:
        reasons.append("controlled defect: mutant source is identical to the reference")
        valid = False
    diff_lines = list(
        difflib.unified_diff(baseline_lines, mutant_lines, lineterm="")
    )[2:]
    changed_lines = sum(line.startswith(("+", "-")) for line in diff_lines)
    if changed_lines > mutant.definition.max_changed_lines:
        reasons.append(
            "controlled defect: changed "
            f"{changed_lines} lines, exceeding max_changed_lines="
            f"{mutant.definition.max_changed_lines}"
        )
        valid = False
    try:
        ast.parse(source)
    except SyntaxError as error:
        reasons.append(f"controlled defect: mutant source has invalid syntax: {error}")
        valid = False
    if violations := forbidden_imports(source, ("invariantlab",)):
        imports = ", ".join(f"{module} on line {line}" for line, module in violations)
        reasons.append(f"controlled defect: forbidden imports: {imports}")
        valid = False
    return valid


def _matches(node_id: str, declaration: str) -> bool:
    return (
        node_id == declaration
        or node_id.startswith(f"{declaration}[")
        or node_id.startswith(f"{declaration}::")
    )


def _expected_failures_matched(
    mutant: RegisteredMutant, result: _SuiteResult, timeout: float, reasons: list[str]
) -> bool:
    suite_reasons = _suite_failure_reasons("scientific", result, timeout)
    reasons.extend(suite_reasons)
    if suite_reasons:
        return False

    declarations = mutant.definition.expected_failures
    failing = [case for case in result.cases if case.outcome in {"failure", "error"}]
    valid = True
    for testcase in failing:
        if not any(_matches(testcase.node_id, expected.test) for expected in declarations):
            reasons.append(
                f"scientific: unexpected failure {testcase.node_id}: "
                f"{testcase.message.strip()}"
            )
            valid = False
        if testcase.outcome == "error":
            reasons.append(f"scientific: {testcase.node_id} was error, not a declared test failure")
            valid = False

    for expected in declarations:
        matched = [case for case in result.cases if _matches(case.node_id, expected.test)]
        matched_failures = [case for case in matched if case.outcome == "failure"]
        if not matched_failures:
            if any(case.outcome == "error" for case in matched):
                state = "was error"
            elif any(case.outcome == "skipped" for case in matched):
                state = "was skipped"
            else:
                state = "did not fail"
            reasons.append(f"scientific: declared failure {expected.test} {state}")
            valid = False
            continue
        for testcase in matched_failures:
            try:
                message_matches = re.search(expected.message, testcase.message) is not None
            except re.error as error:
                reasons.append(
                    f"scientific: invalid message regex for {expected.test}: {error}"
                )
                valid = False
                continue
            if not message_matches:
                reasons.append(
                    f"scientific: {testcase.node_id} message did not match "
                    f"/{expected.message}/: {testcase.message.strip()}"
                )
                valid = False
    return valid


def validate_mutant(
    mutant: RegisteredMutant, *, timeout: float | None = None
) -> MutantValidation:
    """Require public compatibility, declared scientific failures, and a bounded defect."""
    suite_timeout = float(
        mutant.contract.budgets.wall_seconds if timeout is None else timeout
    )
    reasons: list[str] = []
    controlled_defect = _controlled_defect(mutant, reasons)
    with tempfile.TemporaryDirectory(
        prefix="invariantlab-validate-", ignore_cleanup_errors=True
    ) as temporary:
        temp_dir = Path(temporary)
        copy_dir = temp_dir / "task"
        _copy_task(mutant.task_dir, copy_dir)
        mutant_entrypoint = copy_dir / mutant.contract.entrypoint
        mutant_entrypoint.write_bytes(mutant.source_path.read_bytes())
        public_result = _run_suite(copy_dir, mutant.contract.public_tests, suite_timeout)
        public_suite_reasons = _suite_failure_reasons(
            "public", public_result, suite_timeout
        )
        reasons.extend(public_suite_reasons)
        public_passed = not public_suite_reasons
        if not public_suite_reasons:
            public_passed = True
            for testcase in public_result.cases:
                if testcase.outcome in {"failure", "error"}:
                    reasons.append(
                        f"public: {testcase.node_id} failed: {testcase.message.strip()}"
                    )
                    public_passed = False
            if not any(testcase.outcome == "passed" for testcase in public_result.cases):
                reasons.append("public: suite had no passed test cases")
                public_passed = False

        if public_result.timed_out:
            reasons.append("scientific: skipped because the public suite timed out")
            expected_failures_matched = False
        else:
            scientific_result = _run_suite(
                copy_dir, mutant.contract.scientific_tests, suite_timeout
            )
            expected_failures_matched = _expected_failures_matched(
                mutant, scientific_result, suite_timeout, reasons
            )
    return MutantValidation(
        task_id=mutant.definition.task_id,
        mutant_id=mutant.definition.id,
        passed=public_passed and expected_failures_matched and controlled_defect,
        public_passed=public_passed,
        expected_failures_matched=expected_failures_matched,
        controlled_defect=controlled_defect,
        reasons=reasons,
    )
