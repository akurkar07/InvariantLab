"""Architecture guards for agent-facing task packages."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TASK_SOURCE_ROOT = REPOSITORY_ROOT / "tasks"
EXPECTED_TASK_NAMES = ("oscillator", "kepler", "heat1d", "wave1d")
TRUSTED_PREFIXES = (
    "invariantlab.verification",
    "invariantlab.mutations",
    "tests.scientific",
)


def _is_trusted_module(module: str) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in TRUSTED_PREFIXES)


def _forbidden_imports(source: str) -> list[tuple[int, str]]:
    """Return trusted modules imported by agent-facing source."""
    violations: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules = (
                [node.module]
                if _is_trusted_module(node.module)
                else [f"{node.module}.{alias.name}" for alias in node.names]
            )
        else:
            continue
        violations.extend((node.lineno, module) for module in modules if _is_trusted_module(module))
    return violations


@pytest.mark.parametrize(
    ("source", "module"),
    [
        ("from invariantlab.verification import analytical", "invariantlab.verification"),
        ("import invariantlab.verification.solvers", "invariantlab.verification.solvers"),
        ("from invariantlab.mutations.registry import REGISTRY", "invariantlab.mutations.registry"),
        ("from tests.scientific.helpers import oracle", "tests.scientific.helpers"),
        ("from invariantlab import verification", "invariantlab.verification"),
        ("from invariantlab import mutations", "invariantlab.mutations"),
        ("from tests import scientific", "tests.scientific"),
    ],
)
def test_boundary_checker_detects_trusted_imports(source: str, module: str) -> None:
    assert _forbidden_imports(source) == [(1, module)]


@pytest.mark.parametrize(
    "source",
    [
        "import invariantlab.verification_helpers",
        "from invariantlab import verification_helpers",
        "import invariantlab.mutations_extra.registry",
        "from tests import scientific_helpers",
    ],
)
def test_boundary_checker_allows_benign_near_prefixes(source: str) -> None:
    assert _forbidden_imports(source) == []


@pytest.fixture
def task_source_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    task_source_root = tmp_path / "tasks"
    for task_name in EXPECTED_TASK_NAMES:
        (task_source_root / task_name).mkdir(parents=True)
    monkeypatch.setitem(globals(), "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setitem(globals(), "TASK_SOURCE_ROOT", task_source_root)
    return task_source_root


@pytest.mark.parametrize("missing_task", EXPECTED_TASK_NAMES)
def test_repository_boundary_rejects_missing_expected_task_root(
    task_source_layout: Path, missing_task: str
) -> None:
    (task_source_layout / missing_task).rmdir()

    with pytest.raises(AssertionError, match=missing_task):
        test_agent_facing_task_sources_do_not_import_trusted_evaluator()


@pytest.mark.parametrize("task_name", EXPECTED_TASK_NAMES)
def test_repository_boundary_scans_each_expected_task_source_root(
    task_source_layout: Path, task_name: str
) -> None:
    source_file = task_source_layout / task_name / "src" / "package" / "solver.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("import invariantlab.verification\n", encoding="utf-8")

    with pytest.raises(AssertionError, match=r"forbidden import invariantlab\.verification"):
        test_agent_facing_task_sources_do_not_import_trusted_evaluator()


def test_agent_facing_task_sources_do_not_import_trusted_evaluator() -> None:
    task_roots = [TASK_SOURCE_ROOT / task_name for task_name in EXPECTED_TASK_NAMES]
    missing_task_roots = [
        task_root.relative_to(REPOSITORY_ROOT) for task_root in task_roots if not task_root.is_dir()
    ]
    assert not missing_task_roots, f"missing task roots: {missing_task_roots}"

    violations: list[str] = []
    for task_root in task_roots:
        for source_file in sorted((task_root / "src").rglob("*.py")):
            for line, module in _forbidden_imports(source_file.read_text(encoding="utf-8")):
                relative_path = source_file.relative_to(REPOSITORY_ROOT)
                violations.append(f"{relative_path}:{line}: forbidden import {module}")
    assert not violations, "\n".join(violations)


def test_task_namespace_has_no_stale_m2_placeholders() -> None:
    stale_files = [
        path.relative_to(REPOSITORY_ROOT)
        for path in sorted((REPOSITORY_ROOT / "src/invariantlab/tasks").rglob("*.py"))
        if "TODO: implement" in path.read_text(encoding="utf-8")
    ]
    assert not stale_files, f"stale task placeholders: {stale_files}"


SCIENTIFIC_TEST_RELATIVE_PATH = Path("tests/scientific")


def _scientific_candidate_solver_imports(task_name: str, source: str) -> list[tuple[int, str]]:
    """Return imports from a hidden test into a candidate task's solver module."""
    violations: list[tuple[int, str]] = []
    task_module_prefix = f"tasks.{task_name}.src"
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "solver" or alias.name == f"{task_module_prefix}.solver":
                    violations.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "solver" or node.module == f"{task_module_prefix}.solver":
                violations.append((node.lineno, node.module))
            elif node.module == task_module_prefix:
                violations.extend(
                    (node.lineno, f"{node.module}.{alias.name}")
                    for alias in node.names
                    if alias.name == "solver"
                )
    return violations


@pytest.mark.parametrize(
    ("task_name", "source", "expected"),
    [
        ("oscillator", "import solver", [(1, "solver")]),
        ("kepler", "from solver import solve", [(1, "solver")]),
        ("heat1d", "from tasks.heat1d.src import solver", [(1, "tasks.heat1d.src.solver")]),
        ("wave1d", "import tasks.wave1d.src.solver", [(1, "tasks.wave1d.src.solver")]),
    ],
)
def test_scientific_boundary_checker_detects_candidate_solver_imports(
    task_name: str, source: str, expected: list[tuple[int, str]]
) -> None:
    assert _scientific_candidate_solver_imports(task_name, source) == expected


def test_scientific_tests_do_not_import_candidate_solver_modules() -> None:
    violations: list[str] = []
    for task_name in EXPECTED_TASK_NAMES:
        scientific_root = TASK_SOURCE_ROOT / task_name / SCIENTIFIC_TEST_RELATIVE_PATH
        for source_file in sorted(scientific_root.glob("test_*.py")):
            for line, module in _scientific_candidate_solver_imports(
                task_name, source_file.read_text(encoding="utf-8")
            ):
                relative_path = source_file.relative_to(REPOSITORY_ROOT)
                violations.append(
                    f"{task_name}: {relative_path}:{line}: forbidden candidate import {module}"
                )
    assert not violations, "\n".join(violations)
