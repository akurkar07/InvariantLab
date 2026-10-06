"""Acceptance checks for oracle separation and hidden-file isolation."""

from __future__ import annotations

import ast
import copy
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from invariantlab.experiments import repair
from invariantlab.schema import load_task_contract, load_task_definition
from invariantlab.tasks.workspace import build_agent_workspace

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TASK_NAMES = ("oscillator", "kepler", "heat1d", "wave1d")
ORACLE_MODULES: tuple[str, ...] = (
    "src/invariantlab/verification/analytical.py",
    "src/invariantlab/verification/kepler_oracle.py",
    "src/invariantlab/verification/oracles.py",
)
pytestmark = pytest.mark.v1_acceptance("V1-AC3")


def _normalised_body(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    normalized = copy.deepcopy(func)

    class Normalizer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            node.id = "_"
            return node

        def visit_arg(self, node: ast.arg) -> ast.arg:
            node.arg = "_"
            node.annotation = None
            return node

        def visit_Constant(self, node: ast.Constant) -> ast.Constant:
            return ast.Constant(value=None)

        def _visit_function(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> ast.FunctionDef | ast.AsyncFunctionDef:
            node.name = "_"
            node.returns = None
            node.decorator_list = []
            if node.body and isinstance(node.body[0], ast.Expr):
                first = node.body[0].value
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    node.body = node.body[1:]
                    if not node.body:
                        node.body = [ast.Pass()]
            self.generic_visit(node)
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
            return self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
            return self._visit_function(node)

    normalized = Normalizer().visit(normalized)
    return ast.dump(normalized)


def _function_nodes(source: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _agent_facing_sources() -> list[Path]:
    task_sources = sorted(REPOSITORY_ROOT.glob("tasks/*/src/**/*.py"))
    mutation_sources = sorted(REPOSITORY_ROOT.glob("tasks/*/mutations/**/*.py"))
    sources = task_sources + mutation_sources
    assert sources, "agent-facing Python source glob must not be empty"
    task_names = {path.relative_to(REPOSITORY_ROOT / "tasks").parts[0] for path in task_sources}
    assert set(EXPECTED_TASK_NAMES) <= task_names, (
        f"agent-facing source glob must cover all four task src trees: {task_names}"
    )
    return sources


def test_normalised_body_ignores_names_constants_annotations_and_docstrings() -> None:
    original = ast.parse(
        """
def integrate(position: float, velocity: float, dt: float) -> float:
    \"\"\"Advance one integration step.\"\"\"
    next_position = position + velocity * dt
    return next_position + 0.1
"""
    ).body[0]
    renamed = ast.parse(
        """
def advance(x: int, y: int, step: int) -> int:
    result = x + y * step
    return result + 9.5
"""
    ).body[0]

    assert isinstance(original, ast.FunctionDef)
    assert isinstance(renamed, ast.FunctionDef)
    assert _normalised_body(original) == _normalised_body(renamed)


@pytest.mark.parametrize(
    ("original", "changed"),
    [
        ("return x + v * dt", "return x - v * dt"),
        ("return np.sin(angle)", "return np.cos(angle)"),
    ],
)
def test_normalised_body_preserves_operators_and_attribute_names(
    original: str, changed: str
) -> None:
    first = ast.parse(f"def update(x, v, dt, np, angle):\n    {original}\n").body[0]
    second = ast.parse(f"def update(x, v, dt, np, angle):\n    {changed}\n").body[0]

    assert isinstance(first, ast.FunctionDef)
    assert isinstance(second, ast.FunctionDef)
    assert _normalised_body(first) != _normalised_body(second)


def test_oracle_functions_are_not_copied_into_agent_facing_code() -> None:
    agent_sources = _agent_facing_sources()
    matches: list[str] = []
    for oracle_relative_path in ORACLE_MODULES:
        oracle_path = REPOSITORY_ROOT / oracle_relative_path
        oracle_functions = _function_nodes(oracle_path.read_text(encoding="utf-8"))
        for agent_path in agent_sources:
            agent_relative_path = agent_path.relative_to(REPOSITORY_ROOT).as_posix()
            agent_functions = _function_nodes(agent_path.read_text(encoding="utf-8"))
            for oracle_function in oracle_functions:
                for agent_function in agent_functions:
                    if _normalised_body(oracle_function) == _normalised_body(agent_function):
                        matches.append(
                            f"{oracle_relative_path}:{oracle_function.name} == "
                            f"{agent_relative_path}:{agent_function.name}"
                        )

    assert not matches, "\n".join(matches)


def _oracle_forbidden_imports(source: str) -> list[tuple[int, str]]:
    violations: list[tuple[int, str]] = []
    oracle_solvers = "invariantlab.verification.solvers"
    forbidden_roots = {"solver", "tasks", "candidate_runner"}

    def is_forbidden(module: str, *, relative: bool = False) -> bool:
        if module == oracle_solvers or module.startswith(f"{oracle_solvers}."):
            return True
        unprefixed = module.lstrip(".")
        return unprefixed.split(".", 1)[0] in forbidden_roots or (
            relative and unprefixed == "solvers"
        )

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if is_forbidden(alias.name):
                    violations.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module == oracle_solvers:
                violations.append((node.lineno, oracle_solvers))
            elif node.module == "invariantlab.verification":
                if any(alias.name == "solvers" for alias in node.names):
                    violations.append((node.lineno, oracle_solvers))
            elif node.level:
                prefix = "." * node.level
                if node.module is None:
                    for alias in node.names:
                        module = f"{prefix}{alias.name}"
                        if is_forbidden(module, relative=True):
                            violations.append((node.lineno, module))
                else:
                    module = f"{prefix}{node.module}"
                    if is_forbidden(module, relative=True):
                        violations.append((node.lineno, module))
            elif node.module and is_forbidden(node.module):
                violations.append((node.lineno, node.module))
    return violations


@pytest.mark.parametrize(
    ("source", "module"),
    [
        ("import invariantlab.verification.solvers", "invariantlab.verification.solvers"),
        (
            "from invariantlab.verification.solvers import solve",
            "invariantlab.verification.solvers",
        ),
        (
            "from invariantlab.verification import solvers",
            "invariantlab.verification.solvers",
        ),
        ("from .solvers import solve", ".solvers"),
        ("from . import solvers", ".solvers"),
        ("import solver", "solver"),
        ("from solver.integrator import solve", "solver.integrator"),
        ("import tasks.oscillator.src.solver", "tasks.oscillator.src.solver"),
        ("from tasks import oscillator", "tasks"),
        ("from candidate_runner import run", "candidate_runner"),
    ],
)
def test_oracle_import_checker_detects_forbidden_forms(source: str, module: str) -> None:
    assert _oracle_forbidden_imports(source) == [(1, module)]


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np",
        "from scipy.integrate import solve_ivp",
        "from invariantlab.verification import analytical",
    ],
)
def test_oracle_import_checker_allows_independent_dependencies(source: str) -> None:
    assert _oracle_forbidden_imports(source) == []


def test_oracle_modules_do_not_import_agent_facing_code() -> None:
    violations: list[str] = []
    for relative_path in ORACLE_MODULES:
        source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        violations.extend(
            f"{relative_path}:{line}: forbidden import {module}"
            for line, module in _oracle_forbidden_imports(source)
        )
    assert not violations, "\n".join(violations)


def _normalise_file_bytes(contents: bytes) -> bytes:
    return contents.replace(b"\r\n", b"\n")


def _trusted_hidden_files(repo_root: Path) -> dict[bytes, str]:
    trusted: dict[bytes, str] = {}

    def add(path: Path) -> None:
        trusted[_normalise_file_bytes(path.read_bytes())] = path.relative_to(repo_root).as_posix()

    for scientific_root in sorted((repo_root / "tasks").glob("*/tests/scientific")):
        for path in sorted(scientific_root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                add(path)

    oscillator_dir = repo_root / "tasks" / "oscillator"
    verifier = load_task_definition(oscillator_dir).verifier
    add(oscillator_dir / verifier)

    for relative_path in ORACLE_MODULES:
        add(repo_root / relative_path)

    for mutation_file in sorted((repo_root / "tasks").glob("*/mutations/*/mutation.yaml")):
        add(mutation_file)

    return trusted


def _hidden_exposures(files: dict[str, bytes], trusted: dict[bytes, str]) -> list[str]:
    oracle_filenames = {PurePosixPath(module).name for module in ORACLE_MODULES}
    hidden_filenames = {"verifier.py", "evaluate.py", "mutation.yaml"} | oracle_filenames
    exposures: list[str] = []
    for relative_path, contents in files.items():
        path = PurePosixPath(relative_path)
        reasons: list[str] = []
        if "scientific" in path.parts:
            reasons.append("scientific path")
        if path.name in hidden_filenames:
            reasons.append("hidden filename")
        trusted_label = trusted.get(_normalise_file_bytes(contents))
        if trusted_label is not None:
            reasons.append(f"matches trusted file {trusted_label}")
        if reasons:
            exposures.append(f"{relative_path}: {', '.join(reasons)}")
    return exposures


@pytest.mark.parametrize("task_name", EXPECTED_TASK_NAMES)
def test_agent_workspace_exposes_no_hidden_files(task_name: str, tmp_path: Path) -> None:
    task_dir = REPOSITORY_ROOT / "tasks" / task_name
    workspace = build_agent_workspace(task_dir, tmp_path / "ws")
    contract = load_task_contract(task_dir)
    visible_files = {
        path.relative_to(workspace).as_posix(): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }

    assert visible_files, "agent workspace must not be empty"
    assert (workspace / contract.entrypoint).is_file(), (
        "workspace must include its solver entrypoint"
    )
    assert _hidden_exposures(visible_files, _trusted_hidden_files(REPOSITORY_ROOT)) == []


def test_candidate_stage_mount_exposes_no_hidden_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_dir = REPOSITORY_ROOT / "tasks" / "oscillator"
    task = load_task_definition(task_dir)
    source = (task_dir / "src" / "solver.py").read_text(encoding="utf-8")
    calls: list[dict[str, bytes]] = []

    def fake_run_container(
        image: str, mounts: list[str], args: list[str]
    ) -> subprocess.CompletedProcess[str]:
        files: dict[str, bytes] = {}
        for mount in mounts:
            host, target, _mode = mount.rsplit(":", 2)
            host_path = Path(host)
            for path in host_path.rglob("*"):
                if path.is_file():
                    relative_target = (
                        (
                            PurePosixPath(target)
                            / PurePosixPath(path.relative_to(host_path).as_posix())
                        )
                        .as_posix()
                        .lstrip("/")
                    )
                    files[relative_target] = path.read_bytes()
        calls.append(files)
        return subprocess.CompletedProcess(args, 0, stdout='{"passed": true}\n', stderr="")

    monkeypatch.setattr(repair, "_run_container", fake_run_container)
    assert repair._evaluate_source(source, task_dir, task, "python:3.12-slim") == {"passed": True}

    candidate_bytes = _normalise_file_bytes(source.encode("utf-8"))
    candidate_stage_calls = [
        files
        for files in calls
        if any(_normalise_file_bytes(contents) == candidate_bytes for contents in files.values())
    ]
    assert candidate_stage_calls, "expected at least one candidate-stage container call"

    visible_files = {
        relative_path: contents
        for files in candidate_stage_calls
        for relative_path, contents in files.items()
    }
    assert _hidden_exposures(visible_files, _trusted_hidden_files(REPOSITORY_ROOT)) == []


def test_hidden_file_checker_rejects_pre_fix_single_mount_layout(tmp_path: Path) -> None:
    oscillator_dir = REPOSITORY_ROOT / "tasks" / "oscillator"
    task = load_task_definition(oscillator_dir)
    old_layout = tmp_path / "old-layout"
    old_layout.mkdir()
    (old_layout / "solver.py").write_text(
        (oscillator_dir / "src" / "solver.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (old_layout / "evaluate.py").write_bytes((oscillator_dir / task.verifier).read_bytes())
    visible_files = {
        path.relative_to(old_layout).as_posix(): path.read_bytes()
        for path in old_layout.rglob("*")
        if path.is_file()
    }

    exposures = _hidden_exposures(visible_files, _trusted_hidden_files(REPOSITORY_ROOT))
    assert any("evaluate.py" in exposure for exposure in exposures), exposures
