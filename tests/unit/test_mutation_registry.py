"""Unit tests for curated task mutant discovery."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from invariantlab.mutations import (
    MutationRegistryError,
    RegisteredMutant,
    discover_mutants,
    validate_mutant,
)


def _manifest(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": "bad-sign",
        "task_id": "wave",
        "family": "sign_error",
        "expected_effect": "force points in the wrong direction",
        "expected_failures": [
            {
                "test": "tests/scientific/test_reference.py::test_x",
                "message": "reference mismatch",
            }
        ],
    }
    data.update(overrides)
    return data


def _write_task(tasks_root: Path, name: str) -> Path:
    task_dir = tasks_root / name
    (task_dir / "tests" / "scientific").mkdir(parents=True)
    (task_dir / "tests" / "scientific" / "test_reference.py").write_text(
        "def test_x():\n    pass\n", encoding="utf-8"
    )
    (task_dir / "contract.yaml").write_text(
        yaml.safe_dump(
            {
                "id": name,
                "family": "wave_1d",
                "output": {
                    "path": "result.npz",
                    "arrays": [{"name": "result", "shape": [None], "dtype": "float64"}],
                },
            }
        ),
        encoding="utf-8",
    )
    return task_dir


def _write_mutant(task_dir: Path, directory: str, manifest: dict[str, object] | str) -> Path:
    mutation_dir = task_dir / "mutations" / directory
    mutation_dir.mkdir(parents=True)
    if isinstance(manifest, str):
        (mutation_dir / "mutation.yaml").write_text(manifest, encoding="utf-8")
    else:
        (mutation_dir / "mutation.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (mutation_dir / "solver.py").write_text("def solve():\n    return None\n", encoding="utf-8")
    return mutation_dir


def test_discover_mutants_returns_sorted_registered_manifests(tmp_path: Path) -> None:
    """Discovery validates manifests and sorts by task then mutant id."""
    tasks_root = tmp_path / "tasks"
    heat_dir = _write_task(tasks_root, "heat")
    wave_dir = _write_task(tasks_root, "wave")
    alpha_dir = _write_mutant(heat_dir, "alpha", _manifest(id="alpha", task_id="heat"))
    _write_mutant(wave_dir, "zeta", _manifest(id="zeta", task_id="wave"))
    beta_dir = _write_mutant(wave_dir, "beta", _manifest(id="beta", task_id="wave"))

    mutants = discover_mutants(tasks_root)

    assert [(mutant.task_dir.name, mutant.definition.id) for mutant in mutants] == [
        ("heat", "alpha"),
        ("wave", "beta"),
        ("wave", "zeta"),
    ]
    assert all(isinstance(mutant, RegisteredMutant) for mutant in mutants)
    assert mutants[0].contract.id == "heat"
    assert mutants[0].mutation_dir == alpha_dir
    assert mutants[0].source_path == alpha_dir / "solver.py"
    assert mutants[0].definition.expected_effect == "force points in the wrong direction"
    assert mutants[1].mutation_dir == beta_dir


def test_task_without_mutations_has_no_mutants(tmp_path: Path) -> None:
    """Tasks without a mutations directory are valid and contribute nothing."""
    tasks_root = tmp_path / "tasks"
    _write_task(tasks_root, "wave")

    assert discover_mutants(tasks_root) == []


@pytest.mark.parametrize(
    ("manifest", "reason"),
    [
        ("id: [\n", "while parsing"),
        ({**_manifest(), "unexpected": True}, "extra_forbidden"),
        ({**_manifest(), "id": "different"}, "does not match directory name"),
        ({**_manifest(), "task_id": "other"}, "does not match contract id"),
        ({**_manifest(), "family": "not_a_family"}, "Input should be"),
        ({**_manifest(), "source": "C:/outside.py"}, "safe relative path"),
        ({**_manifest(), "source": "../outside.py"}, "safe relative path"),
        ({**_manifest(), "source": "absent.py"}, "does not exist"),
        (
            {
                **_manifest(),
                "expected_failures": [
                    {
                        "test": "tests/public/test_reference.py::test_x",
                        "message": "reference mismatch",
                    }
                ],
            },
            "tests/scientific/",
        ),
        (
            {
                **_manifest(),
                "expected_failures": [
                    {
                        "test": "tests/scientific/missing.py::test_x",
                        "message": "reference mismatch",
                    }
                ],
            },
            "does not exist inside the task",
        ),
    ],
    ids=[
        "malformed-yaml",
        "unknown-key",
        "id-directory-mismatch",
        "task-id-mismatch",
        "unknown-family",
        "absolute-source",
        "parent-source",
        "missing-source",
        "test-outside-scientific",
        "missing-test-file",
    ],
)
def test_rejects_invalid_manifest(
    tmp_path: Path, manifest: dict[str, object] | str, reason: str
) -> None:
    """Invalid manifest declarations fail discovery with a useful diagnostic."""
    tasks_root = tmp_path / "tasks"
    task_dir = _write_task(tasks_root, "wave")
    _write_mutant(task_dir, "bad-sign", manifest)

    with pytest.raises(MutationRegistryError) as exc_info:
        discover_mutants(tasks_root)

    assert reason in str(exc_info.value)
    assert (task_dir / "mutations" / "bad-sign" / "mutation.yaml").as_posix() in str(exc_info.value)


def test_rejects_duplicate_ids_within_task(tmp_path: Path) -> None:
    """Each duplicate declaration is reported by its own manifest path."""
    tasks_root = tmp_path / "tasks"
    task_dir = _write_task(tasks_root, "wave")
    _write_mutant(task_dir, "first", _manifest(id="dup"))
    _write_mutant(task_dir, "second", _manifest(id="dup"))

    with pytest.raises(MutationRegistryError) as exc_info:
        discover_mutants(tasks_root)

    assert "duplicate id 'dup'" in str(exc_info.value)
    assert len(exc_info.value.problems) >= 2


def test_reports_all_problems_across_broken_mutants(tmp_path: Path) -> None:
    """Discovery accumulates independent manifest problems before raising."""
    tasks_root = tmp_path / "tasks"
    task_dir = _write_task(tasks_root, "wave")
    _write_mutant(task_dir, "first", _manifest(id="first", source="absent.py"))
    _write_mutant(task_dir, "second", _manifest(id="second", source="../escape.py"))

    with pytest.raises(MutationRegistryError) as exc_info:
        discover_mutants(tasks_root)

    assert len(exc_info.value.problems) == 2
    assert "does not exist" in str(exc_info.value)
    assert "safe relative path" in str(exc_info.value)


def test_legacy_mutants_are_opt_in(tmp_path: Path) -> None:
    """Legacy study manifests are validated but excluded unless requested."""
    tasks_root = tmp_path / "tasks"
    task_dir = _write_task(tasks_root, "wave")
    package = _manifest(id="package", task_id="wave")
    legacy = _manifest(id="legacy", task_id="wave", interface="legacy_study")
    _write_mutant(task_dir, "package", package)
    _write_mutant(task_dir, "legacy", legacy)

    assert [mutant.definition.id for mutant in discover_mutants(tasks_root)] == ["package"]
    assert [
        mutant.definition.id for mutant in discover_mutants(tasks_root, include_legacy=True)
    ] == ["legacy", "package"]


def test_real_task_legacy_mutants_are_opt_in() -> None:
    """The checked-in oscillator repair mutants remain available to legacy studies."""
    repo_root = Path(__file__).resolve().parents[2]
    tasks_root = repo_root / "tasks"

    legacy = [
        (mutant.task_dir.name, mutant.definition.id)
        for mutant in discover_mutants(tasks_root, include_legacy=True)
        if mutant.definition.interface == "legacy_study"
    ]

    assert legacy == [("oscillator", "sign-error"), ("oscillator", "update-order")]


def test_real_task_package_mutants_validate() -> None:
    """Every curated package mutant passes validate_mutant against unmodified task suites."""
    repo_root = Path(__file__).resolve().parents[2]
    mutants = discover_mutants(repo_root / "tasks")

    assert [
        (mutant.task_dir.name, mutant.definition.id, mutant.definition.family.value)
        for mutant in mutants
    ] == [
        ("oscillator", "non-conservative-damping", "non_conservative_update"),
        ("wave1d", "courant-not-squared", "discretisation_error"),
        ("wave1d", "dirichlet-wrong-node", "boundary_error"),
        ("wave1d", "sign-error-startup", "sign_error"),
        ("wave1d", "unstable-time-recurrence", "stability_error"),
        ("wave1d", "update-order-overwrite", "update_order_error"),
    ]
    for mutant in mutants:
        result = validate_mutant(mutant)
        assert result.passed, (mutant.definition.id, result.reasons)
