import importlib.util
import shutil
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_release_metadata.py"
SPEC = importlib.util.spec_from_file_location("check_release_metadata", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
release_metadata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_metadata)


def make_release_tree(
    tmp_path: Path,
    *,
    version: str | None = "1.0.0",
    date: str | None = "2026-10-06",
    changelog_heading: str = "## [1.0.0] - 2026-10-06",
    classifier: str = "Development Status :: 4 - Beta",
) -> Path:
    citation_fields = [
        "cff-version: 1.2.0",
        "title: Test project",
        "authors:",
        "  - family-names: Example",
        "    given-names: Test",
    ]
    if version is not None:
        citation_fields.append(f"version: {version}")
    if date is not None:
        citation_fields.append(f"date-released: {date}")
    (tmp_path / "CITATION.cff").write_text("\n".join(citation_fields) + "\n", encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(
        "## [Unreleased]\n\nUnreleased text.\n\n"
        f"{changelog_heading}\n\n- Release bullet.\n\n"
        "[1.0.0]: https://example.com/compare/1.0.0\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nclassifiers = [\n  "{classifier}",\n]\n',
        encoding="utf-8",
    )
    return tmp_path


def check(root: Path, tag: str = "v1.0.0") -> list[str]:
    return release_metadata.check_release_metadata(tag, root, validate_cff=lambda _: None)


def test_final_and_prerelease_versions_pass(tmp_path: Path) -> None:
    root = make_release_tree(tmp_path)
    assert check(root) == []
    assert check(root, "v1.0.0-rc.1") == []

    make_release_tree(tmp_path, version="1.0.0-rc.1")
    assert check(tmp_path, "v1.0.0-rc.1") == []


def test_version_mismatch_and_missing_version_fail(tmp_path: Path) -> None:
    root = make_release_tree(tmp_path, version="0.9.0")
    assert any("version" in failure for failure in check(root))

    root = make_release_tree(tmp_path, version=None)
    assert any("missing version" in failure for failure in check(root))


@pytest.mark.parametrize(
    ("date_value", "expected"),
    [(None, "missing date-released"), ("2026-13-40", "invalid date")],
)
def test_release_date_is_required_and_valid(
    tmp_path: Path, date_value: str | None, expected: str
) -> None:
    root = make_release_tree(tmp_path, date=date_value)
    assert any(expected in failure for failure in check(root))


def test_changelog_heading_is_required(tmp_path: Path) -> None:
    root = make_release_tree(tmp_path, changelog_heading="## [0.9.0] - 2026-10-06")
    assert any("CHANGELOG.md" in failure for failure in check(root))


def test_alpha_classifier_fails_only_for_final_releases(tmp_path: Path) -> None:
    root = make_release_tree(tmp_path, classifier="Development Status :: 3 - Alpha")
    assert any("Alpha" in failure for failure in check(root))
    assert check(root, "v1.0.0-rc.1") == []


def test_cff_validation_error_is_reported(tmp_path: Path) -> None:
    root = make_release_tree(tmp_path)
    failures = release_metadata.check_release_metadata(
        "v1.0.0", root, validate_cff=lambda _: "schema error"
    )
    assert any("schema error" in failure for failure in failures)


@pytest.mark.parametrize("tag", ["1.0.0", "v1.0"])
def test_invalid_tag_returns_one_failure(tmp_path: Path, tag: str) -> None:
    failures = check(make_release_tree(tmp_path), tag)
    assert len(failures) == 1
    assert "invalid release tag" in failures[0]


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v1.0.0", ("1.0.0", False)),
        ("v1.0.0-rc.1", ("1.0.0", True)),
        ("v1.0.0-a.2", ("1.0.0", True)),
        ("v1.0.0-b1", ("1.0.0", True)),
    ],
)
def test_parse_tag(tag: str, expected: tuple[str, bool]) -> None:
    assert release_metadata.parse_tag(tag) == expected


def test_changelog_section_stops_before_next_heading_and_links() -> None:
    changelog = (
        "## [Unreleased]\n\nUnreleased text.\n\n"
        "## [1.0.0] - 2026-10-06\n\n- Release bullet.\n\n"
        "## [0.9.0]\n\nOld release.\n\n"
        "[1.0.0]: https://example.com/compare/1.0.0\n"
    )
    assert release_metadata.changelog_section(changelog, "1.0.0") == "- Release bullet."


def test_main_writes_notes_and_reports_each_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_release_tree(tmp_path)
    monkeypatch.setattr(release_metadata, "run_cffconvert", lambda _: None)
    notes = tmp_path / "out" / "release-notes.md"
    assert (
        release_metadata.main(
            ["--tag", "v1.0.0", "--root", str(root), "--notes-output", str(notes)]
        )
        == 0
    )
    assert "- Release bullet." in notes.read_text(encoding="utf-8")
    assert "Unreleased text." not in notes.read_text(encoding="utf-8")
    assert "OK: release metadata consistent for v1.0.0" in capsys.readouterr().out

    root = make_release_tree(tmp_path, version="0.9.0", changelog_heading="## [0.9.0] - 2026-10-06")
    assert release_metadata.main(["--tag", "v1.0.0", "--root", str(root)]) == 1
    errors = capsys.readouterr().err.splitlines()
    assert len(errors) == 2
    assert all(line.startswith("FAIL: ") for line in errors)


@pytest.mark.skipif(shutil.which("uvx") is None, reason="uvx is not installed")
def test_real_cffconvert_accepts_repository_citation() -> None:
    assert release_metadata.run_cffconvert(SCRIPT_PATH.parents[1] / "CITATION.cff") is None


@pytest.mark.skipif(shutil.which("uvx") is None, reason="uvx is not installed")
def test_real_cffconvert_rejects_citation_without_authors(tmp_path: Path) -> None:
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        "cff-version: 1.2.0\ntitle: Test\nmessage: Cite this.\n",
        encoding="utf-8",
    )
    assert release_metadata.run_cffconvert(citation) is not None
