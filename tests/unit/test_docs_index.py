"""Tests that docs/README.md indexes every doc and that links resolve."""

import re
from pathlib import Path

_LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _link_targets() -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    index = repo_root / "docs" / "README.md"
    return _LINK.findall(index.read_text(encoding="utf-8"))


def test_every_doc_is_linked_from_index() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    docs_dir = repo_root / "docs"
    linked = {
        (docs_dir / target).resolve() for target in _link_targets() if not _SCHEME.match(target)
    }
    unlinked = [
        doc.name
        for doc in sorted(docs_dir.glob("*.md"))
        if doc.name != "README.md" and doc.resolve() not in linked
    ]
    assert not unlinked, f"docs not linked from docs/README.md: {unlinked}"


def test_index_relative_links_resolve() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    docs_dir = repo_root / "docs"
    missing = [
        target
        for target in _link_targets()
        if not _SCHEME.match(target) and not (docs_dir / target).resolve().exists()
    ]
    assert not missing, f"broken links in docs/README.md: {missing}"


def test_no_todo_stub_docs() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    docs_dir = repo_root / "docs"
    stubs = [
        doc.name
        for doc in sorted(docs_dir.glob("*.md"))
        if "TODO: Document" in doc.read_text(encoding="utf-8")
    ]
    assert not stubs, f"stub docs containing 'TODO: Document': {stubs}"
