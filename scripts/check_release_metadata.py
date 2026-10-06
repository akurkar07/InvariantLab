"""Validate release tags against the repository's release metadata."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


_TAG_PATTERN = re.compile(r"v(\d+\.\d+\.\d+)(?:-(rc|a|b)\.?(\d+))?")


def parse_tag(tag: str) -> tuple[str, bool]:
    """Return a release's base version and whether it is a prerelease."""
    match = _TAG_PATTERN.fullmatch(tag)
    if match is None:
        raise ValueError(f"invalid release tag {tag!r}; expected vX.Y.Z or vX.Y.Z-(rc|a|b)N")
    return match.group(1), match.group(2) is not None


def run_cffconvert(path: Path) -> str | None:
    """Validate a citation file using cffconvert, returning an error on failure."""
    try:
        result = subprocess.run(
            ["uvx", "cffconvert", "--validate", "-i", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        return "could not run cffconvert: uvx was not found"
    except OSError as error:
        return f"could not run cffconvert: {error}"

    if result.returncode == 0:
        return None

    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    tail = output[-1000:] if output else "no stdout or stderr"
    return f"cffconvert exited with status {result.returncode}: {tail}"


def _valid_release_date(value: object) -> bool:
    if isinstance(value, datetime):
        return False
    if isinstance(value, date):
        return True
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _read_text(path: Path, label: str, failures: list[str]) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        failures.append(f"{label} is missing")
    except (OSError, UnicodeError) as error:
        failures.append(f"could not read {label}: {error}")
    return None


def changelog_section(changelog_text: str, version: str) -> str:
    """Extract a version's changelog body, excluding later headings and references."""
    heading = re.compile(rf"^## \[{re.escape(version)}\].*$", re.MULTILINE)
    match = heading.search(changelog_text)
    if match is None:
        return ""

    body_start = match.end()
    next_heading = re.search(r"^## ", changelog_text[body_start:], re.MULTILINE)
    body_end = body_start + next_heading.start() if next_heading else len(changelog_text)
    body = changelog_text[body_start:body_end]

    link_reference = re.search(r"^\[[^\]]+\]:\s*https?://", body, re.MULTILINE)
    if link_reference is not None:
        body = body[: link_reference.start()]
    return body.strip()


def check_release_metadata(
    tag: str,
    root: Path,
    *,
    validate_cff: Callable[[Path], str | None] = run_cffconvert,
) -> list[str]:
    """Return human-readable failures for metadata associated with a release tag."""
    try:
        base_version, is_prerelease = parse_tag(tag)
    except ValueError as error:
        return [str(error)]

    failures: list[str] = []
    citation_path = root / "CITATION.cff"
    citation_text = _read_text(citation_path, "CITATION.cff", failures)
    if citation_text is not None:
        try:
            citation = yaml.safe_load(citation_text)
        except ValueError as error:
            failures.append(f"CITATION.cff has an invalid date: {error}")
        except yaml.YAMLError as error:
            failures.append(f"CITATION.cff is not valid YAML: {error}")
        else:
            if not isinstance(citation, dict):
                failures.append("CITATION.cff must contain a YAML mapping")
            else:
                version = citation.get("version")
                accepted_versions = {base_version}
                if is_prerelease:
                    accepted_versions.add(tag[1:])
                if version is None:
                    failures.append("CITATION.cff is missing version")
                elif str(version) not in accepted_versions:
                    expected = " or ".join(sorted(accepted_versions))
                    failures.append(
                        f"CITATION.cff version {str(version)!r} does not match {expected}"
                    )

                released = citation.get("date-released")
                if released is None:
                    failures.append("CITATION.cff is missing date-released")
                elif not _valid_release_date(released):
                    failures.append(
                        f"CITATION.cff date-released {str(released)!r} is not a valid ISO date"
                    )

        validation_error = validate_cff(citation_path)
        if validation_error is not None:
            failures.append(f"CITATION.cff validation failed: {validation_error}")

    changelog_path = root / "CHANGELOG.md"
    changelog_text = _read_text(changelog_path, "CHANGELOG.md", failures)
    if changelog_text is not None and not re.search(
        rf"^## \[{re.escape(base_version)}\]", changelog_text, re.MULTILINE
    ):
        failures.append(f"CHANGELOG.md has no heading for [{base_version}]")

    if not is_prerelease:
        pyproject_path = root / "pyproject.toml"
        pyproject_text = _read_text(pyproject_path, "pyproject.toml", failures)
        if pyproject_text is not None:
            if tomllib is None:
                failures.append("TOML parser is unavailable (install tomli on Python 3.10)")
            else:
                try:
                    project = tomllib.loads(pyproject_text).get("project", {})
                except (tomllib.TOMLDecodeError, AttributeError) as error:
                    failures.append(f"pyproject.toml is not valid TOML: {error}")
                else:
                    classifiers = project.get("classifiers", [])
                    if "Development Status :: 3 - Alpha" in classifiers:
                        failures.append(
                            "pyproject.toml still contains the Alpha development-status classifier"
                        )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="release tag, e.g. v1.0.0 or v1.0.0-rc.1")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--notes-output", type=Path)
    args = parser.parse_args(argv)

    failures = check_release_metadata(args.tag, args.root, validate_cff=run_cffconvert)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    if args.notes_output is not None:
        base_version, _ = parse_tag(args.tag)
        changelog = (args.root / "CHANGELOG.md").read_text(encoding="utf-8")
        try:
            args.notes_output.parent.mkdir(parents=True, exist_ok=True)
            args.notes_output.write_text(
                changelog_section(changelog, base_version) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            print(f"FAIL: could not write release notes: {error}", file=sys.stderr)
            return 1

    print(f"OK: release metadata consistent for {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
