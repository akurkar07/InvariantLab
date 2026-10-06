"""Tests that published study intervals match the repair runner."""

import re
from pathlib import Path

from invariantlab.experiments.repair import _wilson_interval

_COUNT = re.compile(r"^\**\s*(\d+)\s*/\s*(\d+)\s*\**$")
_PROPORTION = re.compile(r"^\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]$")
_PERCENT = re.compile(r"^([0-9.]+)%\s+to\s+([0-9.]+)%$")
_SEPARATOR = re.compile(r":?-{3,}:?")


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _published_intervals() -> tuple[
    list[tuple[str, int, int, int, float, float]], set[tuple[str, int]]
]:
    repo_root = Path(__file__).resolve().parents[2]
    docs_dir = repo_root / "docs"
    rows = []
    percent_rows = set()

    for report in sorted(docs_dir.glob("*.md")):
        lines = report.read_text(encoding="utf-8").splitlines()
        for header_index, header_line in enumerate(lines[:-1]):
            if not header_line.strip().startswith("|"):
                continue
            headers = _table_cells(header_line)
            separators = _table_cells(lines[header_index + 1])
            if len(headers) != len(separators) or not all(
                _SEPARATOR.fullmatch(cell) for cell in separators
            ):
                continue

            wilson_index = next(
                (
                    index
                    for index, header in enumerate(headers)
                    if "wilson" in header.casefold()
                ),
                None,
            )
            if wilson_index is None:
                continue

            row_index = header_index + 2
            while row_index < len(lines) and lines[row_index].strip().startswith("|"):
                cells = _table_cells(lines[row_index])
                count_match = next(
                    (
                        match
                        for cell in cells
                        if (match := _COUNT.fullmatch(cell))
                    ),
                    None,
                )
                if count_match is not None:
                    wilson_cell = (
                        cells[wilson_index] if wilson_index < len(cells) else ""
                    )
                    if wilson_cell:
                        location = (
                            f"{report.relative_to(repo_root)}:{row_index + 1}"
                        )
                        proportion_match = _PROPORTION.fullmatch(wilson_cell)
                        percent_match = _PERCENT.fullmatch(wilson_cell)
                        if proportion_match is not None:
                            published_lo, published_hi = map(
                                float, proportion_match.groups()
                            )
                        elif percent_match is not None:
                            published_lo, published_hi = map(
                                float, percent_match.groups()
                            )
                            percent_rows.add(
                                (str(report.relative_to(repo_root)), row_index + 1)
                            )
                        else:
                            raise AssertionError(
                                f"{location}: unsupported Wilson interval "
                                f"{wilson_cell!r}"
                            )

                        rows.append(
                            (
                                str(report.relative_to(repo_root)),
                                row_index + 1,
                                int(count_match.group(1)),
                                int(count_match.group(2)),
                                published_lo,
                                published_hi,
                            )
                        )
                row_index += 1

    return rows, percent_rows


def test_study_reports_publish_at_least_ten_wilson_intervals():
    rows, _ = _published_intervals()

    assert len(rows) >= 10, f"Expected at least 10 intervals, found {len(rows)}"


def test_study_report_intervals_match_the_runner_formula():
    rows, percent_rows = _published_intervals()
    mismatches = []

    for report, line, successes, total, published_lo, published_hi in rows:
        lo, hi = _wilson_interval(successes, total)
        if (report, line) in percent_rows:
            published = (published_lo, published_hi)
            expected = (round(100 * lo, 1), round(100 * hi, 1))
        else:
            published = (published_lo, published_hi)
            expected = (round(lo, 3), round(hi, 3))

        if published != expected:
            mismatches.append(
                f"{report}:{line} {successes}/{total}: "
                f"published {published} vs expected {expected}"
            )

    assert not mismatches, "\n".join(mismatches)
