"""Render a self-contained static HTML report from validated run evidence."""

from __future__ import annotations

import difflib
from html import escape
from typing import Any


def _escaped(value: Any) -> str:
    return escape(str(value), quote=True)


def _pass_status(value: Any) -> str:
    return "PASS" if value else "FAIL"


def render_html_report(
    *,
    experiment_name: str,
    model_id: str,
    seed: int,
    events_sha256: str,
    generation_command: str,
    baseline: dict[str, Any],
    baseline_source: str,
    baseline_source_label: str,
    by_condition_rows: list[dict[str, str]],
    records: list[dict[str, Any]],
) -> str:
    """Render provenance, aggregate outcomes, and every canonical sample."""
    condition_rows: list[str] = []
    for row in by_condition_rows:
        condition = row["condition"]
        condition_links = " ".join(
            f'<a href="#sample-{_escaped(record["condition"])}-'
            f'{_escaped(record["trial"])}">trial {_escaped(record["trial"])}</a>'
            for record in records
            if record["condition"] == condition
        )
        fields = (
            "condition",
            "scientific_passes",
            "n",
            "scientific_pass_rate",
            "wilson_low",
            "wilson_high",
            "scientific_regressions",
            "median_worst_scientific_ratio",
            "pass_rate_difference_vs_weak",
        )
        cells = "".join(
            f'<td data-field="{_escaped(field)}">{_escaped(row[field])}</td>' for field in fields
        )
        condition_rows.append(
            f'<tr data-condition="{_escaped(condition)}">{cells}<td>{condition_links}</td></tr>'
        )

    sample_sections: list[str] = []
    for record in records:
        condition = record["condition"]
        trial = record["trial"]
        anchor = f"sample-{_escaped(condition)}-{_escaped(trial)}"
        repaired = record["repaired"]
        severity_items = "".join(
            f"<dt>{_escaped(key)}</dt><dd>{_escaped(record['severity'][key])}</dd>"
            for key in sorted(record["severity"])
        )
        candidate_error = record.get("candidate_error", "") or "none"
        candidate_source = record.get("candidate_source", "")
        diff = "".join(
            difflib.unified_diff(
                baseline_source.splitlines(keepends=True),
                candidate_source.splitlines(keepends=True),
                fromfile="baseline_solver.py",
                tofile="candidate_source",
                lineterm="\n",
            )
        )
        if not diff:
            diff = "(no differences)"
        sample_sections.append(
            f'<section class="sample" id="{anchor}">'
            f"<h3>{_escaped(condition)} — trial {_escaped(trial)}</h3>"
            "<dl>"
            f"<dt>Public pass</dt><dd>{_pass_status(repaired['public_passed'])}</dd>"
            "<dt>Scientific pass</dt>"
            f"<dd>{_pass_status(repaired['scientific_passed'])}</dd>"
            f"<dt>Successful repair</dt><dd>{_escaped(record['successful_repair'])}</dd>"
            f"<dt>Scientific regression</dt>"
            f"<dd>{_escaped(record['scientific_regression'])}</dd>"
            f"{severity_items}"
            f"<dt>Candidate error</dt><dd>{_escaped(candidate_error)}</dd>"
            "</dl>"
            "<details><summary>Candidate source diff</summary>"
            f"<pre>{_escaped(diff)}</pre></details></section>"
        )

    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>InvariantLab evidence report</title>\n"
        "<style>\n"
        "body{font-family:system-ui,sans-serif;line-height:1.5;margin:2rem auto;"
        "max-width:1100px;padding:0 1rem;color:#17202a}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0}"
        "th,td{border:1px solid #ccd2d8;padding:.45rem;text-align:left}"
        "th{background:#f2f4f6}.provenance, .baseline, .sample{"
        "border:1px solid #ccd2d8;border-radius:.4rem;padding:1rem;margin:1rem 0}"
        "dt{font-weight:700}dd{margin:0 0 .5rem}pre{overflow:auto;"
        "background:#f6f8fa;padding:1rem}a{margin-right:.5rem}"
        "</style>\n</head>\n<body>\n"
        "<h1>InvariantLab evidence report</h1>\n"
        '<section class="provenance"><h2>Provenance</h2><dl>'
        f"<dt>Experiment</dt><dd>{_escaped(experiment_name)}</dd>"
        f"<dt>Model ID</dt><dd>{_escaped(model_id)}</dd>"
        f"<dt>Seed</dt><dd>{_escaped(seed)}</dd>"
        f"<dt>events.jsonl SHA-256</dt><dd>{_escaped(events_sha256)}</dd>"
        f"<dt>Generation command</dt><dd><code>{_escaped(generation_command)}</code></dd>"
        "</dl></section>\n"
        '<section class="baseline"><h2>Baseline status</h2><dl>'
        f"<dt>Public</dt><dd>{_pass_status(baseline.get('public_passed'))}</dd>"
        f"<dt>Scientific</dt><dd>{_pass_status(baseline.get('scientific_passed'))}</dd>"
        f"<dt>Source</dt><dd>{_escaped(baseline_source_label)}</dd>"
        "</dl></section>\n"
        "<section><h2>Results by condition</h2><table>"
        "<thead><tr><th>Condition</th><th>Scientific passes</th><th>Total (n)</th>"
        "<th>Pass rate</th><th>Wilson 95% low</th><th>Wilson 95% high</th>"
        "<th>Regressions</th><th>Median worst ratio</th><th>Difference vs weak</th>"
        "<th>Samples</th></tr></thead><tbody>"
        f"{''.join(condition_rows)}"
        "</tbody></table></section>\n"
        "<section><h2>Samples</h2>"
        f"{''.join(sample_sections)}"
        "</section>\n</body>\n</html>\n"
    )
