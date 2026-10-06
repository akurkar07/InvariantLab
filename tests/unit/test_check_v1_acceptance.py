"""Tests for scripts/check_v1_acceptance.py using a temporary test tree."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_v1_acceptance.py"

ALL_IDS = [f"V1-AC{i}" for i in range(1, 9)]


def _write_tree(tmp_path: Path, body: str) -> Path:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "test_ac.py").write_text(body, encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    return tree


def _marked_tests(ids: list[str], extra: str = "") -> str:
    lines = ["import pytest", ""]
    for ac_id in ids:
        name = ac_id.lower().replace("-", "_")
        lines.append(f'@pytest.mark.v1_acceptance("{ac_id}")')
        lines.append(f"def test_{name}():")
        lines.append("    assert True")
        lines.append("")
    lines.append(extra)
    return "\n".join(lines)


def _run(tmp_path: Path, body: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    tree = _write_tree(tmp_path, body)
    report = tmp_path / "report.json"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--report", str(report), str(tree)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    return proc, json.loads(report.read_text(encoding="utf-8"))


def test_all_criteria_pass(tmp_path: Path) -> None:
    body = _marked_tests(ALL_IDS, extra="def test_unmarked_fails():\n    assert False\n")
    proc, report = _run(tmp_path, body)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert report["ok"] is True
    for ac_id in ALL_IDS:
        assert report["criteria"][ac_id]["status"] == "passed"
    assert "PASSED" in proc.stdout


def test_missing_criterion_fails(tmp_path: Path) -> None:
    proc, report = _run(tmp_path, _marked_tests(ALL_IDS[:7]))
    assert proc.returncode != 0
    assert "V1-AC8" in proc.stdout
    assert report["criteria"]["V1-AC8"]["status"] == "missing"
    assert report["ok"] is False


def test_skipped_criterion_fails(tmp_path: Path) -> None:
    body = _marked_tests(ALL_IDS).replace(
        "def test_v1_ac3():\n    assert True",
        'def test_v1_ac3():\n    pytest.skip("docker")',
    )
    proc, report = _run(tmp_path, body)
    assert proc.returncode != 0
    assert report["criteria"]["V1-AC3"]["status"] == "skipped"
    assert report["ok"] is False


def test_unknown_id_fails(tmp_path: Path) -> None:
    extra = '@pytest.mark.v1_acceptance("V1-AC9")\ndef test_v1_ac9():\n    assert True\n'
    proc, report = _run(tmp_path, _marked_tests(ALL_IDS, extra=extra))
    assert proc.returncode != 0
    assert any(e["id"] == "V1-AC9" for e in report["unknown_ids"])
    assert "V1-AC9" in proc.stdout
    assert report["ok"] is False


def test_xfailed_criterion_fails(tmp_path: Path) -> None:
    body = (
        _marked_tests(ALL_IDS)
        .replace(
            "def test_v1_ac4():\n    assert True",
            "def test_v1_ac4():\n    assert False",
        )
        .replace(
            '@pytest.mark.v1_acceptance("V1-AC4")\ndef test_v1_ac4',
            '@pytest.mark.v1_acceptance("V1-AC4")\n@pytest.mark.xfail(reason="x")\ndef test_v1_ac4',
        )
    )
    proc, report = _run(tmp_path, body)
    assert proc.returncode != 0
    assert report["criteria"]["V1-AC4"]["status"] == "skipped"
    assert report["ok"] is False


def test_failing_criterion_fails(tmp_path: Path) -> None:
    body = _marked_tests(ALL_IDS).replace(
        "def test_v1_ac5():\n    assert True",
        "def test_v1_ac5():\n    assert False",
    )
    proc, report = _run(tmp_path, body)
    assert proc.returncode != 0
    assert report["criteria"]["V1-AC5"]["status"] == "failed"
    assert report["ok"] is False
