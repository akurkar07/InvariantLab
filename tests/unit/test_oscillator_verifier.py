"""Unit tests for the two-stage oscillator verifier pipeline (no Docker)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

TASK = Path(__file__).resolve().parents[2] / "tasks" / "oscillator"


def _thresholds() -> dict[str, float]:
    from invariantlab.schema import load_task_definition

    task = load_task_definition(TASK)
    return {name: spec.threshold for name, spec in task.feedback_metrics.items()}


def _thresholds_file(tmp_path: Path) -> str:
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(_thresholds()), encoding="utf-8")
    return str(path)

CORRECT_SOLVER = """def solve_oscillator_verlet(x0, v0, omega, dt, n_steps):
    trajectory = [(0.0, float(x0), float(v0))]
    x = float(x0)
    v = float(v0)
    omega2 = float(omega) * float(omega)
    t = 0.0

    for _ in range(int(n_steps)):
        a = -omega2 * x
        v_half = v + 0.5 * dt * a
        x = x + dt * v_half
        a_new = -omega2 * x
        v = v_half + 0.5 * dt * a_new
        t += dt
        trajectory.append((t, x, v))

    return trajectory
"""

FAKE_VERDICT_SOLVER = """import json, sys
print(json.dumps({"public_passed": True, "scientific_passed": True, "public": {}, "scientific": {}, "metrics": {}}))
sys.exit(0)
"""

GARBAGE_RETURN_SOLVER = """import json
def solve_oscillator_verlet(x0, v0, omega, dt, n_steps):
    print(json.dumps({"public_passed": True, "scientific_passed": True, "public": {}, "scientific": {}, "metrics": {}}))
    return "garbage"
"""


def _evaluate(source: str, tmp_path: Path) -> dict:
    solver = tmp_path / "solver.py"
    out = tmp_path / "trajectories.json"
    solver.write_text(source, encoding="utf-8")
    subprocess.run(
        [sys.executable, str(TASK / "candidate_runner.py"), str(solver), str(out)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    completed = subprocess.run(
        [sys.executable, str(TASK / "verifier.py"), str(out), _thresholds_file(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _verify_payload(payload: str, tmp_path: Path) -> dict:
    out = tmp_path / "payload.json"
    out.write_text(payload, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(TASK / "verifier.py"), str(out), _thresholds_file(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_update_order_mutation_metrics(tmp_path):
    source = (TASK / "mutations" / "update-order" / "solver.py").read_text(
        encoding="utf-8"
    )
    result = _evaluate(source, tmp_path)
    assert result["public_passed"] is True
    assert result["scientific_passed"] is False
    assert result["metrics"]["max_state_relative_error"] == pytest.approx(
        2.07e-2, rel=5e-3
    )
    assert result["metrics"]["max_energy_relative_drift"] == pytest.approx(
        4.59e-2, rel=5e-3
    )


def test_correct_solver_passes(tmp_path):
    result = _evaluate(CORRECT_SOLVER, tmp_path)
    assert result["public_passed"] is True
    assert result["scientific_passed"] is True


def test_correct_solver_has_10x_margin_below_study_gate(tmp_path):
    thresholds = _thresholds()
    result = _evaluate(CORRECT_SOLVER, tmp_path)
    assert result["scientific_passed"] is True
    # The metrics are maxima over the verifier cases, so this bounds every case.
    for name, threshold in thresholds.items():
        assert result["metrics"][name] < threshold / 10


def test_update_order_mutant_exceeds_every_threshold(tmp_path):
    source = (TASK / "mutations" / "update-order" / "solver.py").read_text(
        encoding="utf-8"
    )
    result = _evaluate(source, tmp_path)
    for name, threshold in _thresholds().items():
        assert result["metrics"][name] > threshold


def test_verifier_requires_thresholds_argument(tmp_path):
    out = tmp_path / "trajectories.json"
    out.write_text("{}", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(TASK / "verifier.py"), str(out)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode != 0


def test_forged_verdict_in_module_body_fails(tmp_path):
    result = _evaluate(FAKE_VERDICT_SOLVER, tmp_path)
    assert result["public_passed"] is False
    assert result["scientific_passed"] is False
    assert "error" in result["metrics"]


def test_forged_verdict_and_garbage_return_fails(tmp_path):
    result = _evaluate(GARBAGE_RETURN_SOLVER, tmp_path)
    assert result["public_passed"] is False
    assert result["scientific_passed"] is False


def test_verifier_missing_file(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(TASK / "verifier.py"),
            str(tmp_path / "nope.json"),
            _thresholds_file(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["public_passed"] is False
    assert result["scientific_passed"] is False
    assert "error" in result["metrics"]


def test_verifier_non_json(tmp_path):
    result = _verify_payload("this is not json", tmp_path)
    assert result["public_passed"] is False
    assert result["scientific_passed"] is False
    assert "error" in result["metrics"]


def test_verifier_top_level_list(tmp_path):
    result = _verify_payload("[]", tmp_path)
    assert result["public_passed"] is False
    assert result["scientific_passed"] is False
    assert "error" in result["metrics"]


def test_verifier_cases_wrong_length(tmp_path):
    payload = json.dumps(
        {
            "short": [[0.0, 1.0, 0.0]] * 3,
            "cases": [[[0.0, 1.0, 0.0]]],
        }
    )
    result = _verify_payload(payload, tmp_path)
    assert result["public_passed"] is False
    assert result["scientific_passed"] is False
    assert "error" in result["metrics"]


def test_verifier_case_row_wrong_length(tmp_path):
    good = [[0.0, 1.0, 0.0], [0.01, 1.0, -0.01]]
    cases = [list(good), list(good), [[0.0, 1.0]]]
    payload = json.dumps({"short": [*good, [0.02, 1.0, -0.02]], "cases": cases})
    result = _verify_payload(payload, tmp_path)
    assert result["public_passed"] is False
    assert result["scientific_passed"] is False
    assert "error" in result["metrics"]


def test_verifier_case_nan_rejected(tmp_path):
    good = [[0.0, 1.0, 0.0], [0.01, 1.0, -0.01]]
    cases = [list(good), [[0.0, 1.0, 0.0], [0.01, float("nan"), 0.0]], list(good)]
    payload = json.dumps({"short": [*good, [0.02, 1.0, -0.02]], "cases": cases})
    assert math.isnan(json.loads(payload)["cases"][1][1][1])
    result = _verify_payload(payload, tmp_path)
    assert result["public_passed"] is False
    assert result["scientific_passed"] is False
    assert "error" in result["metrics"]


def test_evaluate_source_two_stage_docker(monkeypatch):
    from invariantlab.experiments import repair
    from invariantlab.schema import load_task_definition

    forged = json.dumps(
        {
            "public_passed": True,
            "scientific_passed": True,
            "public": {},
            "scientific": {},
            "metrics": {},
        }
    )
    failing = json.dumps(
        {
            "public_passed": True,
            "scientific_passed": False,
            "public": {},
            "scientific": {},
            "metrics": {"max_state_relative_error": 0.5},
        }
    )
    calls = []
    passed_thresholds = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        for i, arg in enumerate(argv):
            if arg == "-v" and argv[i + 1].endswith(":/verifier:ro"):
                host = argv[i + 1][: -len(":/verifier:ro")]
                passed_thresholds.append(
                    json.loads((Path(host) / "thresholds.json").read_text("utf-8"))
                )
        stdout = forged if len(calls) == 1 else failing
        return subprocess.CompletedProcess(argv, 0, stdout=stdout + "\n", stderr="")

    monkeypatch.setattr(repair.shutil, "which", lambda name: "docker")
    monkeypatch.setattr(repair.subprocess, "run", fake_run)

    task_dir = Path("tasks/oscillator")
    task = load_task_definition(task_dir)
    result = repair._evaluate_source(CORRECT_SOLVER, task_dir, task, "img")

    assert len(calls) == 2
    for argv in calls:
        joined = " ".join(argv)
        for flag in (
            "--network none",
            "--read-only",
            "--memory 256m",
            "--cpus 1",
            "--pids-limit 64",
            "--rm",
        ):
            assert flag in joined
        assert argv[0] == "docker" and argv[1] == "run"

    stage1_mounts = [calls[0][i + 1] for i, a in enumerate(calls[0]) if a == "-v"]
    assert any(m.endswith(":/work:ro") for m in stage1_mounts)
    assert any(m.endswith(":/output:rw") for m in stage1_mounts)

    stage2_mounts = [calls[1][i + 1] for i, a in enumerate(calls[1]) if a == "-v"]
    assert len(stage2_mounts) == 2
    assert any(m.endswith(":/verifier:ro") for m in stage2_mounts)
    assert any(m.endswith(":/data:ro") for m in stage2_mounts)
    assert not any("/work" in m or "candidate" in m for m in stage2_mounts)

    assert calls[1][-3:] == [
        "/verifier/verifier.py",
        "/data/trajectories.json",
        "/verifier/thresholds.json",
    ]
    assert passed_thresholds == [_thresholds()]

    assert result == json.loads(failing)
