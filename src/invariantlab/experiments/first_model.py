"""First executable model experiment: repair a scientifically wrong oscillator solver."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from invariantlab.config import load_experiment_config, load_model_config
from invariantlab.models import build_adapter

BROKEN_SOLVER = """def solve_oscillator_verlet(x0, v0, omega, dt, n_steps):
    trajectory = [(0.0, float(x0), float(v0))]
    x = float(x0)
    v = float(v0)
    omega2 = float(omega) * float(omega)
    t = 0.0

    for _ in range(int(n_steps)):
        a = omega2 * x
        v_half = v + 0.5 * dt * a
        x = x + dt * v_half
        a_new = omega2 * x
        v = v_half + 0.5 * dt * a_new
        t += dt
        trajectory.append((t, x, v))

    return trajectory
"""

PROMPT = """You are repairing a numerical solver for the one-dimensional harmonic oscillator

    d2x/dt2 = -omega^2 x

The intended method is velocity Verlet. The public tests only check interface behaviour,
the initial state, finiteness over a very short trajectory and a loose one-step sanity
check. They currently pass. Hidden scientific verification checks long-horizon agreement
with the analytical solution and conservation of total energy.

Return a complete replacement solver.py. Do not change the function signature and do not
use third-party packages.

Current solver.py:

```python
{source}
```
"""

EVALUATOR = r"""import importlib.util
import json
import math

spec = importlib.util.spec_from_file_location('candidate', '/work/solver.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
solve = module.solve_oscillator_verlet

def analytic(t, x0, v0, omega):
    return (
        x0 * math.cos(omega * t) + (v0 / omega) * math.sin(omega * t),
        -x0 * omega * math.sin(omega * t) + v0 * math.cos(omega * t),
    )

def energy(x, v, omega):
    return 0.5 * v * v + 0.5 * omega * omega * x * x

public = {}
scientific = {}

try:
    short = solve(1.0, 0.0, 1.0, 0.05, 2)
    public['shape'] = isinstance(short, list) and len(short) == 3 and all(len(row) == 3 for row in short)
    public['initial_state'] = public['shape'] and all(abs(a-b) < 1e-12 for a, b in zip(short[0], (0.0, 1.0, 0.0)))
    public['finite'] = public['shape'] and all(math.isfinite(float(v)) for row in short for v in row)
    public['one_step_sanity'] = public['shape'] and abs(float(short[1][1]) - math.cos(0.05)) < 0.01

    cases = [
        (1.0, 0.0, 1.0, 0.01, 800),
        (0.3, -0.4, 1.7, 0.005, 1200),
        (-0.8, 0.25, 0.7, 0.01, 900),
    ]
    max_state_error = 0.0
    max_energy_drift = 0.0
    for x0, v0, omega, dt, n_steps in cases:
        traj = solve(x0, v0, omega, dt, n_steps)
        t, x, v = map(float, traj[-1])
        x_ref, v_ref = analytic(t, x0, v0, omega)
        scale = max(1.0, abs(x_ref), abs(v_ref))
        max_state_error = max(max_state_error, math.hypot(x-x_ref, v-v_ref) / scale)
        e0 = energy(float(traj[0][1]), float(traj[0][2]), omega)
        drift = max(
            abs(energy(float(row[1]), float(row[2]), omega) - e0) / max(abs(e0), 1e-15)
            for row in traj
        )
        max_energy_drift = max(max_energy_drift, drift)

    scientific['analytical_state'] = max_state_error < 1e-3
    scientific['energy_invariant'] = max_energy_drift < 1e-3
    metrics = {
        'max_state_relative_error': max_state_error,
        'max_energy_relative_drift': max_energy_drift,
    }
except Exception as exc:
    public.setdefault('execution', False)
    metrics = {'error': repr(exc)}

result = {
    'public': public,
    'scientific': scientific,
    'public_passed': bool(public) and all(public.values()),
    'scientific_passed': bool(scientific) and all(scientific.values()),
    'metrics': metrics,
}
print(json.dumps(result, sort_keys=True))
"""


def _extract_python(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + "\n"
    if "def solve_oscillator_verlet" in text:
        return text.strip() + "\n"
    raise ValueError("Model response did not contain a Python solver")


def _evaluate_in_docker(source: str, image: str) -> dict[str, Any]:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required to execute model-generated code safely")

    with tempfile.TemporaryDirectory(prefix="invariantlab-") as tmp:
        work = Path(tmp)
        (work / "solver.py").write_text(source, encoding="utf-8")
        (work / "evaluate.py").write_text(EVALUATOR, encoding="utf-8")

        completed = subprocess.run(
            [
                "docker", "run", "--rm", "--network", "none",
                "--memory", "256m", "--cpus", "1", "--pids-limit", "64",
                "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
                "-v", f"{work.resolve()}:/work:ro", image, "python", "/work/evaluate.py",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Candidate evaluation failed: {detail}")
        return json.loads(completed.stdout.strip().splitlines()[-1])


def run_first_model_experiment(config_path: Path, output_dir: Path | None = None) -> Path:
    """Run the first repair experiment and write sample-level evidence."""

    experiment = load_experiment_config(config_path)
    model_config = load_model_config(Path(experiment.model))
    adapter = build_adapter(model_config)
    output = output_dir or Path("runs") / experiment.name
    output.mkdir(parents=True, exist_ok=True)

    image = experiment.container_image
    if "placeholder" in image:
        image = "python:3.12-slim"

    baseline = _evaluate_in_docker(BROKEN_SOLVER, image)
    prompt = PROMPT.format(source=BROKEN_SOLVER)

    started = time.perf_counter()
    response = adapter.generate(prompt)
    latency = time.perf_counter() - started
    repaired_source = _extract_python(response)
    repaired = _evaluate_in_docker(repaired_source, image)

    record = {
        "experiment": experiment.name,
        "model": adapter.model_id,
        "seed": experiment.seed,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "baseline": baseline,
        "repaired": repaired,
        "latency_seconds": latency,
        "response": response,
        "candidate_source": repaired_source,
    }

    (output / "events.jsonl").write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    (output / "baseline_solver.py").write_text(BROKEN_SOLVER, encoding="utf-8")
    (output / "candidate_solver.py").write_text(repaired_source, encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "experiment": experiment.name,
                "model": adapter.model_id,
                "baseline_public_passed": baseline["public_passed"],
                "baseline_scientific_passed": baseline["scientific_passed"],
                "repair_public_passed": repaired["public_passed"],
                "repair_scientific_passed": repaired["scientific_passed"],
                "verification_gap_before": int(baseline["public_passed"]) - int(baseline["scientific_passed"]),
                "successful_repair": repaired["scientific_passed"],
                "metrics": repaired["metrics"],
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return output
