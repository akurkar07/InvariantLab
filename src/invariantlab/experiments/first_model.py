"""First executable model experiment: repair a scientifically wrong oscillator solver."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from invariantlab.config import load_experiment_config, load_model_config
from invariantlab.experiments.repair import _evaluate_source
from invariantlab.models import build_adapter
from invariantlab.schema import load_task_definition

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

def _extract_python(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + "\n"
    if "def solve_oscillator_verlet" in text:
        return text.strip() + "\n"
    raise ValueError("Model response did not contain a Python solver")


def _evaluate_in_docker(source: str, image: str) -> dict[str, Any]:
    task_dir = Path("tasks/oscillator")
    return _evaluate_source(source, task_dir, load_task_definition(task_dir), image)


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

    (output / "events.jsonl").write_text(
        json.dumps(record, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
                "verification_gap_before": (
                    int(baseline["public_passed"])
                    - int(baseline["scientific_passed"])
                ),
                "successful_repair": repaired["scientific_passed"],
                "metrics": repaired["metrics"],
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return output
