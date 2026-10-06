"""Regenerate replay responses for package-mutant repair smoke cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from invariantlab.config import load_experiment_config
from invariantlab.experiments.repair import _render_prompt, _resolve_assets

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "configs/experiments/repair-package-replay.yaml"
PACKAGE_REPLAY_CASES = [
    ("oscillator", "non-conservative-damping"),
    ("wave1d", "sign-error-startup"),
]


def main() -> None:
    config = load_experiment_config(CONFIG_PATH)
    events = []
    for task_name, mutation_id in PACKAGE_REPLAY_CASES:
        experiment = config.model_copy(
            update={
                "task": f"tasks/{task_name}",
                "mutation": f"tasks/{task_name}/mutations/{mutation_id}",
            }
        )
        assets = _resolve_assets(experiment)
        prompt = _render_prompt(assets, "")
        source = (
            (REPO_ROOT / f"tasks/{task_name}/src/solver.py")
            .read_text(encoding="utf-8")
            .replace("\r\n", "\n")
        )
        events.append(
            {
                "model": "replay/repair-package",
                "condition": "weak",
                "trial": 1,
                "schedule_index": 1,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "response": f"```python\n{source}```\n",
            }
        )

    events_path = REPO_ROOT / "tests/fixtures/replay/repair-package/events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
