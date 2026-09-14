# First model experiment

The first executable InvariantLab experiment tests the core benchmark claim on one controlled harmonic-oscillator defect.

The supplied velocity-Verlet implementation is scientifically wrong but passes deliberately weak public checks. A coding model receives the equation, method requirement, weak-test status and source code, then returns a replacement `solver.py`. The candidate is executed in a network-disabled, resource-limited Docker container. InvariantLab independently checks long-horizon analytical-state error and energy conservation.

## Deterministic smoke run

```bash
uv sync
uv run invariantlab run --experiment configs/experiments/first-model-oscillator.yaml
```

The replay adapter performs the known repair and proves that the complete experiment pipeline works without an API key.

## First live model run

Set `OPENROUTER_API_KEY` in your shell, then run:

```bash
export OPENROUTER_API_KEY="..."
uv run invariantlab run --experiment configs/experiments/first-model-oscillator-live.yaml
```

Do not commit API keys.

## Evidence

Each run writes `events.jsonl`, `baseline_solver.py`, `candidate_solver.py`, and `summary.json`.

A useful first empirical result requires the baseline to pass the weak checks and fail scientific verification. A successful repair must pass both scientific gates. This experiment is intentionally small: it validates the measurement loop before scaling to multiple mutations, verifier conditions, models, and repeated trials.
