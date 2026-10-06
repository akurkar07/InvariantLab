# First model experiment

The original Study 1 experiment evaluated a controlled harmonic-oscillator sign defect.
The current `first-model-oscillator` configurations express that experiment with the
generic `repair` runner: they bind the oscillator task to the `sign-error` mutation,
schedule one `weak` condition with `n_attempts: 1`, and use the task's `repair_prompt.txt`.
The replay config is deterministic; the live config uses the default Ollama model.

## Run

```bash
uv run invariantlab run --experiment configs/experiments/first-model-oscillator.yaml
```

The reference-stub adapter returns a fixed, known-correct solver, proving the pipeline works without an API key; it is not a model.

## First live model run

The live example is local-first and uses the default Ollama model configuration.
For a local live model, pull the configured Ollama model and run:

```bash
ollama pull qwen2.5-coder:7b-instruct
uv run invariantlab model-check --model configs/models/default.yaml
uv run invariantlab run --experiment configs/experiments/first-model-oscillator-live.yaml
```

The repair runner writes `events.jsonl`, `study-summary.json`, `baseline_solver.py`,
`run-status.json`, and `artifact-integrity.json`. Candidate source and per-attempt
verifier results are recorded in `events.jsonl`; the summary includes baseline and
per-condition scientific pass rates.

## Study 1 mapping

The original run used `runner: first_model`, an inline `BROKEN_SOLVER`, its own prompt,
and an inline verifier copy. The defect is now
`tasks/oscillator/mutations/sign-error/solver.py`; the prompt comes from the task's
`repair_prompt.txt` under the `weak` condition. The recorded Study 1 results were not
re-run as part of this migration.

See [model execution](local-models.md) for Ollama, vLLM and optional API endpoint setup.
