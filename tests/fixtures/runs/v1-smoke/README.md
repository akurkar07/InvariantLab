# v1-smoke run fixture

A complete four-record repair run (oscillator `update-order` mutation; conditions `weak`,
`metrics`; 2 trials; seed 1729; no order randomisation) written by the real
`run_repair_experiment` with the M5 `replay` adapter. Model responses come from
`replay-source.jsonl` (a copy of `tests/fixtures/replay/oscillator-update-order/events.jsonl`);
candidates are scored by the same deterministic content-based evaluator used in
`tests/unit/test_repair_experiment.py`, so the run needs no API access and no candidate
containers. The image digest in `manifest.json` was resolved from Docker at generation time.

- Inputs: `experiment.yaml`, `model.yaml`, `replay-source.jsonl`, `regenerate.py`.
- Run artifacts: `events.jsonl`, `events.canonical.jsonl`, `manifest.json`,
  `study-summary.json`, `artifact-integrity.json`, `run-status.json`, `checksums.sha256`
  (the runner's `baseline_solver.py` copy is omitted; the mutation source is pinned by
  `manifest.json` `artifact_sha256.mutation_source`).
- Expected report (`invariantlab report` output): `expected/summary.json`,
  `expected/by_condition.csv`, `expected/samples.csv`.

Used by `tests/acceptance/test_v1_report_reconstruction.py` (V1-AC5, V1-AC6). Regenerate from the
repository root with `uv run python tests/fixtures/runs/v1-smoke/regenerate.py`; rebuild only the
expected tables with
`uv run invariantlab report --experiment tests/fixtures/runs/v1-smoke/experiment.yaml --run-dir tests/fixtures/runs/v1-smoke --output tests/fixtures/runs/v1-smoke/expected`.
