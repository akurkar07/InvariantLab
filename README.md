# InvariantLab

Physics-grounded evaluation for AI-generated numerical software.

InvariantLab is a small benchmark for a specific question:

> Can a coding agent produce numerical physics software that is scientifically correct, not merely code that passes ordinary tests?

The current repository implements four deterministic benchmark tasks:

| Task | Numerical method | Independent scientific evidence |
|---|---|---|
| Harmonic oscillator | Velocity Verlet | closed-form trajectory and energy |
| Kepler two-body orbit | Velocity Verlet | analytical circular/elliptic cases and a high-accuracy DOP853 oracle |
| 1-D heat equation | FTCS | manufactured solution, boundary behaviour, stability and convergence |
| 1-D wave equation | Leapfrog | analytical standing wave, CFL behaviour and convergence |

## How the benchmark works

Each task is a self-contained package:

```text
tasks/<task>/
├── contract.yaml
├── specification.md
├── src/solver.py
└── tests/
    ├── public/
    └── scientific/
```

The candidate implementation is executed through the same documented boundary for every task:

```bash
python src/solver.py --input input.json --output result.npz
```

Public tests represent ordinary tests that an agent may see while implementing a task.

Scientific tests act as an independent evaluator. They execute the candidate as a subprocess, load only the declared NPZ output, and compare it against trusted analytical or high-accuracy evidence. Candidate task code cannot import the trusted verifier.

The useful distinction is therefore:

```text
public-test success
        versus
scientific correctness
```

## Model studies

The evaluation harness has been used for two controlled oscillator repair studies:

- **Study 1** ran 18 repair attempts (three mutation families, weak vs. hardened feedback, Cohere North Mini Code): 17/18 scientific passes, with one update-order repair that introduced a new stale-acceleration bug.
- **Study 2** ran 240 preregistered update-order repair attempts across weak, placebo, raw-metric and interpreted-metric feedback with Qwen2.5-Coder-7B-Instruct and DeepSeek-Coder-6.7B-Instruct: 240/240 scientific passes and no regressions, so this defect is saturated for these models.

Neither study is evidence of a general feedback-condition effect.

**Study 1 results:** [First Multi-Condition Study Results](docs/first-multi-condition-study-results.md)  
**Study 2 protocol:** [Update-order feedback replication](docs/update-order-feedback-replication.md)  
**Study 2 results:** [Two-Model Comparison](docs/study-two-model-comparison.md)

## Evaluation protocol

V1 measures **one-shot repair of a mutated solver**. The implemented runner is
`invariantlab.experiments.repair.run_repair_experiment` (`invariantlab run`). For each
experiment config it:

1. evaluates the configured mutant through the task verifier to obtain the baseline verdict
   and scientific metrics (the mutant is expected to pass the public checks and fail the
   scientific ones);
2. builds a seeded, balanced schedule of `(condition, trial)` cells, `n_attempts` per
   condition, shuffled with `seed` when `randomize_order` is true (`_build_schedule`);
3. for each cell, fills the task's `prompt_template` (for the oscillator,
   `tasks/oscillator/repair_prompt.txt`) with the mutant source and the condition-specific
   context from `_condition_context`, and makes **one** model call;
4. extracts a complete replacement `solver.py` from the response and evaluates it in a Docker
   sandbox (`--network none`, 256 MB memory, 1 CPU, 64 pids, read-only root, 60 s timeout):
   the candidate runner writes trajectories, then the trusted task verifier reads only that
   JSON and returns `public_passed`, `scientific_passed` and `metrics`;
5. appends one event per cell to `events.jsonl` and rewrites `study-summary.json`,
   `artifact-integrity.json` and `run-status.json`. Re-running the same command resumes from
   the completed cells; rate limits, connection/provider errors and sandbox failures pause
   the run instead of recording a model failure.

The model never inspects files, runs tests or calls tools: each cell is a single
prompt -> response repair. **A multi-turn, tool-using agent scaffold (file inspection,
visible-test execution, tool and token budgets) is not implemented in V1.**

### Conditions

`CONDITIONS = ("weak", "placebo", "metrics", "interpreted")` in `repair.py`. Every condition
sees the same prompt template (equation, intended method, a statement that the public tests
pass and that hidden scientific verification exists, and the mutant source); only the
inserted `condition_context` differs.

| Condition | Extra context shown to the model (`_condition_context`) | Group |
|---|---|---|
| `weak` | nothing | weak |
| `placebo` | non-diagnostic run metadata (evaluator completed, report serialisable) | weak (control for prompt length) |
| `metrics` | the baseline's raw scientific metrics, labelled from `task.yaml` `feedback_metrics` (oscillator: max state relative error, max energy relative drift) | hardened |
| `interpreted` | the `metrics` lines plus `task.yaml` `interpreted_feedback` (threshold and meaning of each metric) | hardened |

"Weak" means the `weak` condition. **"Hardened" means the verifier-feedback conditions,
`metrics` and `interpreted`.** Held-out verifier cases are never shown in any condition.

Study 1 predates the four-condition runner and used two labels, *Weak* and *Hardened*. Its
*Hardened* prompt exposed the baseline's scientific error metrics
([Study 1 results](docs/first-multi-condition-study-results.md), section 6), i.e. the
verifier-feedback group; it did not separate raw from interpreted metrics, and its prompt
and config are not committed, so it is not identical to either `metrics` or `interpreted`.
Study 2 split that feedback into `metrics` and `interpreted` and added `placebo`.

### Public and scientific checks used by the study runner

The public and scientific verdicts in study results come from the task verifier named in
`task.yaml` (`tasks/oscillator/verifier.py`), not from the task package's `tests/public` and
`tests/scientific` suites or `contract.yaml`:

- public (weak) checks: four inline checks on a 3-row trajectory (`shape`, `initial_state`,
  `finite`, `one_step_sanity`);
- scientific checks: `analytical_state` (max state relative error `< 1e-3`) and
  `energy_invariant` (max energy relative drift `< 1e-3`) over three held-out cases; the
  same `1e-3` thresholds appear in `task.yaml` `feedback_metrics`.

These differ from the contract tolerances in `tasks/oscillator/contract.yaml`
(`state_relative_l2: 1e-5`, `energy_relative_drift: 1e-6`), which the task-package suites
use. Single-sourcing the study thresholds against the contract is tracked in
[#109](https://github.com/akurkar07/InvariantLab/issues/109), and evaluating candidates
through the M2 task package and the M3 verification stack in
[#120](https://github.com/akurkar07/InvariantLab/issues/120) and
[#119](https://github.com/akurkar07/InvariantLab/issues/119).

## Metrics

Each metric is tagged **Implemented** (with the code that computes it) or **Planned**.
Per-condition summary fields are defined in
[Experiment Authoring: Summary fields](docs/experiment-authoring.md#summary-fields).

| Metric | Status | Where |
|---|---|---|
| Public pass rate `P_public = N_public_pass / N_attempted` | Implemented | `invariantlab.metrics.pass_rate`, reported as `by_condition.<c>.public_pass_rate` by `repair._summary` |
| Scientific pass rate `P_science = N_scientific_pass / N_attempted` (primary endpoint) | Implemented | `invariantlab.metrics.pass_rate`, `by_condition.<c>.scientific_pass_rate` |
| Verification gap `G = P_public - P_science` | Implemented | `invariantlab.metrics.verification_gap` (per condition); `repair._baseline_verification_gap` for the unrepaired mutant |
| Wilson 95% intervals for both pass rates | Implemented | `invariantlab.metrics.wilson_interval` |
| Difference from the `weak` condition | Implemented | `pass_rate_difference_vs_weak` in `repair._summary` |
| Scientific regressions and severity ratios versus the mutant | Implemented | `repair._severity_ratios` (`state_error_ratio`, `energy_drift_ratio`, `worst_scientific_ratio`); `scientific_regressions`, median/max ratio in `repair._summary` |
| Repair success per defect family | Implemented per run (one mutation per experiment config) | `repair._summary`; cross-family tables are Planned ([#104](https://github.com/akurkar07/InvariantLab/issues/104)) |
| Repair success per task family | Planned | runner evaluates the oscillator only ([#120](https://github.com/akurkar07/InvariantLab/issues/120)) |
| Diagnostic localisation (causal fix vs. compensation vs. test targeting) | Planned | failed candidates are only flagged `needs_manual_failure_review` |
| Numerical quality: max state relative error, max energy relative drift | Implemented | `tasks/oscillator/verifier.py` (`metrics` in each verdict) |
| Numerical quality: relative L1/L2/Linf error, RMS drift, observed convergence order, phase error, boundary residual, stability failures, non-finite-state count | Planned | M3 verification stack ([#119](https://github.com/akurkar07/InvariantLab/issues/119)) |
| Efficiency: wall-clock latency, input/output tokens | Implemented (recorded per event, not aggregated) | `latency_seconds` and `usage` in each `events.jsonl` record, from `run_repair_experiment` and the adapters' `ModelResponse` |
| Efficiency: model-reported cost, successful repairs per compute budget | Planned | not recorded |
| Efficiency: tool calls, test executions | Planned | not applicable to one-shot repair; needs an agent scaffold |
| Paired analyses across shared tasks; stratification by task, model and pooled headlines | Planned | summaries are per run (one model, one mutation) ([#104](https://github.com/akurkar07/InvariantLab/issues/104)) |

## Local-first model execution

InvariantLab is designed to run against local models first. The bundled
`configs/models/default.yaml` targets Qwen2.5-Coder-7B-Instruct through Ollama, and a vLLM
preset is included for an OpenAI-compatible local server.

```bash
# One-time model setup
ollama pull qwen2.5-coder:7b-instruct

# Verify the default local model
uv run invariantlab model-check \
  --model configs/models/default.yaml

# Run the first live repair experiment locally
uv run invariantlab run \
  --experiment configs/experiments/first-model-oscillator-live.yaml
```

Longer studies support safe batching and resume from committed run evidence. For example:

```bash
uv run invariantlab run \
  --experiment configs/experiments/update-order-feedback-replication-ollama.yaml \
  --max-new-attempts 20
```

Running the same command again continues from the existing `events.jsonl`.

### Reproducible outputs

Each repair run directory starts with `manifest.json` (code revision, task/mutation/verifier
hashes, model config, seed and the resolved container image digest) and, once complete, ends
with `checksums.sha256` in `sha256sum` format. Resuming with different provenance stops with
`invalid_artifact`. Placeholder images are rejected; pin a digest in `container_image` or pass
`--image` (recorded in the manifest). See
[run provenance](docs/local-models.md#run-provenance-manifest-image-digest-and-checksums).

Hosted APIs are optional rather than the default. To use one, configure the generic
`openai_compatible` adapter with an explicit `base_url` and API-key environment variable;
see `configs/models/api-example.yaml`.

### Model backends

Implemented adapter IDs are `reference_stub`, `replay`, `ollama` and `openai_compatible`. vLLM, OpenRouter and
other hosted APIs use `openai_compatible`; Anthropic and Hugging Face `transformers` are not
native adapters. Use an OpenAI-compatible endpoint for Anthropic models, or serve Hugging Face
weights with vLLM or TGI and point `openai_compatible` at that server.

Study 1 used Cohere North Mini Code via OpenRouter (`configs/models/cohere-north-mini-code-free.yaml`).
Study 2 used Qwen2.5-Coder-7B-Instruct and DeepSeek-Coder-6.7B-Instruct via local Ollama
(`configs/models/ollama-qwen2.5-coder-7b.yaml`,
`configs/models/ollama-deepseek-coder-6.7b.yaml`). The default config uses
Qwen2.5-Coder-7B-Instruct through Ollama.

See [Model adapters](docs/model-adapters.md) for configuration and backend details, and
[model execution](docs/local-models.md) for local server setup.

## Command-line interface

Install with `uv sync --extra dev`, then run every command as `uv run invariantlab ...`.
On Windows, set `PYTHONIOENCODING=utf-8` first; otherwise the CLI can crash while printing
its status symbols.

| Command | Purpose |
|---|---|
| `version` | Print the installed InvariantLab version. |
| `validate-task --task-dir <dir>` | Validate one task contract and the files it declares. |
| `model-check --model <config>` | Send one short prompt to the configured model and print the reply. |
| `run --experiment <config> [--output <dir>] [--dry-run] [--max-new-attempts N]` | Run an experiment with the `repair` runner. Output goes to `runs/<experiment name>/` unless `--output` is given. |
| `audit-run --experiment <config> --run-dir <dir> [--write-canonical]` | Audit a repair run's raw `events.jsonl` against the configured cell schedule. |

```bash
# Validate one task package
uv run invariantlab validate-task --task-dir tasks/oscillator

# Check a model config (replay needs no server; default.yaml needs a running Ollama)
uv run invariantlab model-check --model configs/models/reference-stub-oscillator.yaml
uv run invariantlab model-check --model configs/models/default.yaml

# Validate an experiment config without calling a model or Docker
uv run invariantlab run --experiment configs/experiments/first-model-oscillator.yaml --dry-run

# Deterministic replay smoke run (needs Docker, no model server)
uv run invariantlab run --experiment configs/experiments/first-model-oscillator.yaml

# Resumable repair study in batches of 20 cells (needs Docker and Ollama); rerun to continue
uv run invariantlab run \
  --experiment configs/experiments/update-order-feedback-replication-ollama.yaml \
  --max-new-attempts 20

# Audit a repair run and write the canonical projection
uv run invariantlab audit-run \
  --experiment configs/experiments/update-order-feedback-replication-ollama.yaml \
  --run-dir runs/update-order-feedback-replication-ollama-qwen25-7b \
  --write-canonical
```

`--max-new-attempts` is accepted only by the resumable `repair` runner. Candidates run in the experiment's `container_image` (`python:3.12-slim`);
if Docker Hub rate-limits it, `mirror.gcr.io/library/python:3.12-slim` is equivalent.

### Planned CLI (not implemented)

These interfaces are planned and do not exist yet; the CLI rejects them as unknown commands
or options:

- `invariantlab tasks validate --suite <suite>`: validate a whole task suite. Today use
  `validate-task` per task or `python scripts/validate_task.py --task-dir tasks/`.
- `invariantlab run --suite <suite> --condition <condition>`: suite-by-condition runs. Today
  conditions are set in the experiment config's `conditions` list.
- `invariantlab report`: report generation from run directories.
- `invariantlab export hf`: Hugging Face dataset export.

## Reproducible outputs

`invariantlab run` writes to `runs/<experiment name>/`, or to the `--output` directory.

`repair` runner (experiment configs with `runner: repair`, e.g. `configs/experiments/first-model-oscillator*.yaml` and the Study 2 configs):

| File | Meaning |
|---|---|
| `events.jsonl` | Append-only raw records, one per attempted cell; reruns resume from it. |
| `baseline_solver.py` | The mutated solver from the configured mutation. |
| `study-summary.json` | Per-condition public and scientific pass rates with Wilson 95% intervals, verification gap and regressions; rewritten after every attempt. |
| `run-status.json` | Run state (`complete`, `batch_complete`, `paused_*`, ...) with completed and target cell counts and the stop reason. |
| `artifact-integrity.json` | Audit of the raw records against the scheduled cells (duplicates, out-of-schedule, metadata mismatches, malformed records). Also written by `audit-run`. |
| `events.canonical.jsonl` | Written only by `audit-run --write-canonical`: the first valid record per scheduled cell. Raw `events.jsonl` is never modified. |
| `manifest.json` | Run provenance written at the start of a run: code revision, task/mutation/verifier hashes, model config, seed and resolved container image digest. |
| `checksums.sha256` | `sha256sum`-format checksums over every run file; written once the run completes. |

Planned, not written yet: `environment.json`, per-suite test-result files and a
`reports/<id>/` tree.

## Repository layout

Generated from `git ls-files`; every path below exists on `main`.

```text
src/invariantlab/
├── cli.py                       # validate-task, model-check, run, audit-run
├── config.py                    # model and experiment config loading
├── schema.py                    # task/output contract and experiment models
├── experiments/                 # repair and feedback-replication runners
├── models/
│   └── adapter.py               # reference_stub, replay, ollama and openai_compatible adapters
├── tasks/
│   └── validation.py            # package and path validation
└── verification/
    ├── analytical.py            # exact solutions and physical quantities
    ├── kepler_oracle.py         # independent DOP853 Kepler oracle
    └── solvers.py               # trusted numerical references used by tests

.github/workflows/          # CI: tests.yml (lint, typecheck, tests), task-validation.yml
configs/
├── experiments/            # first-model smoke/live, Study 2 repair configs
├── models/                 # replay, Ollama, vLLM, Cohere and API-example model configs
└── task-suites/            # v1-smoke.yaml: the four task dirs (no runner reads suites yet)
docs/                       # task/experiment authoring, local models, Study 1/2 protocol and results
scripts/
└── validate_task.py        # validates the four committed V1 task packages
src/invariantlab/
├── cli.py                  # version, validate-task, model-check, run, audit-run
├── config.py               # experiment, task-suite and model config loading
├── schema.py               # task contract, task/mutation definitions and result models
├── metrics.py              # pass rates, Wilson intervals and verification gap
├── experiments/            # repair.py (Docker runner and audit), feedback_replication.py (Study 2 wrapper)
├── models/
│   └── adapter.py          # reference_stub, replay, ollama and openai_compatible adapters
├── tasks/
│   └── validation.py       # package and path validation
└── verification/           # analytical.py, kepler_oracle.py, solvers.py: trusted references
tasks/
├── oscillator/             # task package plus repair assets: task.yaml, candidate_runner.py,
│                           #   verifier.py, repair_prompt.txt, mutations/update-order/
├── kepler/                 # contract.yaml, specification.md, src/solver.py, tests/
├── heat1d/                 # same layout as kepler/
└── wave1d/                 # same layout as kepler/
tests/
├── unit/                   # oracles, solvers, convergence, schema, adapters, runners, metrics, study reports
└── acceptance/             # trust-boundary and repository acceptance checks
```

Placeholders: `tasks/*/.gitkeep` are empty markers. There are no stub
Python packages: the former `mutations/`, `reporting/` and `dashboard/` packages, the
`verification/{invariants,convergence,metamorphic,robustness}.py` stubs and the empty
`tests/property/` and `tests/integration/` directories were removed on `develop` (#44) and are
gone from `main` since the develop/main merge.

## Scientific evidence

The benchmark currently checks several kinds of failure that ordinary unit tests can miss.

### Independent solutions

Where a closed form exists, numerical output is compared with the exact solution. Eccentric Kepler trajectories use a separately implemented SciPy DOP853 integration with substantially tighter tolerances than the candidate method.

### Physical behaviour

Task-specific tests check properties such as:

- oscillator energy behaviour
- Kepler energy and angular momentum drift
- zero Dirichlet boundaries
- heat-equation diffusion decay
- wave-equation CFL stability and phase accuracy
- finite float64 output

### Empirical convergence

Reference methods are also tested over controlled refinement sequences. The repository measures observed order rather than merely checking that an error decreases.

Current evidence includes:

- second-order oscillator Velocity Verlet
- second-order circular and eccentric Kepler Velocity Verlet
- first-order FTCS temporal convergence
- second-order FTCS coupled spatial convergence
- second-order Crank-Nicolson temporal and spatial convergence
- second-order leapfrog convergence under fixed CFL

## Trust boundary

Agent-facing task code is deliberately separate from trusted verification code.

Scientific tests consume serialized task output rather than calling candidate Python functions directly. Acceptance tests enforce that hidden scientific tests do not import candidate solver modules and that agent-facing code does not import trusted verification modules.

This separation is intentional. The benchmark should not certify an implementation using the same numerical update code that it is evaluating.

## Contracts

Each `contract.yaml` declares the task identity, executable paths, numerical settings and exact NPZ output.

Example:

```yaml
id: oscillator_verlet
family: oscillator
language: python
entrypoint: src/solver.py
public_tests: tests/public
scientific_tests: tests/scientific
output:
  path: result.npz
  arrays:
    - name: time
      shape: [null]
      dtype: float64
    - name: state
      shape: [null, 2]
      dtype: float64
```

The validator fails closed on malformed contracts, missing required V1 packages and unsafe declared paths.

See [docs/task-authoring.md](docs/task-authoring.md) for the task protocol.

## Running the current benchmark checks

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Validate the four task packages:

```bash
python scripts/validate_task.py --task-dir tasks/
```

Run top-level tests:

```bash
pytest tests/
```

Run a task directly:

```bash
cd tasks/oscillator
python src/solver.py --input input.json --output result.npz
```

Task-local public and scientific suites can be run with pytest from the repository root.

## V1 acceptance criteria

V1 ships only when every acceptance criterion below is proved by at least one automated test; the criteria-to-tests map and the fail-closed release-gate checker live in [docs/v1-acceptance.md](docs/v1-acceptance.md) (`uv run python scripts/check_v1_acceptance.py --report v1.json`).

- **V1-AC1** all reference implementations pass every scientific gate;
- **V1-AC2** every controlled mutant passes its designated weak profile and fails its expected scientific gate;
- **V1-AC3** independent oracle and agent-facing code paths share no numerical update implementation;
- **V1-AC4** repeated replay produces identical evaluator outcomes;
- **V1-AC5** report totals equal the number of enumerated sample records;
- **V1-AC6** result tables can be regenerated without API access;
- **V1-AC7** CI exercises task validation, a complete smoke run and report reconstruction;
- **V1-AC8** the public dataset contains task metadata, trajectories, patches, measurements and provenance without hidden credentials.

## Current scope

InvariantLab is the benchmark core **plus** the evaluation harness used to run model studies against it.

It currently provides:

- four task packages (`oscillator`, `kepler`, `heat1d`, `wave1d`) with public and scientific suites
- trusted reference solvers and analytical / high-accuracy oracles
- task contract and artifact validation (`invariantlab validate-task`, `scripts/validate_task.py`)
- the oscillator repair experiment runner, which executes candidate repairs in Docker (`invariantlab run`)
- `reference_stub`, `replay`, `ollama` and `openai_compatible` model adapters (`invariantlab model-check`)
- raw run-evidence auditing (`invariantlab audit-run`)

It does **not** provide yet:

- verification layers 3-6 (invariants, convergence, metamorphic, robustness) as reusable gates
- a mutation registry
- reporting
- a dashboard
- Hugging Face dataset export
