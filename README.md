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
| `run --experiment <config> [--output <dir>] [--dry-run] [--max-new-attempts N]` | Run an experiment with the `first_model` or `repair` runner. Output goes to `runs/<experiment name>/` unless `--output` is given. |
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

`--max-new-attempts` is accepted only by the resumable `repair` runner; the `first_model`
runner rejects it. Candidates run in the experiment's `container_image` (`python:3.12-slim`);
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

`first_model` runner (`configs/experiments/first-model-oscillator*.yaml`):

| File | Meaning |
|---|---|
| `events.jsonl` | One JSON record: prompt hash, raw model response, extracted candidate source, latency, and public/scientific results before and after repair. |
| `baseline_solver.py` | The broken oscillator solver the model was asked to repair. |
| `candidate_solver.py` | The Python solver extracted from the model response. |
| `summary.json` | Baseline and repaired public/scientific pass flags, verification gap before repair, and the repaired metrics. |

`repair` runner (experiment configs with `runner: repair`, e.g. the Study 2 configs):

| File | Meaning |
|---|---|
| `events.jsonl` | Append-only raw records, one per attempted cell; reruns resume from it. |
| `baseline_solver.py` | The mutated solver from the configured mutation. |
| `study-summary.json` | Per-condition public and scientific pass rates with Wilson 95% intervals, verification gap and regressions; rewritten after every attempt. |
| `run-status.json` | Run state (`complete`, `batch_complete`, `paused_*`, ...) with completed and target cell counts and the stop reason. |
| `artifact-integrity.json` | Audit of the raw records against the scheduled cells (duplicates, out-of-schedule, metadata mismatches, malformed records). Also written by `audit-run`. |
| `events.canonical.jsonl` | Written only by `audit-run --write-canonical`: the first valid record per scheduled cell. Raw `events.jsonl` is never modified. |

Planned, not written yet: a run `manifest.json` with pinned image digest and provenance,
`environment.json`, `checksums.sha256`, per-suite test-result files and a `reports/<id>/`
tree (the manifest is tracked in #80).

## Repository layout

Generated from `git ls-files`; every path below exists on `main`.

```text
src/invariantlab/
├── cli.py                       # validate-task, model-check, run, audit-run
├── config.py                    # model and experiment config loading
├── schema.py                    # task/output contract and experiment models
├── experiments/                 # repair, first-model and feedback-replication runners
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
├── experiments/            # first-model smoke/live, Study 2 repair configs, v1-smoke (placeholder)
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
├── experiments/            # first_model.py, repair.py (Docker runner and audit), feedback_replication.py (Study 2 wrapper)
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

Placeholders: `configs/experiments/v1-smoke.yaml` has a placeholder image digest (the runner
falls back to `python:3.12-slim`), and `tasks/*/.gitkeep` are empty markers. There are no stub
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
