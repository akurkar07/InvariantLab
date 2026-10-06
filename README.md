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

## Status

What is implemented on `main` today. The [V1 design target](docs/methodology.md) describes the full design.

| Component | Status | Code path | Tracking |
|---|---|---|---|
| Task packages (4: oscillator, kepler, heat1d, wave1d) | Implemented | `tasks/` | [M2](https://github.com/akurkar07/InvariantLab/milestone/1) |
| Reference solvers and oracles | Implemented | `src/invariantlab/verification/` | [M2](https://github.com/akurkar07/InvariantLab/milestone/1) |
| Verification layer 0: execution and archive schema | Implemented | `src/invariantlab/verification/execution.py` | [#110](https://github.com/akurkar07/InvariantLab/issues/110) |
| Verification layer 2: analytical and high-accuracy oracles | Implemented (`check_oracle` gate) | `src/invariantlab/verification/oracles.py` | [#112](https://github.com/akurkar07/InvariantLab/issues/112) |
| Verification layer 3: invariant checks | Implemented (`check_invariants` gate) | `src/invariantlab/verification/invariants.py` | [#113](https://github.com/akurkar07/InvariantLab/issues/113) |
| Verification layer 4: convergence studies | Implemented (`check_convergence` gate) | `src/invariantlab/verification/convergence.py` | [#114](https://github.com/akurkar07/InvariantLab/issues/114) |
| Verification layer 5: metamorphic tests | Implemented (`check_metamorphic` gate) | `src/invariantlab/verification/metamorphic.py` | [#115](https://github.com/akurkar07/InvariantLab/issues/115) |
| Verification layer 6: held-out robustness cases | Implemented (`check_robustness` gate) | `src/invariantlab/verification/robustness.py` | [#116](https://github.com/akurkar07/InvariantLab/issues/116) |
| Defect injection / mutants | Partial (registry; three validated package mutants, two legacy oscillator study mutants) | `src/invariantlab/mutations/`, `tasks/*/mutations/` | [M4](https://github.com/akurkar07/InvariantLab/milestone/3) |
| Model adapters | Implemented | `src/invariantlab/models/adapter.py` | [M5](https://github.com/akurkar07/InvariantLab/milestone/4) |
| Repair runner + `audit-run` | Implemented | `src/invariantlab/experiments/repair.py` | [M5](https://github.com/akurkar07/InvariantLab/milestone/4) |
| Reporting / dashboard / HF export | Partial (`report`, static `report.html`, local HF export via `export-hf`) | `src/invariantlab/reporting/` | [#106](https://github.com/akurkar07/InvariantLab/issues/106), [#107](https://github.com/akurkar07/InvariantLab/issues/107) |
| Release | Planned | - | [M7](https://github.com/akurkar07/InvariantLab/milestone/6) |

## Installation

**V1 runs from a source checkout**: `git clone https://github.com/akurkar07/InvariantLab.git`,
then `uv sync --extra dev` inside it. The wheel provides the library and the `invariantlab`
CLI, but benchmark tasks (`tasks/`) and configs (`configs/`) are read from the checkout, so run
the CLI from the repository root. Packaging tasks/configs as package data is deferred to
post-V1; see [Releasing](docs/releasing.md) and [Known limitations](docs/limitations.md).

## Quickstart

Prerequisites:

- Python >= 3.10 (`requires-python` in `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/)
- Docker, for experiment runs only: `invariantlab run` evaluates every candidate in a container

```bash
uv sync --extra dev

# Top-level unit and acceptance tests
uv run pytest tests -q

# Run one task on its committed example input, then its public and scientific suites
cd tasks/oscillator
uv run python src/solver.py --input examples/input.json --output result.npz
uv run pytest tests
cd ../..

# All four task suites (each runs from its own task directory)
make test-tasks

# Deterministic smoke run: no API key or model server, needs Docker
uv run invariantlab run --experiment configs/experiments/first-model-oscillator.yaml
```

Expected outcome of the smoke run: the baseline `sign-error` mutant passes the public checks
and fails the scientific checks; the single repair, a fixed known-correct solver returned by
the `reference_stub` adapter, passes both. Results go to `runs/first-model-oscillator/`. If
Docker Hub rate-limits `python:3.12-slim`, add `--image mirror.gcr.io/library/python:3.12-slim`.

CI's "Reproduce Smoke Run" workflow runs `uv run python scripts/reproduce_report.py --smoke`
(`make reproduce`), which runs `configs/experiments/replay-smoke.yaml` in Docker, rebuilds the
report and fails on any drift from `tests/fixtures/expected/replay-smoke-summary.json`.

To run against a local model with Ollama, see [Model execution](docs/local-models.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full local verification commands (including the
per-task suite loop), branch policy and required CI checks.

## Current scope

InvariantLab is the benchmark core **plus** the evaluation harness used to run model studies against it.

It currently provides:

- four task packages (`oscillator`, `kepler`, `heat1d`, `wave1d`) with public and scientific suites
- trusted reference solvers and analytical / high-accuracy oracles
- reusable verification gates in `invariantlab.verification`: `run_task` (layer 0),
  `check_oracle`, `check_invariants`, `check_convergence`, `check_metamorphic` and
  `check_robustness` (layers 2-6)
- task contract and artifact validation (`invariantlab validate-task`, `scripts/validate_task.py`)
- the oscillator repair experiment runner, which executes candidate repairs in Docker (`invariantlab run`)
- `reference_stub`, `replay`, `ollama` and `openai_compatible` model adapters (`invariantlab model-check`)
- raw run-evidence auditing (`invariantlab audit-run`)
- rebuilding a repair run's summary and CSV tables from `events.jsonl` alone, with an optional
  static `report.html` (`invariantlab report --html`)
- local Hugging Face dataset export with per-record provenance (`invariantlab export-hf`)

It does **not** provide yet:

- controlled mutants for kepler and heat1d (only the oscillator and wave1d have package mutants)
- plots, interactive filtering or a served dashboard (the static `report.html` is the V1 dashboard)

## Empirical results

The evaluation harness has been used for two controlled oscillator repair studies. Both
measured **repair success under feedback conditions on injected oscillator defects**; neither
measured the verification gap of agent-written code.

- **Study 1** ran 18 repair attempts (three mutation families, weak vs. hardened feedback, Cohere North Mini Code): 17/18 scientific passes, with one update-order repair that introduced a new stale-acceleration bug.
- **Study 2** ran 240 preregistered update-order repair attempts across weak, placebo, raw-metric and interpreted-metric feedback with Qwen2.5-Coder-7B-Instruct and DeepSeek-Coder-6.7B-Instruct: 240/240 scientific passes and no regressions, so this defect is saturated for these models.

Neither study is evidence of a general feedback-condition effect.

**Study 1 results:** [First Multi-Condition Study Results](docs/first-multi-condition-study-results.md)  
**Study 2 protocol:** [Update-order feedback replication](docs/update-order-feedback-replication.md)  
**Study 2 results:** [Two-Model Comparison](docs/study-two-model-comparison.md)  
**Research track:** [Studies 1-3 and the verification gap](docs/research-track.md)

## Design and methodology

The V1 design and reference material lives in [docs/methodology.md](docs/methodology.md):

- [How the benchmark works](docs/methodology.md#how-the-benchmark-works): self-contained task packages behind one subprocess/NPZ boundary; public tests versus independent scientific tests.
- [Evaluation protocol](docs/methodology.md#evaluation-protocol): one-shot repair of a mutated solver under the `weak`, `placebo`, `metrics` and `interpreted` conditions.
- [Metrics](docs/methodology.md#metrics): public and scientific pass rates, verification gap, Wilson intervals and regressions, each tagged Implemented or Planned.
- [Scientific evidence](docs/methodology.md#scientific-evidence): independent solutions, physical-behaviour checks and empirical convergence order.
- [Trust boundary](docs/methodology.md#trust-boundary): agent-facing task code is kept separate from trusted verification code.
- [Contracts](docs/methodology.md#contracts): what each `contract.yaml` declares; see also [Task Authoring](docs/task-authoring.md).
- [Reproducible outputs](docs/methodology.md#reproducible-outputs): files in a repair run directory, run manifest and checksums.
- [Reporting a run](docs/methodology.md#reporting-a-run): `invariantlab report` rebuilds summary and CSV tables (and, with `--html`, a static `report.html`) from `events.jsonl` without a model or Docker.
- [Exporting a dataset](docs/methodology.md#exporting-a-dataset): `invariantlab export-hf` writes a local Hugging Face-loadable dataset with per-record provenance after a credential scan.

## Local-first model execution

InvariantLab is designed to run against local models first. The bundled
`configs/models/default.yaml` targets Qwen2.5-Coder-7B-Instruct through Ollama, and a vLLM
preset is included for an OpenAI-compatible local server.

Hosted APIs are optional rather than the default. To use one, configure the generic
`openai_compatible` adapter with an explicit `base_url` and API-key environment variable;
see `configs/models/api-example.yaml`.

See [model execution](docs/local-models.md) for Ollama and vLLM commands, batching and resume.

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

## Repository layout

Generated from `git ls-files`; every path below exists on `main`.

```text
src/invariantlab/
├── cli.py                       # validate-task, model-check, run, audit-run, report, export-hf
├── config.py                    # model and experiment config loading
├── schema.py                    # task/output contract and experiment models
├── experiments/                 # repair and feedback-replication runners
├── reporting/                   # rebuild run tables from events.jsonl; export HF datasets
├── models/
│   └── adapter.py               # reference_stub, replay, ollama and openai_compatible adapters
├── tasks/
│   └── validation.py            # package and path validation
└── verification/
    ├── analytical.py            # exact solutions and physical quantities
    ├── kepler_oracle.py         # independent DOP853 Kepler oracle
    ├── oracles.py               # Layer 2 oracle-comparison gate
    ├── invariants.py            # Layer 3 physical-invariant gates
    ├── convergence.py           # Layer 4 observed-order convergence gate
    ├── metamorphic.py           # Layer 5 metamorphic-relation gates
    ├── robustness.py            # Layer 6 held-out robustness cases
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
└── verification/           # analytical.py, kepler_oracle.py, solvers.py: trusted references; oracles.py, invariants.py, convergence.py, metamorphic.py, robustness.py: gates
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
`verification/{invariants,convergence,metamorphic,robustness}.py` stubs (all four have since
returned as real gates) and the empty
`tests/property/` and `tests/integration/` directories were removed on `develop` (#44) and are
gone from `main` since the develop/main merge.

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
