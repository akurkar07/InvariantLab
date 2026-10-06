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

Hosted APIs are optional rather than the default. To use one, configure the generic
`openai_compatible` adapter with an explicit `base_url` and API-key environment variable;
see `configs/models/api-example.yaml`.

See [model execution](docs/local-models.md) for Ollama, vLLM and optional API endpoint setup.

## Repository layout

```text
src/invariantlab/
├── cli.py                       # validate-task, model-check, run, audit-run
├── config.py                    # model and experiment config loading
├── schema.py                    # task/output contract and experiment models
├── experiments/                 # repair, first-model and feedback-replication runners
├── models/
│   └── adapter.py               # replay, ollama and openai_compatible adapters
├── tasks/
│   └── validation.py            # package and path validation
└── verification/
    ├── analytical.py            # exact solutions and physical quantities
    ├── kepler_oracle.py         # independent DOP853 Kepler oracle
    └── solvers.py               # trusted numerical references used by tests

configs/
├── experiments/                 # experiment configs (Study 1/2, smoke)
└── models/                      # model configs (local-first default)

tasks/
├── oscillator/
├── kepler/
├── heat1d/
└── wave1d/

tests/
├── unit/                        # oracle, solver and convergence evidence
└── acceptance/                  # trust-boundary and repository acceptance checks

scripts/
└── validate_task.py             # validates the four committed V1 packages
```

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

## Current scope

InvariantLab is the benchmark core **plus** the evaluation harness used to run model studies against it.

It currently provides:

- four task packages (`oscillator`, `kepler`, `heat1d`, `wave1d`) with public and scientific suites
- trusted reference solvers and analytical / high-accuracy oracles
- task contract and artifact validation (`invariantlab validate-task`, `scripts/validate_task.py`)
- the oscillator repair experiment runner, which executes candidate repairs in Docker (`invariantlab run`)
- `replay`, `ollama` and `openai_compatible` model adapters (`invariantlab model-check`)
- raw run-evidence auditing (`invariantlab audit-run`)

It does **not** provide yet:

- verification layers 3-6 (invariants, convergence, metamorphic, robustness) as reusable gates
- a mutation registry
- reporting
- a dashboard
- Hugging Face dataset export
