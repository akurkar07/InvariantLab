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
├── experiments/                 # repair and feedback-replication runners
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
