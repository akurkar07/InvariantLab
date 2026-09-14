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

## Repository layout

```text
src/invariantlab/
├── schema.py                    # task/output contract models
├── tasks/
│   └── validation.py            # package and path validation
└── verification/
    ├── analytical.py            # exact solutions and physical quantities
    ├── kepler_oracle.py         # independent DOP853 Kepler oracle
    └── solvers.py               # trusted numerical references used by tests

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

This repository intentionally stops at the implemented benchmark core.

It does **not** currently provide:

- an LLM/model execution framework
- automatic mutation generation
- a general-purpose sandbox/container runner
- reporting or dashboard infrastructure
- Hugging Face dataset export
- a generic multi-domain benchmark plugin system

Those are possible future extensions only if actual experiments require them.

The immediate goal is to keep the benchmark small enough that every layer has a clear scientific purpose.
