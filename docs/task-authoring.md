# Task Authoring

An InvariantLab task is a self-contained executable package under `tasks/<family>/`.
The package is copied into an isolated workspace for an evaluation attempt; trusted
verifier code is mounted separately and is never available to the agent.

## Package layout

Every task contract names real paths beneath its own directory:

```text
tasks/<family>/
├── contract.yaml
├── specification.md
├── src/
│   └── solver.py
└── tests/
    ├── public/
    │   └── test_solver.py
    └── scientific/
        └── test_reference.py
```

- `contract.yaml` declares the entrypoint, visible and hidden tests, output archive,
  budgets, dtype, seed, and task-specific tolerances.
- `specification.md`, `src/`, and `tests/public/` are visible in the agent workspace.
- `tests/scientific/`, trusted oracle code, mutation manifests, and evaluator source
  are excluded from the agent mount and added only by the evaluator.
- Every declared path is relative to the task root. Absolute paths, `..` traversal,
  and links escaping the task root are invalid.

## Invocation protocol

The evaluator starts each Python task without a shell and with the task directory as
its working directory:

```text
python src/solver.py --input input.json --output result.npz
```

The paths come from the evaluator and contract; an entrypoint must not search outside
the task workspace. A task run follows this protocol:

1. The evaluator writes `input.json` and removes any previous output archive.
2. The entrypoint reads the input, performs one deterministic solve, and writes the
   requested archive.
3. A successful run exits with status `0`. Invalid input or a failed solve exits
   non-zero and may write a diagnostic to standard error.
4. Standard output is diagnostic only. The evaluator never parses it as a result.
5. The entrypoint writes no files other than the requested output beneath the task
   workspace.

`input.json` is a UTF-8 JSON object with this envelope:

```json
{
  "task_id": "oscillator_verlet",
  "parameters": {
    "x0": 1.0,
    "v0": 0.25,
    "omega": 1.5,
    "dt": 0.01,
    "n_steps": 100
  },
  "numerics": {
    "dtype": "float64",
    "seed": 42
  }
}
```

`task_id` must equal the contract ID. `parameters` are defined by the task
specification rather than inferred by the evaluator. `numerics.dtype` and
`numerics.seed` must agree with the contract. Unknown or missing required fields fail
closed; task packages must not silently substitute a different physical problem.

## Output protocol

A successful entrypoint writes one compressed NumPy archive at the requested output
path, normally `result.npz`:

```python
np.savez_compressed(output_path, time=time, state=state)
```

The archive must satisfy the contract exactly:

- the set of array names is exact; extra and missing arrays are invalid;
- each array has the declared rank, fixed dimensions, and dtype;
- every numeric value is finite;
- wildcard dimensions in the contract are written as `null`;
- object arrays and pickled values are forbidden.

The contract, not a reference implementation, defines the public output shape.

## Trust boundaries

The three numerical code paths have different roles:

| Boundary | Location | Visible to agent | Responsibility |
|---|---|---:|---|
| Agent-facing implementation | `tasks/<family>/src/` | Yes | Candidate solver edited and executed in the workspace |
| Trusted reference and verifier | `src/invariantlab/verification/` | No | Independent analytical/numerical evidence and verification |
| Controlled mutation machinery | `src/invariantlab/mutations/` | No | Deterministic defect construction and labels |

Agent-facing source must not import:

- `invariantlab.verification` or any of its submodules;
- `invariantlab.mutations` or any of its submodules;
- hidden `tests.scientific` helpers.

Trusted code consumes the entrypoint's serialized archive. It does not call or share
the candidate's numerical update functions. Shared dependencies are limited to data
formats and general third-party numerical primitives. CI enforces the import boundary
by parsing task source with Python's AST.

## Tests and tolerances

Public tests establish only ordinary interface behavior and a small visible example.
They must not disclose the held-out cases or thresholds that constitute scientific
certification.

Scientific tests use independent analytical or high-accuracy numerical references.
Each tolerance belongs to a named metric for that task and includes a short
justification. Do not copy energy-drift tolerances onto a diffusion task or use one
state threshold indiscriminately across methods and precisions.

At minimum, a completed reference task must prove:

- its entrypoint runs from the task root using the documented command;
- its archive satisfies the declared contract;
- public tests pass in the agent-visible mount;
- M2 oracle and refinement tests pass for non-special parameter values;
- no forbidden dependency crosses the task/verifier boundary.

The reusable six-layer verification pipeline, controlled mutants, model evaluation,
and report generation are later milestones; they are not prerequisites for authoring
an executable M2 task package.
