# Verification

InvariantLab grades a candidate solver with **Layers 0-6 (seven layers)**.
`verify_candidate` in `src/invariantlab/verification/verify.py` runs all seven on one
candidate workspace and returns a `VerificationResult`; `invariantlab verify` is the CLI
for it. This page lists what each layer checks, where its code and tests live, the gates
and thresholds per task, and the trust boundary the layers rely on.

## Running `invariantlab verify`

From the repository root, after `uv sync --extra dev`:

```bash
uv run invariantlab verify --task tasks/oscillator --candidate tasks/oscillator --output runs/verify/oscillator.json
```

This grades the trusted reference package against itself and prints
`PASSED oscillator_verlet: public=True scientific=True -> runs/verify/oscillator.json`.
`--candidate` is any directory that contains the task's `src/` tree, for example a
workspace built by `build_agent_workspace` and edited by an agent; only its `src/` tree is
used. `--attempt-id` (default `local`) is copied into the result. The command exits 0
whenever verification ran, including a `NOT PASSED` verdict, and exits 1 if it could not
run.

The JSON has one list of `GateResult`s per layer under `layers` (`L0` ... `L6`):

- `public_passed` is the Layer 1 result;
- `scientific_passed` requires every gate in L0 and L2-L6 to pass;
- `passed_all` requires both;
- if any L0 gate fails, L2-L6 each contain one failing gate named `skipped`.

## Layer status

Status reflects `main` at the time of writing. "Implemented" means the module exists, is
called by `verify_candidate`, and has the listed tests.

| Layer | Purpose | Module | Tests | Status |
|---|---|---|---|---|
| L0 | Run the entrypoint as a subprocess on the canonical case; check exit code, output present, NPZ keys/rank/shape/dtype (no pickle) and finite values | `src/invariantlab/verification/execution.py` (`run_task`) | `tests/unit/test_execution_gate.py` | Implemented |
| L1 | The task's visible public tests, run with pytest in the evaluation workspace | `tasks/*/tests/public` (run by `_run_public_tests` in `verify.py`) | `tests/integration/test_verify.py`, `tests/unit/test_workspace.py` | Implemented |
| L2 | Compare the state against an analytical or high-accuracy oracle | `src/invariantlab/verification/oracles.py` (`check_oracle`; references in `analytical.py`, `kepler_oracle.py`) | `tests/unit/test_oracle_gate.py` | Implemented |
| L3 | Physical invariants (energy, angular momentum, boundary values, maximum principle, decay, amplitude) | `src/invariantlab/verification/invariants.py` (`check_invariants`) | `tests/unit/test_invariant_gate.py` | Implemented |
| L4 | Observed convergence order on a 3-level refinement plan, re-run through the CLI/NPZ boundary | `src/invariantlab/verification/convergence.py` (`check_convergence`) | `tests/unit/test_convergence_gate.py` | Implemented |
| L5 | One metamorphic relation per task: re-run on a transformed input and compare | `src/invariantlab/verification/metamorphic.py` (`check_metamorphic`) | `tests/unit/test_metamorphic_gate.py` | Implemented |
| L6 | Held-out robustness cases: valid inputs graded by L0, L2 and L3, plus inputs that must be rejected | `src/invariantlab/verification/robustness.py` (`check_robustness`) | `tests/unit/test_robustness_gate.py` | Implemented |
| L0-L6 | Compose the layers into a `VerificationResult`; `invariantlab verify` | `src/invariantlab/verification/verify.py` (`verify_candidate`) | `tests/integration/test_verify.py` | Implemented |

Not planned for V1: unit-system conversion checks and floating-point stress cases. Earlier
README drafts listed them as Layers 5 and 6; no code or issue implements them. In V1,
Layer 5 is metamorphic relations and Layer 6 is held-out robustness cases.

## Gates per task

Dispatch is on the contract `id`. Each layer runs on the task's canonical case
(`canonical_parameters` in `verify.py`) unless stated otherwise. "Contract" thresholds come
from `numerics.tolerances` in `tasks/<task>/contract.yaml`; the others are module constants.

### Gates shared by every task

| Layer | Gate | Metric | Threshold source |
|---|---|---|---|
| L0 | `execution`, `output_present`, `archive_schema`, `finite` | Exit code 0 within the time limit; declared archive written; keys, rank, fixed dimensions and dtype match `output.arrays` and load without pickle; every value finite. A failed gate marks the later L0 gates `skipped`. | Pass/fail; time limit is contract `budgets.wall_seconds` |
| L1 | `public_tests` | `pytest tests/public` exits 0 | Pass/fail; time limit is contract `budgets.wall_seconds` |
| L2 | `state_relative_l2` | `‖state − exact‖₂ / ‖exact‖₂` over the whole output `state`; must be below the threshold | Contract `state_relative_l2` (1e-5 for all four tasks) |
| L4 | `observed_order[0]`, `observed_order[1]` | `p = log(E_k / E_{k+1}) / log 2`, where `E_k` is the L2 `state_relative_l2` at refinement level `k`; a level that fails L0 or the oracle fails both gates | `ORDER_MIN` / `ORDER_MAX` in `convergence.py`: `1.8 <= p <= 2.2` |
| L6 | `robustness[<case>]/<gate>` | Each valid case runs the L0, L2 and L3 gates above with the same thresholds | Case tables `<TASK>_VALID_CASES` in `robustness.py` |
| L6 | `robustness[<case>]/rejected` | Each rejection case must exit non-zero without writing the archive | Case tables `<TASK>_REJECTION_CASES` in `robustness.py` |

### `oscillator_verlet` (`tasks/oscillator`)

Canonical case: `x0=0.7, v0=-0.35, omega=1.7, dt=1e-3, n_steps=15000`.

| Layer | Gate | Metric | Threshold source |
|---|---|---|---|
| L2 | `state_relative_l2` | `(x, v)` trajectory vs. the closed form `analytical.oscillator_trajectory` | Contract `state_relative_l2` = 1e-5 |
| L3 | `energy_relative_drift` | `max_t abs(E(t) − E(0)) / abs(E(0))`, `E = v²/2 + ω²x²/2`; `<` threshold | Contract `energy_relative_drift` = 1e-6 |
| L4 | `observed_order[0..1]` | `dt = 4.32 / n`, `n = 540, 1080, 2160` | `convergence.py` band |
| L5 | `time_reversal` | Run forward, restart from `(x_T, −v_T)` and require the result to return to `(x0, v0)`; relative L2 residual `<=` threshold | `TIME_REVERSAL_THRESHOLD` = 1e-10 (`metamorphic.py`) |
| L6 | `robustness[...]` | 3 valid cases; 1 rejection case (negative `omega`) | `OSCILLATOR_*_CASES` |

### `kepler_verlet` (`tasks/kepler`)

Canonical case: circular orbit `kepler_circular_orbit(0.0, 2.5, 1.7, 0.37)` with
`mu=2.5, dt=0.004, n_steps=1080`.

| Layer | Gate | Metric | Threshold source |
|---|---|---|---|
| L2 | `state_relative_l2` | `(rx, ry, vx, vy)` trajectory vs. the DOP853 oracle `kepler_oracle.solve_kepler_high_accuracy`; an oracle error fails the gate | Contract `state_relative_l2` = 1e-5 |
| L3 | `energy_relative_drift` | Maximum relative drift of the orbital energy; `<` threshold | Contract `energy_relative_drift` = 1e-6 |
| L3 | `angular_momentum_relative_drift` | Maximum relative drift of `rx·vy − ry·vx`; `<` threshold | Contract `angular_momentum_relative_drift` = 1e-12 |
| L4 | `observed_order[0..1]` | Circular orbit, `mu=2.5`, `dt = 4.32 / n`, `n = 540, 1080, 2160` | `convergence.py` band |
| L5 | `rotation_covariance` | Rotate the initial state by `ROTATION_ANGLE` = 0.7 rad, re-run, rotate back; relative L2 vs. the base trajectory `<=` threshold | `ROTATION_COVARIANCE_THRESHOLD` = 1e-10 |
| L6 | `robustness[...]` | 3 valid cases; 1 rejection case (`dt = NaN`) | `KEPLER_*_CASES` |

### `heat_ftcs` (`tasks/heat1d`)

Canonical case: `nx=161, nt=1800, alpha=0.17, length=1.3, t_final=0.237`. `state` is the
final profile; `u0 = sin(πx/L)` with zero end points.

| Layer | Gate | Metric | Threshold source |
|---|---|---|---|
| L2 | `state_relative_l2` | Final profile vs. the decaying sine mode `analytical.heat_trajectory` | Contract `state_relative_l2` = 1e-5 |
| L3 | `dirichlet_boundary` | `max(abs(u[0]), abs(u[-1]))`; `<=` threshold | `DIRICHLET_BOUNDARY_TOLERANCE` = 1e-12 (`invariants.py`) |
| L3 | `max_principle` | `max abs(u_T) / max abs(u0)`; `<=` threshold | `HEAT_MAX_PRINCIPLE_BOUND` = 1.0 |
| L3 | `l2_decay` | `‖u_T‖₂ / ‖u0‖₂`; `<` threshold | `HEAT_L2_DECAY_BOUND` = 1.0 |
| L4 | `observed_order[0..1]` | `alpha·dt/dx² = 0.4` held fixed: `(nx, nt) = (41, 40), (81, 160), (161, 640)`, `t_final=0.1` | `convergence.py` band |
| L5 | `diffusive_scaling` | Double `length` and quadruple `alpha`; the profile must not change; relative L2 `<=` threshold | `DIFFUSIVE_SCALING_THRESHOLD` = 1e-10 |
| L6 | `robustness[...]` | 4 valid cases; 2 rejection cases (`alpha·dt/dx² = 0.52`, negative `alpha`) | `HEAT_*_CASES` |

### `wave_leapfrog` (`tasks/wave1d`)

Canonical case: `nx=401, nt=400, c=0.65, length=1.3, t_final=0.39`. `state` is the final
displacement; `u0 = sin(πx/L)` with zero end points.

| Layer | Gate | Metric | Threshold source |
|---|---|---|---|
| L2 | `state_relative_l2` | Final displacement vs. the standing wave `analytical.wave_standing_trajectory` | Contract `state_relative_l2` = 1e-5 |
| L3 | `dirichlet_boundary` | `max(abs(u[0]), abs(u[-1]))`; `<=` threshold | `DIRICHLET_BOUNDARY_TOLERANCE` = 1e-12 |
| L3 | `amplitude_bound` | `max abs(u_T) / max abs(u0)`; `<=` threshold | `1 + WAVE_AMPLITUDE_SLACK` = 1 + 1e-4 |
| L4 | `observed_order[0..1]` | Courant number 0.3 held fixed: `c=0.75`, `(nx, nt) = (81, 60), (161, 120), (321, 240)` | `convergence.py` band |
| L5 | `wave_scaling` | Double `length` and `c`; the displacement must not change; relative L2 `<=` threshold | `WAVE_SCALING_THRESHOLD` = 1e-10 |
| L6 | `robustness[...]` | 3 valid cases; 1 rejection case (Courant number 1.2) | `WAVE_*_CASES` |

## Trust boundary

The layers are only meaningful if the candidate never sees or edits the trusted evidence.
V1 enforces this with copy-based workspaces and an import boundary. It does **not**
sandbox candidate code during `invariantlab verify`.

### What agents see: `build_agent_workspace`

`build_agent_workspace(task_dir, dest)` in `src/invariantlab/tasks/workspace.py` copies
only:

- `contract.yaml` and `specification.md`;
- the entrypoint's source tree (`src/`);
- `tests/conftest.py` and the public tests (`tests/public`);
- `examples/`, if present.

Agents therefore see the contract, including its `numerics.tolerances`. They do **not** see:

- `tests/scientific/`;
- the trusted gates and references in `src/invariantlab/verification/`, including the L4
  refinement plans, the L5 thresholds and the L6 held-out case tables;
- mutation manifests (`tasks/*/mutations/`) and the legacy oscillator study harness
  (`tasks/oscillator/verifier.py`, `candidate_runner.py`).

The builder rejects symlinks, refuses a destination inside the task root, and fails if the
contract places the scientific tests inside the source or public-test tree. Tests:
`tests/unit/test_workspace.py` and `test_agent_workspace_contains_no_scientific_tests` in
`tests/acceptance/test_task_boundaries.py`.

### What is graded: `build_evaluation_workspace`

`build_evaluation_workspace(task_dir, candidate_workspace, dest)` takes **only** the
candidate's source tree (`src/`). It adds the trusted `contract.yaml`, `specification.md`,
the full `tests/` tree and `examples/` from `task_dir`. Candidate edits to the contract,
the tests or anything outside `src/` are discarded. `verify_candidate` always builds this
workspace first, and every layer runs against it.

### Import boundary

- Agent-facing code under `tasks/*/src` must not import `invariantlab.verification`,
  `invariantlab.mutations` or `tests.scientific`.
- Scientific tests must not import candidate solver modules.
- Trusted gates never import candidate code. They execute the entrypoint as a subprocess
  and read only the declared NPZ archive (`np.load(..., allow_pickle=False)`).

`tests/acceptance/test_task_boundaries.py` enforces the first two rules by parsing the
sources with Python's AST.

### `run_task` is not a sandbox

`run_task` (Layer 0, reused by L4-L6) and the Layer 1 pytest run start a plain host
subprocess: `sys.executable <entrypoint>` with the evaluation workspace as working
directory. It runs as the same user, with the same filesystem and network access as the
caller. The only limit is the wall-clock timeout. During evaluation, candidate code can
read anything the caller can, including the trusted `tests/` tree copied into the
evaluation workspace and the trusted code in the checkout. Copy-based workspaces hide the
evidence from the agent while it works; they do not isolate the evaluation itself. Only run
`invariantlab verify` on code you are willing to execute on that machine.

The Docker sandbox (`--network none`, memory, CPU and pid limits, read-only root) belongs to
the repair runner (`invariantlab run`). For package mutants (`interface: package`) that
runner calls `verify_candidate` with a `DockerExecutor`, so the candidate entrypoint and public
tests run in that container (#120); `interface: legacy_study` mutants are still graded by the
legacy `tasks/oscillator/verifier.py` harness. See
[Known Limitations](limitations.md#evaluation-harness).
