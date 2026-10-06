# Config-driven repair experiments

Repair studies are assembled from three independent pieces: a task, a mutation and a model.
The runner reads those pieces from configuration instead of embedding solver source or
verifier code in the experiment module.

## Add a task

Create a task directory under `tasks/` with:

```text
tasks/<task>/
├── contract.yaml
├── task.yaml
├── verifier.py
└── repair_prompt.txt
```

`contract.yaml` remains the agent-facing task contract. `task.yaml` supplies evaluator
metadata:

```yaml
id: oscillator_verlet
family: oscillator
contract: contract.yaml
verifier: verifier.py
prompt_template: repair_prompt.txt
feedback_metrics:
  max_state_relative_error:
    label: max state relative error
    threshold: 1.0e-3
    interpretation: long-horizon disagreement with the analytical solution
interpreted_feedback: >-
  Explain what the exposed metrics mean without revealing held-out cases.
```

Evaluation runs in two sandboxed stages. Stage 1 runs `candidate_runner.py` next to the
candidate and writes `/output/trajectories.json`. Stage 2 runs the verifier alone as
`verifier.py /data/trajectories.json /verifier/thresholds.json`. The runner writes
`thresholds.json` from the `feedback_metrics.*.threshold` values in `task.yaml`, so
`task.yaml` is the only place a study-gate threshold is set. The verifier prints one JSON
result containing `public_passed`, `scientific_passed` and `metrics`.

### Oscillator study gate versus contract tolerances

The oscillator study gate (`tasks/oscillator/verifier.py`, both thresholds `1.0e-3` in
`task.yaml`) is separate from the contract tolerances in `contract.yaml`
(`state_relative_l2: 1.0e-5`, `energy_relative_drift: 1.0e-6`). They measure different
things:

- The contract tolerances are calibrated for the contract's own scientific case
  (x0=0.7, v0=-0.35, omega=1.7, dt=1e-3, 15000 steps). They are checked by the M3 NPZ gates.
- The study gate checks three coarser held-out cases. It uses the final-state error scaled
  by `max(1, |x_ref|, |v_ref|)` and the maximum relative energy drift over the trajectory,
  each maximised over the cases.

Measured values for a correct velocity-Verlet solver and the `update-order` mutant:

| Case (x0, v0, omega, dt, steps) | Correct state error | Correct energy drift | Mutant state error | Mutant energy drift |
| --- | --- | --- | --- | --- |
| 1.0, 0.0, 1.0, 0.01, 800 | 3.72e-5 | 2.50e-5 | 2.07e-2 | 4.16e-2 |
| 0.3, -0.4, 1.7, 0.005, 1200 | 1.31e-5 | 1.12e-5 | 1.47e-2 | 4.59e-2 |
| -0.8, 0.25, 0.7, 0.01, 900 | 8.56e-6 | 1.02e-5 | 9.24e-3 | 2.23e-2 |
| Contract case: 0.7, -0.35, 1.7, 0.001, 15000 | 3.19e-6 | 6.65e-7 | n/a | n/a |

A correct solver's energy drift on the study cases (up to 2.5e-5) is above the contract's
1e-6 tolerance. Copying the contract tolerances into the study gate would therefore fail
correct solvers. The 1e-3 gate leaves a margin of more than 25x over correct solvers and
sits more than 9x below the mutant on every metric. The value 1e-3 was kept, not
recalibrated, so that verdicts stay comparable with Study 1 and Study 2.
`tests/unit/test_oscillator_verifier.py` asserts a margin of at least 10x for a correct
solver and that the mutant exceeds both thresholds. `tests/unit/test_repair_experiment.py`
pins both thresholds at 1e-3.

## Add a mutation

Store each controlled defect beneath its task:

```text
tasks/<task>/mutations/<mutation>/
├── mutation.yaml
└── solver.py
```

Example:

```yaml
id: update-order
task_id: oscillator_verlet
family: update_order_error
interface: legacy_study
source: solver.py
expected_effect: stale acceleration in the second velocity half-step
```

The mutation identifier is written to run records. The directory path is only configuration,
so moving from the legacy `mutation: update-order` field to a path does not change the
Study 2 record identifier.

Package mutants use `interface: package`, declare `expected_failures` entries with a
scientific pytest node id in `test` and an output regex in `message`, and may set
`max_changed_lines` (default 10). `invariantlab.mutations.discover_mutants` validates
the manifests and returns registered mutants; it raises `MutationRegistryError` with
all discovered problems when any declaration is invalid.

`invariantlab.mutations.validate_reference` requires at least one testcase in each public
and scientific suite and no failures, errors, or skips. `validate_mutant` verifies that
(1) public tests pass, (2) every declared scientific failure occurs with its declared
message and there are no unexpected failures or errors, and (3) the parseable mutant source
diff stays within its line budget and imports no `invariantlab` modules. These implement
the four reference/mutant validation checks; each suite timeout defaults to
`budgets.wall_seconds`. Both run tests on the repository virtual environment against a
temporary task copy without `mutations/`; their stable result names are consumed by #64
and #88.

Curated package mutants (each passes `validate_mutant`; checked by
`tests/unit/test_mutation_registry.py::test_real_task_package_mutants_validate`):

| Task | Mutant id | Family | Defect |
|---|---|---|---|
| kepler | `non-conservative-velocity-damping` | `non_conservative_update` | closing velocity half-step scaled by `1 - 2e-9` (fails `energy_relative_drift`) |
| kepler | `unit-error-au-rounding` | `unit_error` | `mu` scaled by `(1.495978707e11 / 1.496e11)^3` (IAU vs rounded AU round trip) |
| oscillator | `non-conservative-damping` | `non_conservative_update` | closing velocity half-step scaled by `1 - 1e-6` |
| wave1d | `sign-error-startup` | `sign_error` | ghost level `u(-dt)` built with `-0.5 * C²` instead of `+` |
| wave1d | `update-order-overwrite` | `update_order_error` | `state = next_state; previous = state` overwrites the old level |

The oscillator update-order (stale acceleration) and sign-error defects fail the unmodified
oscillator public example (`abs=1e-4`), so those families are realised on wave1d. The legacy
`update-order` (Study 2 record id) and `sign-error` directories keep their defects and
`interface: legacy_study` until #120.

## Add a model

Create a YAML file under `configs/models/`; its `adapter` field selects the backend.
See [Model adapters](model-adapters.md) for adapter ids, `extra` keys and examples.

The `reference_stub` adapter returns a fixed, known-correct oscillator solver for smoke
tests; it ignores the prompt and is not a model. The `replay` adapter re-serves responses
recorded in a repair run's `events.jsonl`, matched by prompt SHA-256. Duplicate prompts
are served in `schedule_index`/`trial` order; a missing or exhausted prompt raises
`ReplayMissError`. If all recorded responses name the same model, `model_id` defaults to
`replay/<recorded model>`.

```yaml
adapter: replay
extra:
  events_path: runs/<name>/events.jsonl
```

## Add an experiment

Bind the task, mutation and model in `configs/experiments/`:

```yaml
name: oscillator-update-order
model: configs/models/ollama-qwen2.5-coder-7b.yaml
runner: repair
task: tasks/oscillator
mutation: tasks/oscillator/mutations/update-order
conditions:
  - weak
  - placebo
  - metrics
  - interpreted
n_attempts: 30
seed: 1729
randomize_order: true
container_image: python:3.12-slim
```

Validate the configuration without model calls:

```bash
uv run invariantlab run \
  --experiment configs/experiments/oscillator-update-order.yaml \
  --dry-run
```

The generic runner evaluates the configured mutant as the baseline, builds condition-specific
prompts from task metadata, executes model repairs, invokes the task verifier, and writes the
same resumable evidence and integrity artifacts used by Study 2.

## Summary fields

The repair runner writes `study-summary.json` after every completed attempt. Rates are
computed by `invariantlab.metrics` over the canonical records (one per scheduled cell).
Each rate's denominator is the number of completed canonical cells in that condition;
failed, erroring and timed-out candidates count as non-passing and stay in the
denominator. Rates, gaps and intervals are `null` for a condition with no completed cells.

Top level:

| Field | Definition |
| --- | --- |
| `primary_endpoint` | Always `scientific_pass_rate`. |
| `baseline` | Verifier output for the unrepaired mutant. |
| `baseline_verification_gap` | `int(baseline.public_passed) - int(baseline.scientific_passed)`; 1 for a mutant that passes public tests but fails scientific ones. |
| `target_cells` / `completed_cells` / `complete` | Scheduled cells, completed canonical cells, and whether they are equal. |

Per condition (`by_condition.<condition>`):

| Field | Definition |
| --- | --- |
| `completed` / `target` | Completed canonical cells and `n_attempts`. |
| `public_passes` | Candidates whose `repaired.public_passed` is true. |
| `public_pass_rate` | `P_public = public_passes / completed`. |
| `public_pass_rate_wilson95` | Wilson score interval (z = 1.96) for `P_public`. |
| `scientific_passes` | Candidates whose scientific checks pass (`successful_repair`). |
| `scientific_pass_rate` | `P_science = scientific_passes / completed`. |
| `scientific_pass_rate_wilson95` | Wilson score interval (z = 1.96) for `P_science`. |
| `verification_gap` | `G = P_public - P_science`. |
| `scientific_regressions` | Candidates whose worst scientific ratio exceeds 1 (worse than the mutant). |
| `median_worst_scientific_ratio` / `max_worst_scientific_ratio` | Median and maximum of the per-candidate worst scientific error ratio versus the mutant. |
| `pass_rate_difference_vs_weak` | `scientific_pass_rate` minus the `weak` condition's rate (present when a `weak` condition exists). |

## Legacy compatibility

`feedback_replication.py` remains import-compatible but delegates execution and auditing to
the generic repair runner. The `first_model` runner has been removed; the original Study 1
defect and its mapping to the repair runner are documented in
[First Model Experiment](first-model-experiment.md).

`task_suite` is optional and currently ignored by all runners. A dry run warns when it is
present; multi-task suite evaluation is reserved for #120. Dry runs validate task and
mutation paths and their referenced files, construct the configured model adapter without
making a request, and reject placeholder container images.
