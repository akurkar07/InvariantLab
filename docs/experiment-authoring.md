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

The verifier must read `/work/solver.py`, run public and scientific checks, and print one
JSON result containing `public_passed`, `scientific_passed` and `metrics`.

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
source: solver.py
expected_effect: stale acceleration in the second velocity half-step
```

The mutation identifier is written to run records. The directory path is only configuration,
so moving from the legacy `mutation: update-order` field to a path does not change the
Study 2 record identifier.

## Add a model

Create a YAML file under `configs/models/`. Model backends are selected by the `adapter`
field and built through `invariantlab.models.build_adapter`.

For an OpenAI-compatible endpoint:

```yaml
adapter: openai_compatible
model_id: provider/model
temperature: 0.7
max_tokens: 4096
extra:
  base_url: https://example.test/v1
  api_key_env: EXAMPLE_API_KEY
  max_retries: 5
```

For Ollama:

```yaml
adapter: ollama
model_id: qwen2.5-coder:7b-instruct
temperature: 0.7
max_tokens: 4096
extra:
  base_url: http://localhost:11434
```

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
task_suite: configs/task-suites/v1-smoke.yaml
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

`first_model.py` remains available for the original one-shot smoke experiment.
`feedback_replication.py` remains import-compatible but delegates execution and auditing to
the generic repair runner.
