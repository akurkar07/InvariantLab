# Generic repair experiments

Generic repair experiments separate the scientific task, concrete mutation, model backend and feedback condition. New studies should prefer `runner: generic_repair` rather than embedding source code or verifier logic in a bespoke runner.

## Task definition

A repair task is a YAML file validated by `RepairTaskSpec`:

```yaml
id: oscillator-verlet
family: oscillator
description: Repair a harmonic-oscillator velocity-Verlet solver.
verifier: tasks/oscillator/scientific_verifier.py
entrypoint: solver.py
function_name: solve_oscillator_verlet
metric_keys:
  - max_state_relative_error
  - max_energy_relative_drift
scientific_threshold: 1.0e-3
```

The verifier is an executable Python file. During evaluation the runner mounts the candidate as `/work/<entrypoint>`, executes the verifier in the configured container and reads the final JSON line. The verifier must return `public_passed`, `scientific_passed`, `public`, `scientific` and `metrics` fields.

## Mutation definition

A mutation points to the broken implementation and records its causal label:

```yaml
id: oscillator-update-order
task: oscillator-verlet
family: update_order_error
source: tasks/oscillator/mutations/update_order_solver.py
location: velocity Verlet second half-step
expected_effect: stale acceleration increases state error and energy drift
```

The runner verifies the mutation before any model call. A valid repair mutation must pass its public checks and fail scientific verification.

## Experiment definition

```yaml
name: update-order-feedback-replication-ollama-qwen25-7b
task: tasks/oscillator/repair-task.yaml
mutation_config: tasks/oscillator/mutations/update-order.yaml
model: configs/models/ollama-qwen2.5-coder-7b.yaml
runner: generic_repair
conditions: [weak, placebo, metrics, interpreted]
n_attempts: 30
seed: 1729
randomize_order: true
container_image: python:3.12-slim
```

The four standard conditions differ only in verifier information shown to the model. The schedule is balanced across condition and trial, optionally shuffled with the experiment seed, and resumable from `events.jsonl`.

## Adding a task

1. Add a task YAML implementing `RepairTaskSpec`
2. Add a self-contained scientific verifier that emits the standard result JSON
3. Add at least one mutation YAML whose `task` matches the task ID
4. Add the mutated source file
5. Validate the experiment with `uv run invariantlab run --experiment <config> --dry-run`

Task-specific scientific checks belong in the task verifier, not in the generic runner.

## Adding a mutation

Add a `RepairMutationSpec` YAML and source file. Keep causal metadata explicit: mutation family, fault location and expected scientific effect. Mutation validation will be expanded by the oscillator mutation-system milestone; the generic runner already rejects mutations that do not preserve the intended public/scientific gap.

## Adding a model

Model selection remains independent of task and mutation configuration. Add a model YAML under `configs/models/` using an existing adapter, then point the experiment's `model` field at it. No runner changes are required.

## Backwards compatibility

`first_model` and `feedback_replication` remain available for replaying historical experiments. Study 2 configs now use the generic runner, so the same task and mutation definition can be reused across hosted, Ollama and vLLM model backends.
