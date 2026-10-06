# Study 2: Update-order feedback replication

## Research question

Does diagnostic scientific feedback change repair quality on the subtle harmonic-oscillator
update-order defect observed in the first multi-condition study?

The first study produced one hardened-condition repair that fixed the original second
half-step expression but introduced stale cached acceleration, making the scientific error
substantially worse. Study 2 tests whether that observation replicates rather than treating
one run as evidence of a general effect.

## Preregistered hypotheses

**H1:** Diagnostic scientific feedback changes repair behaviour on subtle update-order
defects.

**H2:** Diagnostic feedback increases the rate of failed repairs that are scientifically
worse than the original defect. Failed candidates are retained for manual classification of
whether the local defect was removed and a new state-evolution defect was introduced.

## Design

One mutation is held fixed: the velocity-Verlet second half-step uses the stale acceleration

```python
v = v_half + 0.5 * dt * a
```

instead of the recomputed acceleration

```python
v = v_half + 0.5 * dt * a_new
```

Four prompt conditions are compared:

| Condition | Information shown |
|---|---|
| weak | Public tests pass and hidden scientific verification exists |
| placebo | Additional non-diagnostic evaluator metadata, but no scientific metrics |
| metrics | Raw state-error and energy-drift metrics |
| interpreted | Raw metrics plus threshold and metric interpretation |

Each condition receives 30 repair attempts for 120 total model calls. The condition/trial
schedule is shuffled deterministically with seed 1729 so calls are interleaved rather than
blocked by condition.

## Model

Study 2 was run on two local models served by Ollama through its OpenAI-compatible endpoint
(`http://localhost:11434/v1`), as two separate 120-cell runs of the design above:

| Model | Backend | Model tag | Temperature | Model config | Experiment config |
|---|---|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct | Ollama (local) | `qwen2.5-coder:7b-instruct` | 0.2 | `configs/models/ollama-qwen2.5-coder-7b.yaml` | `configs/experiments/update-order-feedback-replication-ollama.yaml` |
| DeepSeek-Coder-6.7B-Instruct | Ollama (local) | `deepseek-coder:6.7b-instruct` | 0.2 | `configs/models/ollama-deepseek-coder-6.7b.yaml` | `configs/experiments/update-order-feedback-replication-ollama-deepseek.yaml` |

The runs wrote to `runs/update-order-feedback-replication-ollama-qwen25-7b/` and
`runs/update-order-feedback-replication-ollama-deepseek/`, matching each experiment
config's `name`. The DeepSeek configs were committed after the run. Its model tag and
temperature follow the Qwen preset and have not yet been checked against the run's
`events.jsonl` metadata (marked `# unverified` in the model config).

The local presets use temperature 0.2 so that repeated identical prompts are not forced to
the same completion (see [`local-models.md`](./local-models.md)). Each model response is
treated as an independent observed attempt rather than a seeded deterministic sample.
Results are reported per model and are not pooled across models.

The originally preregistered model, `cohere/north-mini-code:free` through OpenRouter at
temperature 0.0 (`configs/models/cohere-north-mini-code-free.yaml`, used by
`configs/experiments/update-order-feedback-replication.yaml`), was not used for the
reported Study 2 results. See the deviations below.

## Deviations from the original protocol

This protocol was first committed in 4583762 (#45, 2026-09-15). The reported results
([`study-two-model-comparison.md`](./study-two-model-comparison.md), dated 2026-09-16)
differ from it as follows:

1. **Model and backend.** Preregistered: `cohere/north-mini-code:free` through OpenRouter.
   Run: local Ollama with Qwen2.5-Coder-7B-Instruct and DeepSeek-Coder-6.7B-Instruct, one
   run per model. Local endpoint support, the Ollama Qwen model config and the Ollama Study 2
   experiment config were added in 4db7373 (#46, 2026-09-15). The results and run directories
   were recorded in 5e3e8f4 (#52, 2026-09-16). The DeepSeek model and experiment configs were
   not committed until they were reconstructed for #99. This "Model" section still named the
   Cohere model until #99.
2. **Temperature.** Preregistered: 0.0. Run: 0.2 for both local models (4db7373, "Use
   stochastic sampling for local Ollama trials"). With a deterministic local server,
   temperature 0.0 would repeat the same completion on every trial. The local results are
   therefore a separate model/decoding condition from the Cohere temperature-0.0 setting.
3. **Artifact integrity procedure.** Added after preregistration and before the runs
   (4db7373): strict validation of the 120-cell schedule, `invalid_artifact` stop
   behaviour, and `invariantlab audit-run --write-canonical` producing
   `events.canonical.jsonl` and `artifact-integrity.json`. The reported tables were
   generated from the audited artifacts.
4. **Execution procedure.** Added after preregistration and before the runs (4db7373):
   bounded provider retries, `run-status.json` and batched resumable execution
   (`--max-new-attempts`). The original "Run" section ran a single invocation from the
   `exp/update-order-feedback-replication` branch.
5. **Experiment config schema (after the runs).** 9b0787d (#53, 2026-09-18) changed the
   Study 2 experiment configs from `runner: feedback_replication` / `mutation: update-order`
   to `runner: repair` / `task: tasks/oscillator` /
   `mutation: tasks/oscillator/mutations/update-order`. The runs used the earlier form. The
   committed configs, including the reconstructed DeepSeek config, use the current form.

Unchanged from the preregistration: the hypotheses, the update-order mutation, the four
prompt conditions, 30 attempts per condition, seed 1729 with randomised order, the
endpoints and the Wilson 95% confidence intervals.

## Endpoints

Primary endpoint:

- scientific pass rate by condition

Secondary endpoints:

- scientific regression rate
- repaired state error divided by baseline state error
- repaired energy drift divided by baseline energy drift
- worst scientific severity ratio
- manual failure classification

A scientific regression is a failed repair whose worst measured scientific ratio exceeds
1.0, meaning at least one tracked scientific metric is worse than the original defect.

Pass-rate summaries include Wilson 95% confidence intervals and pass-rate differences
relative to the weak condition. No claim of a condition effect should be made solely from a
single failed candidate.

## Artifact integrity

The configured Study 2 schedule is a strict set of 120 unique `(condition, trial)` cells.
Randomisation changes only execution order; it never creates additional cells.

On resume and before summary generation, InvariantLab validates `events.jsonl` for:

- duplicate scheduled cells;
- condition/trial pairs outside the configured schedule;
- mixed experiment, model, mutation or seed metadata;
- malformed records.

An invalid raw artifact is never silently counted. The runner writes
`artifact-integrity.json`, marks the run `invalid_artifact`, and stops.

Existing run directories can be audited without modifying the raw evidence:

```bash
uv run invariantlab audit-run \
  --experiment configs/experiments/update-order-feedback-replication-ollama.yaml \
  --run-dir runs/update-order-feedback-replication-ollama-qwen25-7b \
  --write-canonical
```

This writes a separate `events.canonical.jsonl` containing at most one valid record for
each scheduled cell, in schedule order. The original `events.jsonl` is preserved exactly
as collected. Duplicate, out-of-schedule, malformed and metadata-mismatched records remain
listed in `artifact-integrity.json` for provenance.

## Evidence and resumability

Every completed attempt is appended immediately to `events.jsonl`, including:

- condition and trial
- randomised schedule index
- prompt and prompt SHA-256
- raw model response
- candidate source
- baseline and repaired verifier output
- latency
- severity ratios
- manual-review flag

`study-summary.json` is refreshed after every completed attempt. Re-running the command in
the same output directory skips already completed condition/trial cells.

## Run

Install the development environment:

```bash
uv sync --locked --extra dev
```

### OpenRouter

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
uv run invariantlab run \
  --experiment configs/experiments/update-order-feedback-replication.yaml \
  --max-new-attempts 20
```

The runner retries transient provider failures with bounded exponential backoff. If a rate
limit or connection failure persists, it writes `run-status.json` and exits cleanly.
Re-running the same command resumes from `events.jsonl`.

### Local Ollama

```bash
ollama pull qwen2.5-coder:7b-instruct

uv run invariantlab model-check \
  --model configs/models/ollama-qwen2.5-coder-7b.yaml

uv run invariantlab run \
  --experiment configs/experiments/update-order-feedback-replication-ollama.yaml \
  --max-new-attempts 20
```

### Local vLLM

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \
  --generation-config vllm \
  --max-model-len 8192

uv run invariantlab model-check \
  --model configs/models/vllm-qwen2.5-coder-7b-awq.yaml

uv run invariantlab run \
  --experiment configs/experiments/update-order-feedback-replication-vllm.yaml \
  --max-new-attempts 20
```

Local-model setup is documented in [`local-models.md`](./local-models.md).

A configuration-only check can be run without making model calls:

```bash
uv run invariantlab run \
  --experiment configs/experiments/update-order-feedback-replication.yaml \
  --dry-run
```
