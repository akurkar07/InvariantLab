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

The replication uses the same model family as the first study:

`cohere/north-mini-code:free` through OpenRouter, temperature 0.0.

Provider-side nondeterminism can still exist at temperature 0.0, so each model response is
treated as an independent observed attempt rather than a seeded deterministic sample.

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
