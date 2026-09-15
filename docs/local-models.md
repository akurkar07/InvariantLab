# Running InvariantLab with local models

InvariantLab can use any local server that exposes an OpenAI-compatible
`/v1/chat/completions` endpoint. Two ready-to-use presets are included:

- Ollama + `qwen2.5-coder:7b-instruct`
- vLLM + `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`

The scientific evaluator still runs generated code inside Docker, so Docker must also be
available locally.

The local presets use `temperature: 0.2` rather than zero. Repeating an identical prompt
against a deterministic temperature-zero local server can produce the same completion on
every trial, which would overstate the effective sample size. Treat local-model results as
a separate model/decoding condition rather than pooling them directly with the existing
Cohere temperature-zero Study 2 cells.

## Ollama

Pull the model:

```bash
ollama pull qwen2.5-coder:7b-instruct
```

Make sure the Ollama service is running. On installations where it is not already started:

```bash
ollama serve
```

Verify that InvariantLab can reach it:

```bash
uv run invariantlab model-check \
  --model configs/models/ollama-qwen2.5-coder-7b.yaml
```

Run Study 2 in 20-cell batches:

```bash
uv run invariantlab run \
  --experiment configs/experiments/update-order-feedback-replication-ollama.yaml \
  --max-new-attempts 20
```

Run the same command again to continue. Completed cells are read from `events.jsonl` and
are never repeated.

The bundled Ollama model name is `qwen2.5-coder:7b-instruct`. Ollama currently distributes
that 7B instruction model as a Q4_K_M build.

## vLLM

InvariantLab also includes an OpenAI-compatible vLLM preset using Qwen's official 4-bit AWQ
checkpoint.

One example server command is:

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \
  --generation-config vllm \
  --max-model-len 8192
```

Then verify the endpoint:

```bash
uv run invariantlab model-check \
  --model configs/models/vllm-qwen2.5-coder-7b-awq.yaml
```

Run the experiment:

```bash
uv run invariantlab run \
  --experiment configs/experiments/update-order-feedback-replication-vllm.yaml \
  --max-new-attempts 20
```

If you serve a different model, copy the YAML preset and change only `model_id` and, if
needed, `base_url`.

## Resumability and provider failures

Every successful model response is appended immediately to `events.jsonl`. The runner also
maintains `run-status.json` with one of these states:

- `running`
- `batch_complete`
- `paused_rate_limit`
- `paused_connection`
- `paused_provider_error`
- `complete`

HTTP 429, transient 5xx responses, timeouts and connection failures are retried with bounded
exponential backoff. If the provider remains unavailable, the experiment stops cleanly and
can be resumed with the same command.

Malformed model output or candidate code does not abort the study. It is recorded as a
failed repair with `candidate_error`, allowing weaker local models to be evaluated without
losing the remaining run.

## Configuring another OpenAI-compatible server

Model configs support the following `extra` fields:

```yaml
extra:
  base_url: http://localhost:11434/v1
  api_key_env: ""
  request_timeout_seconds: 300
  max_retries: 2
  backoff_initial_seconds: 2
  backoff_max_seconds: 10
  min_request_interval_seconds: 0
```

Set `api_key_env` to an empty string for unauthenticated local endpoints. For hosted
providers, set it to the environment-variable name containing the API key.

## References

- Ollama OpenAI compatibility: https://ollama.com/blog/openai-compatibility
- Ollama Qwen2.5-Coder: https://ollama.com/library/qwen2.5-coder
- vLLM OpenAI-compatible server:
  https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/
- Qwen2.5-Coder-7B-Instruct-AWQ:
  https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
