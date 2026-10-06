# Model adapters

InvariantLab currently implements three model adapters: `replay`, `ollama` and
`openai_compatible`. Model configurations select an adapter by ID; backends are not loaded
through optional provider SDKs.

## Interface

`ModelAdapter` is a protocol with a `model_id` string and two methods:

- `complete(prompt) -> ModelResponse`, returning `text`, `input_tokens`, `output_tokens` and
  `finish_reason` (`ModelResponse` usage fields and finish reason may be `None`).
- `generate(prompt) -> str`, returning only the response text.

The adapters share `SYSTEM_PROMPT`. Build one from a validated `ModelConfig` with
`invariantlab.models.build_adapter(ModelConfig)`.

`ModelConfig` has these top-level fields:

| Field | Default | Purpose |
| --- | --- | --- |
| `adapter` | Required | Adapter identifier. |
| `model_id` | `""` | Provider's model name; replay uses `replay/oscillator-reference` when empty. |
| `temperature` | `0.0` | Sampling temperature for network adapters. |
| `max_tokens` | `32000` | Maximum generated tokens for network adapters. |
| `extra` | `{}` | Adapter-specific endpoint and request settings. |

## Supported adapters

| Adapter ID | Request endpoint | Response usage fields |
| --- | --- | --- |
| `replay` | None; returns a fixed reference solver. | No usage fields. |
| `ollama` | `{base_url}/api/chat` (default base URL `http://localhost:11434`). | `prompt_eval_count` → input tokens, `eval_count` → output tokens, `done_reason` → finish reason. |
| `openai_compatible` | `{base_url}/chat/completions`; `base_url` is required and any trailing slash is removed. | `usage.prompt_tokens` → input tokens, `usage.completion_tokens` → output tokens, `choices[0].finish_reason` → finish reason. |

The network adapters send the shared system prompt and the supplied prompt as chat messages.
Missing or non-integer usage counts and non-string finish reasons are returned as `None`.

### Adapter-specific `extra` keys

`replay` accepts no adapter-specific keys. An empty `model_id` selects
`replay/oscillator-reference`.

`ollama` defaults:

| Key | Default |
| --- | --- |
| `base_url` | `"http://localhost:11434"` |
| `request_timeout_seconds` | `300.0` |
| `max_retries` | `3` |
| `backoff_initial_seconds` | `1.0` |
| `backoff_max_seconds` | `30.0` |

`openai_compatible` defaults:

| Key | Default |
| --- | --- |
| `base_url` | Required; no default |
| `api_key_env` | `""` (omit the authorization header) |
| `request_timeout_seconds` | `120.0` |
| `max_retries` | `5` |
| `backoff_initial_seconds` | `2.0` |
| `backoff_max_seconds` | `60.0` |
| `min_request_interval_seconds` | `0.0` |

When `api_key_env` names an unset environment variable, the request raises `RuntimeError`.

## Reaching other backends

vLLM, OpenRouter, Ollama's OpenAI-compatible `/v1` endpoint and other hosted APIs use
`openai_compatible`. Existing examples include
[`configs/models/vllm-qwen2.5-coder-7b-awq.yaml`](../configs/models/vllm-qwen2.5-coder-7b-awq.yaml),
[`configs/models/cohere-north-mini-code-free.yaml`](../configs/models/cohere-north-mini-code-free.yaml),
and the two Ollama `/v1` configs
([Qwen](../configs/models/ollama-qwen2.5-coder-7b.yaml),
[DeepSeek](../configs/models/ollama-deepseek-coder-6.7b.yaml)).

Anthropic and Hugging Face `transformers` are not supported natively. For Anthropic models,
use an OpenAI-compatible endpoint such as OpenRouter or the provider's OpenAI-compatible API.
For Hugging Face weights, serve them with vLLM or TGI and point `openai_compatible` to that
server.

## Retries and run status

Network adapters retry HTTP 408, 425, 429, 500, 502, 503 and 504 responses, as well as
timeouts and URL connection errors. Retries use bounded exponential backoff; a `Retry-After`
header is honored up to `backoff_max_seconds`. `max_retries` is the number of retries after
the initial request.

After retries are exhausted, persistent HTTP 429 raises `ModelRateLimitError`; timeout or
connection failures raise `ModelConnectionError`; other HTTP failures raise
`ModelRequestError`. Invalid JSON and unexpected response shapes also raise
`ModelRequestError` without retry. The repair runner records these as `paused_rate_limit`,
`paused_connection` and `paused_provider_error`, respectively, and stops cleanly. Rerun the
same command to resume. A missing API-key environment variable instead raises `RuntimeError`
from request setup; the repair runner does not convert it to a paused run status.

See [Local Models](local-models.md#resumability-and-endpoint-failures) for run resumption and
endpoint troubleshooting.

## Add a backend

Implement a dataclass in `src/invariantlab/models/adapter.py` with `model_id`,
`complete(prompt)` and `generate(prompt)`. Reuse `_post_json` and `_with_retries` where
applicable, add its selection branch to `build_adapter`, export it from
`src/invariantlab/models/__init__.py` if needed, add a YAML config under `configs/models/`,
and add unit tests in `tests/unit/test_model_adapter.py`. Update this guide alongside the
implementation.

## Add a model

Create a YAML file under `configs/models/`; `adapter` selects the backend.

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
