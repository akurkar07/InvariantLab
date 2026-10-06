"""Small model-adapter layer used by executable experiments."""

from __future__ import annotations

import contextlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from invariantlab.config import ModelConfig


class ModelRequestError(RuntimeError):
    """Base class for model-provider request failures."""


class ModelRateLimitError(ModelRequestError):
    """Raised when a provider remains rate limited after retries."""


class ModelConnectionError(ModelRequestError):
    """Raised when a provider cannot be reached after retries."""


SYSTEM_PROMPT = (
    "You repair numerical Python code. Return only the complete "
    "replacement solver.py inside one Python code fence."
)
DEFAULT_REPLAY_MODEL_ID = "replay/oscillator-reference"
RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

T = TypeVar("T")


@dataclass(frozen=True)
class ModelResponse:
    """Model text plus provider-reported usage metadata."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None


class ModelAdapter(Protocol):
    """Common interface for one-shot coding-model requests."""

    model_id: str

    def generate(self, prompt: str) -> str:
        """Return the model's text response."""

    def complete(self, prompt: str) -> ModelResponse:
        """Return the model's text response with usage metadata."""


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _backoff_delay(
    attempt: int,
    initial_seconds: float,
    max_seconds: float,
    error: urllib.error.HTTPError | None = None,
) -> float:
    delay = min(initial_seconds * (2**attempt), max_seconds)
    retry_after = error.headers.get("Retry-After") if error and error.headers else None
    if retry_after is not None:
        with contextlib.suppress(ValueError):
            delay = max(delay, float(retry_after))
    return float(min(delay, max_seconds))


def _with_retries(
    send: Callable[[], T],
    *,
    provider: str,
    max_retries: int,
    backoff_initial_seconds: float,
    backoff_max_seconds: float,
    before_attempt: Callable[[], None] | None = None,
) -> T:
    """Call ``send`` with retry/backoff on transient HTTP and connection errors."""

    attempts = max_retries + 1
    for attempt in range(attempts):
        if before_attempt is not None:
            before_attempt()
        try:
            return send()
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_STATUSES:
                raise ModelRequestError(
                    f"{provider} returned HTTP {exc.code}: {exc.reason}"
                ) from exc
            if attempt >= max_retries:
                if exc.code == 429:
                    raise ModelRateLimitError(
                        f"{provider} rate limit persisted after {attempts} attempts"
                    ) from exc
                raise ModelRequestError(
                    f"{provider} HTTP {exc.code} persisted after {attempts} attempts"
                ) from exc
            time.sleep(
                _backoff_delay(attempt, backoff_initial_seconds, backoff_max_seconds, exc)
            )
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt >= max_retries:
                raise ModelConnectionError(
                    f"{provider} connection failed after {attempts} attempts: {exc}"
                ) from exc
            time.sleep(
                _backoff_delay(attempt, backoff_initial_seconds, backoff_max_seconds)
            )

    raise AssertionError("unreachable")


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    provider: str,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw_payload = response.read().decode("utf-8")
    try:
        parsed = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ModelRequestError(
            f"{provider} returned invalid JSON: {raw_payload[:200]!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ModelRequestError(f"Unexpected {provider} response: {parsed}")
    return parsed


@dataclass
class ReplayAdapter:
    """Deterministic adapter for CI and end-to-end smoke testing."""

    model_id: str = DEFAULT_REPLAY_MODEL_ID

    def generate(self, prompt: str) -> str:
        return self.complete(prompt).text

    def complete(self, prompt: str) -> ModelResponse:
        del prompt
        return ModelResponse(text=_REPLAY_SOLUTION)


_REPLAY_SOLUTION = """```python
def solve_oscillator_verlet(x0, v0, omega, dt, n_steps):
    trajectory = [(0.0, float(x0), float(v0))]
    x = float(x0)
    v = float(v0)
    omega2 = float(omega) * float(omega)
    t = 0.0

    for _ in range(int(n_steps)):
        a = -omega2 * x
        v_half = v + 0.5 * dt * a
        x = x + dt * v_half
        a_new = -omega2 * x
        v = v_half + 0.5 * dt * a_new
        t += dt
        trajectory.append((t, x, v))

    return trajectory
```"""


@dataclass
class OpenAICompatibleAdapter:
    """Chat Completions adapter with throttling and retry support."""

    model_id: str
    base_url: str
    api_key_env: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    request_timeout_seconds: float = 120.0
    max_retries: int = 5
    backoff_initial_seconds: float = 2.0
    backoff_max_seconds: float = 60.0
    min_request_interval_seconds: float = 0.0
    _last_request_started: float | None = field(default=None, init=False, repr=False)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "InvariantLab/0.1",
        }
        if not self.api_key_env:
            return headers

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key environment variable {self.api_key_env!r}"
            )
        headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _throttle(self) -> None:
        if self._last_request_started is not None:
            elapsed = time.monotonic() - self._last_request_started
            remaining = self.min_request_interval_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_started = time.monotonic()

    def _request(self, prompt: str) -> ModelResponse:
        payload = _post_json(
            self.base_url.rstrip("/") + "/chat/completions",
            {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            self._headers(),
            self.request_timeout_seconds,
            "Provider",
        )
        try:
            choice = payload["choices"][0]
            text = str(choice["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelRequestError(
                f"Unexpected provider response: {payload}"
            ) from exc

        usage = payload.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        return ModelResponse(
            text=text,
            input_tokens=_optional_int(usage.get("prompt_tokens")),
            output_tokens=_optional_int(usage.get("completion_tokens")),
            finish_reason=_optional_str(choice.get("finish_reason")),
        )

    def complete(self, prompt: str) -> ModelResponse:
        return _with_retries(
            lambda: self._request(prompt),
            provider="Provider",
            max_retries=self.max_retries,
            backoff_initial_seconds=self.backoff_initial_seconds,
            backoff_max_seconds=self.backoff_max_seconds,
            before_attempt=self._throttle,
        )

    def generate(self, prompt: str) -> str:
        return self.complete(prompt).text


@dataclass
class OllamaAdapter:
    """Adapter for locally running Ollama models."""

    model_id: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 4096
    request_timeout_seconds: float = 300.0
    max_retries: int = 3
    backoff_initial_seconds: float = 1.0
    backoff_max_seconds: float = 30.0

    def _request(self, prompt: str) -> ModelResponse:
        payload = _post_json(
            self.base_url.rstrip("/") + "/api/chat",
            {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
            {"Content-Type": "application/json"},
            self.request_timeout_seconds,
            "Ollama",
        )
        try:
            text = str(payload["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise ModelRequestError(f"Unexpected Ollama response: {payload}") from exc
        return ModelResponse(
            text=text,
            input_tokens=_optional_int(payload.get("prompt_eval_count")),
            output_tokens=_optional_int(payload.get("eval_count")),
            finish_reason=_optional_str(payload.get("done_reason")),
        )

    def complete(self, prompt: str) -> ModelResponse:
        return _with_retries(
            lambda: self._request(prompt),
            provider="Ollama",
            max_retries=self.max_retries,
            backoff_initial_seconds=self.backoff_initial_seconds,
            backoff_max_seconds=self.backoff_max_seconds,
        )

    def generate(self, prompt: str) -> str:
        return self.complete(prompt).text


def resolve_model_id(config: ModelConfig) -> str:
    """Return the model id an adapter built from ``config`` will report."""

    if config.adapter == "replay":
        return config.model_id or DEFAULT_REPLAY_MODEL_ID
    return config.model_id


def build_adapter(config: ModelConfig) -> ModelAdapter:
    """Construct an adapter from a validated ModelConfig."""

    if config.adapter == "replay":
        return ReplayAdapter(model_id=resolve_model_id(config))

    extra = config.extra
    if config.adapter == "ollama":
        return OllamaAdapter(
            model_id=config.model_id,
            base_url=str(extra.get("base_url", "http://localhost:11434")),
            temperature=float(config.temperature),
            max_tokens=int(config.max_tokens),
            request_timeout_seconds=float(
                extra.get("request_timeout_seconds", 300.0)
            ),
            max_retries=int(extra.get("max_retries", 3)),
            backoff_initial_seconds=float(
                extra.get("backoff_initial_seconds", 1.0)
            ),
            backoff_max_seconds=float(extra.get("backoff_max_seconds", 30.0)),
        )

    if config.adapter == "openai_compatible":
        base_url = str(extra.get("base_url", "")).strip()
        if not base_url:
            raise ValueError(
                "openai_compatible model configs must declare extra.base_url explicitly"
            )
        return OpenAICompatibleAdapter(
            model_id=config.model_id,
            base_url=base_url,
            api_key_env=str(extra.get("api_key_env", "")),
            temperature=float(config.temperature),
            max_tokens=int(config.max_tokens),
            request_timeout_seconds=float(
                extra.get("request_timeout_seconds", 120.0)
            ),
            max_retries=int(extra.get("max_retries", 5)),
            backoff_initial_seconds=float(
                extra.get("backoff_initial_seconds", 2.0)
            ),
            backoff_max_seconds=float(extra.get("backoff_max_seconds", 60.0)),
            min_request_interval_seconds=float(
                extra.get("min_request_interval_seconds", 0.0)
            ),
        )

    raise ValueError(f"Unsupported model adapter: {config.adapter}")
