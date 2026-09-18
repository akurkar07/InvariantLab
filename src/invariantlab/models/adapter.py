"""Small model-adapter layer used by executable experiments."""

from __future__ import annotations

import contextlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from invariantlab.config import ModelConfig


class ModelRequestError(RuntimeError):
    """Base class for model-provider request failures."""


class ModelRateLimitError(ModelRequestError):
    """Raised when a provider remains rate limited after retries."""


class ModelConnectionError(ModelRequestError):
    """Raised when a provider cannot be reached after retries."""


class ModelAdapter(Protocol):
    """Common interface for one-shot coding-model requests."""

    model_id: str

    def generate(self, prompt: str) -> str:
        """Return the model's text response."""


@dataclass
class ReplayAdapter:
    """Deterministic adapter for CI and end-to-end smoke testing."""

    model_id: str = "replay/oscillator-reference"

    def generate(self, prompt: str) -> str:
        del prompt
        return """```python
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

    def _retry_delay(self, attempt: int, error: urllib.error.HTTPError) -> float:
        delay = min(
            self.backoff_initial_seconds * (2**attempt),
            self.backoff_max_seconds,
        )
        retry_after = error.headers.get("Retry-After") if error.headers else None
        if retry_after is not None:
            with contextlib.suppress(ValueError):
                delay = max(delay, float(retry_after))
        return float(min(delay, self.backoff_max_seconds))

    def _request(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You repair numerical Python code. Return only the complete "
                            "replacement solver.py inside one Python code fence."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=body,
            method="POST",
            headers=self._headers(),
        )
        with urllib.request.urlopen(
            request,
            timeout=self.request_timeout_seconds,
        ) as response:
            raw_payload = response.read().decode("utf-8")

        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise ModelRequestError(
                f"Provider returned invalid JSON: {raw_payload[:200]!r}"
            ) from exc

        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelRequestError(
                f"Unexpected provider response: {payload}"
            ) from exc

    def generate(self, prompt: str) -> str:
        retryable_statuses = {408, 425, 429, 500, 502, 503, 504}

        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                return self._request(prompt)
            except urllib.error.HTTPError as exc:
                if exc.code not in retryable_statuses:
                    raise ModelRequestError(
                        f"Provider returned HTTP {exc.code}: {exc.reason}"
                    ) from exc

                if attempt >= self.max_retries:
                    if exc.code == 429:
                        raise ModelRateLimitError(
                            "Provider rate limit persisted after "
                            f"{self.max_retries + 1} attempts"
                        ) from exc
                    raise ModelRequestError(
                        f"Provider HTTP {exc.code} persisted after "
                        f"{self.max_retries + 1} attempts"
                    ) from exc

                time.sleep(self._retry_delay(attempt, exc))
            except (TimeoutError, urllib.error.URLError) as exc:
                if attempt >= self.max_retries:
                    raise ModelConnectionError(
                        "Provider connection failed after "
                        f"{self.max_retries + 1} attempts: {exc}"
                    ) from exc
                delay = min(
                    self.backoff_initial_seconds * (2**attempt),
                    self.backoff_max_seconds,
                )
                time.sleep(delay)

        raise AssertionError("unreachable")


@dataclass
class OllamaAdapter:
    """Adapter for locally running Ollama models."""

    model_id: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 4096
    request_timeout_seconds: float = 300.0

    def generate(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You repair numerical Python code. Return only the complete "
                            "replacement solver.py inside one Python code fence."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/api/chat",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.request_timeout_seconds,
            ) as response:
                raw_payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise ModelRateLimitError(
                    f"Ollama rate limited the request: {exc.reason}"
                ) from exc
            raise ModelRequestError(
                f"Ollama request failed: {exc.code} {exc.reason}"
            ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            raise ModelConnectionError(f"Ollama connection failed: {exc}") from exc

        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise ModelRequestError(
                f"Ollama returned invalid JSON: {raw_payload[:200]!r}"
            ) from exc

        try:
            return str(payload["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise ModelRequestError(f"Unexpected Ollama response: {payload}") from exc


def build_adapter(config: ModelConfig) -> ModelAdapter:
    """Construct an adapter from a validated ModelConfig."""

    if config.adapter == "replay":
        return ReplayAdapter(model_id=config.model_id or "replay/oscillator-reference")

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
