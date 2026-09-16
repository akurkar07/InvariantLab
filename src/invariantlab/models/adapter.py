"""Small model-adapter layer used by executable experiments."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from invariantlab.config import ModelConfig


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
    """Minimal Chat Completions adapter for OpenAI-compatible providers."""

    model_id: str
    base_url: str
    api_key_env: str
    temperature: float = 0.0
    max_tokens: int = 4096

    def generate(self, prompt: str) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key environment variable {self.api_key_env!r}"
            )

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
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "InvariantLab/0.1",
            },
        )

        max_retries = 10
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 60 * (attempt + 1)
                    console = __import__("rich.console", fromlist=["Console"]).Console()
                    console.print(f"  [yellow]429 rate limited. Waiting {wait}s...[/yellow]")
                    time.sleep(wait)
                    if attempt == max_retries - 1:
                        raise RuntimeError("Rate limit exceeded after retries") from e
                    continue
                raise

        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected provider response: {payload}") from exc


def build_adapter(config: ModelConfig) -> ModelAdapter:
    """Construct an adapter from a validated ModelConfig."""

    if config.adapter == "replay":
        return ReplayAdapter(model_id=config.model_id or "replay/oscillator-reference")
    if config.adapter == "ollama":
        return OllamaAdapter(
            model_id=config.model_id,
            base_url=str(getattr(config, "base_url", "http://localhost:11434")),
            temperature=float(config.temperature),
            max_tokens=int(config.max_tokens),
        )
    if config.adapter == "openai_compatible":
        extra = config.extra
        return OpenAICompatibleAdapter(
            model_id=config.model_id,
            base_url=str(extra.get("base_url", "https://openrouter.ai/api/v1")),
            api_key_env=str(extra.get("api_key_env", "OPENROUTER_API_KEY")),
            temperature=float(config.temperature),
            max_tokens=int(config.max_tokens),
        )
    raise ValueError(f"Unsupported model adapter: {config.adapter}")


@dataclass
class OllamaAdapter:
    """Adapter for locally-running Ollama models."""

    model_id: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 4096

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
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama request failed: {e.code} {e.reason}") from e

        try:
            return str(payload["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Ollama response: {payload}") from exc
