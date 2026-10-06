"""Unit tests for resilient model-adapter requests (no network)."""

import hashlib
import io
import json
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import pytest

from invariantlab.config import ModelConfig
from invariantlab.models import (
    ModelConnectionError,
    ModelRateLimitError,
    ModelRequestError,
    ModelResponse,
    OllamaAdapter,
    OpenAICompatibleAdapter,
    ReferenceStubAdapter,
    ReplayAdapter,
    ReplayMissError,
    build_adapter,
    resolve_model_id,
)


class _FakeResponse:
    def __init__(self, content: str = "", payload: object = None, raw: bytes | None = None):
        if payload is None:
            payload = {"choices": [{"message": {"content": content}}]}
        self._payload = raw if raw is not None else json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return self._payload


def _http_error(code: int, reason: str, retry_after: str | None = "0") -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        "http://localhost/v1/chat/completions",
        code,
        reason,
        headers,
        io.BytesIO(reason.encode("utf-8")),
    )


def _rate_limit_error() -> urllib.error.HTTPError:
    return _http_error(429, "Too Many Requests")


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("invariantlab.models.adapter.time.sleep", sleeps.append)
    return sleeps


def test_local_endpoint_does_not_require_api_key(monkeypatch):
    seen_authorization = []

    def fake_urlopen(request, timeout):
        del timeout
        seen_authorization.append(request.get_header("Authorization"))
        return _FakeResponse("local response")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    adapter = OpenAICompatibleAdapter(
        model_id="local/test",
        base_url="http://localhost:11434/v1",
        api_key_env="",
        max_retries=0,
    )

    assert adapter.generate("hello") == "local response"
    assert seen_authorization == [None]


def test_rate_limit_retries_then_succeeds(monkeypatch):
    attempts = 0

    def fake_urlopen(request, timeout):
        nonlocal attempts
        del request, timeout
        attempts += 1
        if attempts == 1:
            raise _rate_limit_error()
        return _FakeResponse("recovered")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("invariantlab.models.adapter.time.sleep", lambda _: None)
    adapter = OpenAICompatibleAdapter(
        model_id="provider/test",
        base_url="https://example.test/v1",
        api_key_env="",
        max_retries=2,
        backoff_initial_seconds=0,
        backoff_max_seconds=0,
    )

    assert adapter.generate("hello") == "recovered"
    assert attempts == 2


def test_rate_limit_becomes_typed_error_after_retries(monkeypatch):
    def fake_urlopen(request, timeout):
        del request, timeout
        raise _rate_limit_error()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("invariantlab.models.adapter.time.sleep", lambda _: None)
    adapter = OpenAICompatibleAdapter(
        model_id="provider/test",
        base_url="https://example.test/v1",
        api_key_env="",
        max_retries=1,
        backoff_initial_seconds=0,
        backoff_max_seconds=0,
    )

    with pytest.raises(ModelRateLimitError):
        adapter.generate("hello")


def test_openai_compatible_config_requires_explicit_base_url():
    config = ModelConfig(
        adapter="openai_compatible",
        model_id="provider/test",
        extra={},
    )

    with pytest.raises(ValueError, match=r"extra\.base_url"):
        build_adapter(config)


def _ollama_payload(**extra):
    payload = {"message": {"role": "assistant", "content": "ollama response"}}
    payload.update(extra)
    return payload


def test_ollama_success_reports_usage(monkeypatch):
    seen = []

    def fake_urlopen(request, timeout):
        seen.append((request.full_url, json.loads(request.data), timeout))
        return _FakeResponse(
            payload=_ollama_payload(prompt_eval_count=12, eval_count=34, done_reason="stop")
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    adapter = OllamaAdapter(model_id="qwen:test", base_url="http://localhost:11434/")

    response = adapter.complete("hello")

    assert response == ModelResponse(
        text="ollama response", input_tokens=12, output_tokens=34, finish_reason="stop"
    )
    url, body, timeout = seen[0]
    assert url == "http://localhost:11434/api/chat"
    assert body["stream"] is False
    assert body["messages"][1] == {"role": "user", "content": "hello"}
    assert timeout == 300.0


def test_ollama_missing_usage_fields_become_none(monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout: _FakeResponse(payload=_ollama_payload())
    )

    response = OllamaAdapter(model_id="qwen:test").complete("hello")

    assert response.text == "ollama response"
    assert response.input_tokens is None
    assert response.output_tokens is None
    assert response.finish_reason is None
    assert OllamaAdapter(model_id="qwen:test").generate("hello") == "ollama response"


def test_ollama_rate_limit_retries_then_succeeds(monkeypatch, no_sleep):
    attempts = 0

    def fake_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _rate_limit_error()
        return _FakeResponse(payload=_ollama_payload(eval_count=5))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    adapter = OllamaAdapter(model_id="qwen:test", max_retries=2, backoff_initial_seconds=0.5)

    assert adapter.complete("hello").output_tokens == 5
    assert attempts == 2
    assert no_sleep == [0.5]


def test_ollama_connection_failure_after_retries(monkeypatch, no_sleep):
    attempts = 0

    def fake_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    adapter = OllamaAdapter(
        model_id="qwen:test", max_retries=2, backoff_initial_seconds=1.0, backoff_max_seconds=1.5
    )

    with pytest.raises(ModelConnectionError, match="after 3 attempts"):
        adapter.generate("hello")
    assert attempts == 3
    assert no_sleep == [1.0, 1.5]


def test_ollama_non_retryable_status_fails_immediately(monkeypatch, no_sleep):
    attempts = 0

    def fake_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        raise _http_error(404, "model not found")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ModelRequestError, match="HTTP 404"):
        OllamaAdapter(model_id="missing", max_retries=3).complete("hello")
    assert attempts == 1
    assert no_sleep == []


def test_ollama_unexpected_response(monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout: _FakeResponse(payload={"error": "x"})
    )

    with pytest.raises(ModelRequestError, match="Unexpected Ollama response"):
        OllamaAdapter(model_id="qwen:test").complete("hello")


def test_openai_compatible_parses_usage_and_finish_reason(monkeypatch):
    payload = {
        "choices": [{"message": {"content": "patched"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
    }
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout: _FakeResponse(payload=payload)
    )
    adapter = OpenAICompatibleAdapter(
        model_id="provider/test", base_url="https://example.test/v1", max_retries=0
    )

    assert adapter.complete("hello") == ModelResponse(
        text="patched", input_tokens=100, output_tokens=25, finish_reason="length"
    )


def test_openai_compatible_missing_usage_becomes_none(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: _FakeResponse("ok"))
    adapter = OpenAICompatibleAdapter(
        model_id="provider/test", base_url="https://example.test/v1", max_retries=0
    )

    response = adapter.complete("hello")

    assert response == ModelResponse(text="ok")


def test_openai_compatible_sends_api_key_and_throttles(monkeypatch, no_sleep):
    seen_authorization = []

    def fake_urlopen(request, timeout):
        seen_authorization.append(request.get_header("Authorization"))
        return _FakeResponse("ok")

    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret-value")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    adapter = OpenAICompatibleAdapter(
        model_id="provider/test",
        base_url="https://example.test/v1",
        api_key_env="TEST_PROVIDER_KEY",
        min_request_interval_seconds=60.0,
    )

    adapter.generate("one")
    adapter.generate("two")

    assert seen_authorization == ["Bearer secret-value", "Bearer secret-value"]
    assert len(no_sleep) == 1
    assert 0 < no_sleep[0] <= 60.0


def test_openai_compatible_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("TEST_MISSING_KEY", raising=False)
    adapter = OpenAICompatibleAdapter(
        model_id="provider/test",
        base_url="https://example.test/v1",
        api_key_env="TEST_MISSING_KEY",
    )

    with pytest.raises(RuntimeError, match="TEST_MISSING_KEY"):
        adapter.generate("hello")


def test_openai_compatible_honours_retry_after(monkeypatch, no_sleep):
    attempts = 0

    def fake_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _http_error(503, "Service Unavailable", retry_after="7")
        return _FakeResponse("ok")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    adapter = OpenAICompatibleAdapter(
        model_id="provider/test",
        base_url="https://example.test/v1",
        max_retries=1,
        backoff_initial_seconds=1.0,
        backoff_max_seconds=60.0,
    )

    assert adapter.generate("hello") == "ok"
    assert no_sleep == [7.0]


def test_openai_compatible_persistent_server_error(monkeypatch, no_sleep):
    def fake_urlopen(request, timeout):
        raise _http_error(500, "Internal Server Error", retry_after=None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    adapter = OpenAICompatibleAdapter(
        model_id="provider/test", base_url="https://example.test/v1", max_retries=1
    )

    with pytest.raises(ModelRequestError, match="HTTP 500 persisted after 2 attempts"):
        adapter.generate("hello")


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (b"not json", "invalid JSON"),
        (b"[1, 2]", "Unexpected Provider response"),
        (b'{"choices": []}', "Unexpected provider response"),
    ],
)
def test_openai_compatible_rejects_malformed_payloads(monkeypatch, raw, match):
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: _FakeResponse(raw=raw))
    adapter = OpenAICompatibleAdapter(
        model_id="provider/test", base_url="https://example.test/v1", max_retries=0
    )

    with pytest.raises(ModelRequestError, match=match):
        adapter.generate("hello")


def test_reference_stub_adapter_complete_has_no_usage():
    response = ReferenceStubAdapter().complete("anything")

    assert "def solve_oscillator_verlet" in response.text
    assert response.input_tokens is None
    assert ReferenceStubAdapter().generate("anything") == response.text


def test_build_adapter_reference_stub():
    adapter = build_adapter(ModelConfig(adapter="reference_stub"))

    assert isinstance(adapter, ReferenceStubAdapter)
    assert adapter.model_id == "reference_stub/oscillator-verlet"


def test_replay_adapter_loads_recorded_response_and_usage(tmp_path):
    prompt = "recorded prompt"
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "response": "recorded response",
                "model": "orig/model",
                "usage": {"input_tokens": 17, "output_tokens": 8},
                "finish_reason": "stop",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    adapter = ReplayAdapter.from_events(events_path)

    assert adapter.model_id == "replay/orig/model"
    assert adapter.complete(prompt) == ModelResponse(
        text="recorded response",
        input_tokens=17,
        output_tokens=8,
        finish_reason="stop",
    )


def test_replay_adapter_serves_duplicate_prompts_in_schedule_order(tmp_path):
    prompt_hash = hashlib.sha256(b"duplicate prompt").hexdigest()
    records = [
        {
            "prompt_sha256": prompt_hash,
            "response": "schedule three",
            "model": "orig/model",
            "schedule_index": 3,
            "trial": 2,
        },
        {
            "prompt_sha256": prompt_hash,
            "response": "schedule one",
            "model": "orig/model",
            "schedule_index": 1,
            "trial": 1,
        },
    ]
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    adapter = ReplayAdapter.from_events(events_path)

    assert adapter.generate("duplicate prompt") == "schedule one"
    assert adapter.generate("duplicate prompt") == "schedule three"
    with pytest.raises(ReplayMissError, match="2 recorded, 2 already served"):
        adapter.generate("duplicate prompt")


def test_replay_adapter_raises_for_unknown_prompt(tmp_path):
    prompt = "recorded prompt"
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "response": "recorded response",
                "model": "orig/model",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    adapter = ReplayAdapter.from_events(events_path)

    with pytest.raises(ReplayMissError, match="0 recorded, 0 already served"):
        adapter.complete("unknown prompt")


def test_build_adapter_replay_requires_events_path():
    with pytest.raises(ValueError, match=r"extra\.events_path"):
        build_adapter(ModelConfig(adapter="replay"))


def test_build_adapter_replay_loads_events_and_overrides_model_id(tmp_path):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "prompt_sha256": "a" * 64,
                "response": "response",
                "model": "orig/model",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    adapter = build_adapter(
        ModelConfig(
            adapter="replay",
            model_id="custom/replay",
            extra={"events_path": str(events_path)},
        )
    )

    assert isinstance(adapter, ReplayAdapter)
    assert adapter.model_id == "custom/replay"


def test_build_adapter_replay_uses_derived_model_id(tmp_path):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "prompt_sha256": "b" * 64,
                "response": "response",
                "model": "orig/model",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    adapter = build_adapter(ModelConfig(adapter="replay", extra={"events_path": str(events_path)}))

    assert adapter.model_id == "replay/orig/model"


def test_replay_adapter_requires_explicit_model_id_without_recorded_models(tmp_path):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        json.dumps({"prompt_sha256": "c" * 64, "response": "response"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="explicit model_id"):
        ReplayAdapter.from_events(events_path)


def test_build_adapter_ollama_reads_retry_settings():
    adapter = build_adapter(
        ModelConfig(
            adapter="ollama",
            model_id="qwen:test",
            temperature=0.2,
            max_tokens=128,
            extra={
                "base_url": "http://ollama.test:11434",
                "request_timeout_seconds": 10,
                "max_retries": 7,
                "backoff_initial_seconds": 0.25,
                "backoff_max_seconds": 4,
            },
        )
    )

    assert adapter == OllamaAdapter(
        model_id="qwen:test",
        base_url="http://ollama.test:11434",
        temperature=0.2,
        max_tokens=128,
        request_timeout_seconds=10.0,
        max_retries=7,
        backoff_initial_seconds=0.25,
        backoff_max_seconds=4.0,
    )


def test_build_adapter_openai_compatible():
    adapter = build_adapter(
        ModelConfig(
            adapter="openai_compatible",
            model_id="provider/test",
            extra={
                "base_url": "https://example.test/v1",
                "api_key_env": "KEY",
                "max_retries": 2,
                "min_request_interval_seconds": 1.5,
            },
        )
    )

    assert isinstance(adapter, OpenAICompatibleAdapter)
    assert adapter.base_url == "https://example.test/v1"
    assert adapter.api_key_env == "KEY"
    assert adapter.max_retries == 2
    assert adapter.min_request_interval_seconds == 1.5


def test_build_adapter_unknown_raises():
    config = ModelConfig(adapter="nope", model_id="x")

    with pytest.raises(ValueError, match="Unsupported model adapter"):
        build_adapter(config)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_resolve_model_id_matches_built_adapter():
    assert (
        resolve_model_id(ModelConfig(adapter="reference_stub"))
        == "reference_stub/oscillator-verlet"
    )

    events_path = (
        REPO_ROOT / "tests/fixtures/replay/oscillator-update-order/events.jsonl"
    ).as_posix()
    replay_config = ModelConfig(adapter="replay", extra={"events_path": events_path})

    assert resolve_model_id(replay_config) == "replay/fixture/scripted-oscillator"
