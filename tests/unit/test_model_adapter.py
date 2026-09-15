"""Unit tests for resilient OpenAI-compatible model requests."""

import io
import json
import urllib.error
import urllib.request
from email.message import Message

import pytest

from invariantlab.models import (
    ModelRateLimitError,
    OpenAICompatibleAdapter,
)


class _FakeResponse:
    def __init__(self, content: str):
        self._payload = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return self._payload


def _rate_limit_error() -> urllib.error.HTTPError:
    headers = Message()
    headers["Retry-After"] = "0"
    return urllib.error.HTTPError(
        "http://localhost/v1/chat/completions",
        429,
        "Too Many Requests",
        headers,
        io.BytesIO(b"rate limited"),
    )


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
