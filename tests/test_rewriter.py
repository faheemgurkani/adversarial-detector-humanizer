from __future__ import annotations

import json

import httpx
import pytest

from adh.exceptions import InputError, RewriterError
from adh.rewriter import IdentityRewriter, OpenAICompatibleRewriter, ScriptedRewriter


def test_identity_rejects_empty() -> None:
    with pytest.raises(InputError):
        IdentityRewriter().rewrite("  ")


def test_scripted_missing_mapping() -> None:
    rewriter = ScriptedRewriter({"a": ["b"]})
    with pytest.raises(RewriterError):
        rewriter.rewrite("missing")


def test_openai_compatible_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(RewriterError, match="OPENAI_API_KEY"):
        OpenAICompatibleRewriter(api_key="", base_url="https://api.openai.com/v1")


def test_openai_compatible_parses_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        status_code = 200
        text = "{}"

        def json(self):
            return {
                "choices": [
                    {"message": {"content": '"Rewritten sentence."'}},
                    {"message": {"content": "Second take."}},
                ]
            }

    def fake_post(*_args, **_kwargs):
        return _Response()

    monkeypatch.setattr(httpx, "post", fake_post)
    rewriter = OpenAICompatibleRewriter(
        api_key="test",
        base_url="https://example.test/v1",
        model="dummy",
    )
    texts = rewriter.rewrite("Original sentence.", n=2)
    assert texts[0] == "Rewritten sentence."
    assert texts[1] == "Second take."


def test_openai_compatible_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        status_code = 401
        text = "nope"

        def json(self):
            return {}

    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: _Response())
    rewriter = OpenAICompatibleRewriter(api_key="x", base_url="https://example.test/v1")
    with pytest.raises(RewriterError, match="401"):
        rewriter.rewrite("Hello there.")


def test_openai_compatible_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        status_code = 200
        text = "not-json"

        def json(self):
            raise json.JSONDecodeError("bad", "not-json", 0)

    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: _Response())
    rewriter = OpenAICompatibleRewriter(api_key="x", base_url="https://example.test/v1")
    with pytest.raises(RewriterError, match="non-JSON"):
        rewriter.rewrite("Hello there.")
