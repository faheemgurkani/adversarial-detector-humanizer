from __future__ import annotations

import json

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from adh.api import create_app
from adh.cli import app
from adh.config import resolve_adh_config
from adh.detectors.fake import FakeDetector
from adh.exceptions import InputError, RewriterError
from adh.factory import load_rewriter
from adh.profiles import TRY_SAMPLE_TEXT
from adh.rewriter import IdentityRewriter
from adh.semantic import LexicalSemanticGate

runner = CliRunner()

SAMPLE = "Furthermore, note this."


def test_identity_rewriter_loads_without_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    rewriter = load_rewriter(name="identity")
    assert rewriter.name == "identity"
    assert rewriter.rewrite(SAMPLE) == [SAMPLE]


def test_default_rewriter_still_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    try:
        load_rewriter()
    except RewriterError:
        return
    raise AssertionError("expected RewriterError when OPENAI_API_KEY is missing")


def test_unknown_profile_raises() -> None:
    try:
        resolve_adh_config(profile="nope")
    except InputError as error:
        assert "unknown profile" in str(error)
        return
    raise AssertionError("expected InputError")


def test_fast_profile_no_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    result = runner.invoke(
        app,
        ["humanize", "--profile", "fast", "--text", SAMPLE, "--json"],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert "stop_reason" in body
    assert body["detector"] == "fake"


def test_try_command_exits_zero(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    result = runner.invoke(app, ["try"])
    assert result.exit_code == 0, result.output
    assert "stop_reason" in result.stdout
    assert "output_text" in result.stdout
    body = json.loads(result.stdout)
    assert body["input_text"] == TRY_SAMPLE_TEXT


def test_fast_profile_uses_fake_detector(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = runner.invoke(
        app,
        ["humanize", "--profile", "fast", "--text", SAMPLE, "--json"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["detector"] == "fake"


def test_profile_overridden_by_explicit_flag(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    result = runner.invoke(
        app,
        [
            "humanize",
            "--profile",
            "fast",
            "--detector",
            "statistical",
            "--text",
            SAMPLE,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["detector"] == "statistical"


def test_fast_api_humanize_no_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client = TestClient(
        create_app(
            detector=FakeDetector(),
            rewriter=IdentityRewriter(),
            semantic_gate=LexicalSemanticGate(),
            default_detector="fake",
        )
    )
    response = client.post(
        "/v1/humanize",
        json={"text": SAMPLE, "detector": "fake", "compact": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert "stop_reason" in body
    assert body["detector"] == "fake"


def test_fast_api_profile_no_key_without_injection(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client = TestClient(create_app())
    response = client.post(
        "/v1/humanize",
        json={"text": SAMPLE, "profile": "fast", "compact": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["detector"] == "fake"
    assert "stop_reason" in body
