from __future__ import annotations

import re
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from adh.api import create_app
from adh.errors import ERROR_CODES
from adh.exceptions import InputError
from adh.rewriter import IdentityRewriter
from adh.schemas import compact_from_report
from adh.semantic import LexicalSemanticGate
from tests.conftest import CueDetector
from tests.test_engine import StripCueRewriter

runner = CliRunner()
REPORT_ID = re.compile(r"^report_[0-9a-f]{16}$")
REQUEST_ID = re.compile(r"^req_[0-9a-f]{16}$")


def _client(**kwargs) -> TestClient:
    from adh.detectors.fake import FakeDetector

    app = create_app(
        detector=kwargs.get("detector", FakeDetector(document_score=80.0)),
        rewriter=kwargs.get("rewriter", IdentityRewriter()),
        semantic_gate=kwargs.get("gate", LexicalSemanticGate()),
        default_detector="fake",
        idempotency_store=kwargs.get("idempotency_store"),
    )
    return TestClient(app)


def test_humanize_response_includes_report_id() -> None:
    response = _client().post(
        "/v1/humanize",
        json={"text": "Furthermore, note this.", "profile": "fast"},
    )
    assert response.status_code == 200
    body = response.json()
    assert REPORT_ID.match(body["report_id"])


def test_request_id_header_on_all_routes() -> None:
    client = _client()
    for method, path, payload in (
        ("GET", "/health", None),
        (
            "POST",
            "/v1/score",
            {"text": "A complete sentence for scoring.", "detector": "fake"},
        ),
    ):
        if method == "GET":
            response = client.get(path)
        else:
            response = client.post(path, json=payload)
        assert response.status_code == 200
        assert REQUEST_ID.match(response.headers["X-Request-Id"])


def test_unknown_detector_structured_error() -> None:
    app = create_app(
        detector=None,
        rewriter=IdentityRewriter(),
        semantic_gate=LexicalSemanticGate(),
        default_detector="fake",
    )
    client = TestClient(app)
    response = client.post(
        "/v1/score",
        json={"text": "Hello there world.", "detector": "nope"},
    )
    assert response.status_code == 422
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "unknown_detector"
    assert REQUEST_ID.match(body["error"]["request_id"])
    assert body["error"]["doc_url"].startswith("https://")


def test_pangram_inner_loop_structured_error() -> None:
    response = _client().post(
        "/v1/humanize",
        json={"text": "A reasonably long sentence for the stub.", "detector": "pangram"},
    )
    assert response.status_code == 501
    body = response.json()
    assert body["error"]["code"] == "remote_detector_unsupported"


def test_cli_json_error_matches_http_shape(monkeypatch) -> None:
    from adh.cli import app

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    result = runner.invoke(
        app,
        [
            "humanize",
            "--json",
            "--text",
            "Furthermore, the method is important to note.",
        ],
    )
    assert result.exit_code == 1
    body = __import__("json").loads(result.stdout)
    assert body["error"]["code"] == "rewriter_unavailable"
    assert "request_id" in body["error"]


def test_rewriter_missing_key_error(monkeypatch) -> None:
    from adh.detectors.fake import FakeDetector

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client = TestClient(
        create_app(
            detector=FakeDetector(document_score=80.0),
            rewriter=None,
            semantic_gate=LexicalSemanticGate(),
            default_detector="fake",
        )
    )
    response = client.post(
        "/v1/humanize",
        json={"text": "Furthermore, note this.", "detector": "fake"},
    )
    assert response.status_code == 502
    body = response.json()
    assert body["error"]["code"] == "rewriter_unavailable"
    assert body["error"]["retryable"] is False


def test_error_codes_documented_in_backend_prd() -> None:
    from pathlib import Path

    text = Path(__file__).resolve().parents[1].joinpath("docs/BACKEND_PRD.md").read_text(
        encoding="utf-8"
    )
    for code in ERROR_CODES:
        assert code in text


def test_structured_error_from_input_error() -> None:
    from adh.errors import error_response
    from adh.ids import new_request_id

    payload = error_response(InputError("bad input"), new_request_id())
    assert payload["code"] == "invalid_input"
    assert payload["retryable"] is False
