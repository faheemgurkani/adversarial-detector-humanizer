from __future__ import annotations

from fastapi.testclient import TestClient

from adh.api import create_app
from adh.detectors.fake import FakeDetector
from adh.rewriter import IdentityRewriter, ScriptedRewriter
from adh.semantic import LexicalSemanticGate
from tests.conftest import CueDetector
from tests.test_engine import StripCueRewriter


def _client(**kwargs) -> TestClient:
    app = create_app(
        detector=kwargs.get("detector", FakeDetector(document_score=80.0)),
        rewriter=kwargs.get("rewriter", IdentityRewriter()),
        semantic_gate=kwargs.get("gate", LexicalSemanticGate()),
        default_detector="fake",
    )
    return TestClient(app)


def test_openapi_docs() -> None:
    response = _client().get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    for path in ("/health", "/v1/models", "/v1/score", "/v1/humanize", "/v1/sentences"):
        assert path in paths


def test_health() -> None:
    response = _client().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["detector"] == "fake"
    assert "version" in body


def test_list_models() -> None:
    response = _client().get("/v1/models")
    assert response.status_code == 200
    names = {row["name"] for row in response.json()["models"]}
    assert "qwen3-variable" in names
    assert "distilbert" in names


def test_score_ok() -> None:
    response = _client().post(
        "/v1/score",
        json={"text": "A complete sentence for scoring.", "detector": "fake"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["score"] == 80.0
    assert body["label"] == "ai-leaning"
    assert body["detector"] == "fake"


def test_score_empty_rejected() -> None:
    response = _client().post("/v1/score", json={"text": "   ", "detector": "fake"})
    assert response.status_code == 422


def test_score_unknown_detector() -> None:
    app = create_app(
        detector=None,
        rewriter=IdentityRewriter(),
        semantic_gate=LexicalSemanticGate(),
        default_detector="fake",
    )
    client = TestClient(app)
    response = client.post("/v1/score", json={"text": "Hello there world.", "detector": "nope"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unknown_detector"


def test_humanize_already_below_target() -> None:
    client = _client(detector=CueDetector(), rewriter=IdentityRewriter())
    response = client.post(
        "/v1/humanize",
        json={
            "text": "This is a human sentence.",
            "detector": "cue",
            "compact": False,
            "allow_lexical_gate": True,
            "semantic": "lexical",
            "min_semantic_similarity": 0.2,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "already_below_target"
    assert body["score_after"] == body["score_before"]


def test_humanize_compact_and_loop() -> None:
    client = _client(detector=CueDetector(), rewriter=StripCueRewriter())
    response = client.post(
        "/v1/humanize",
        json={
            "text": "Furthermore, the method is important to note in 2024.",
            "detector": "cue",
            "compact": True,
            "target_score": 30,
            "min_semantic_similarity": 0.2,
            "semantic": "lexical",
            "allow_lexical_gate": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "ai_score_before" in body
    assert body["ai_score_after"] < body["ai_score_before"]
    assert "Furthermore" not in body["output_text"]


def test_humanize_pangram_stub_is_not_found_as_loop_detector() -> None:
    response = _client().post(
        "/v1/humanize",
        json={"text": "A reasonably long sentence for the stub.", "detector": "pangram"},
    )
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "remote_detector_unsupported"


def test_sentences() -> None:
    response = _client().post(
        "/v1/sentences",
        json={"text": "First clause here. Second clause there."},
    )
    assert response.status_code == 200
    sentences = response.json()["sentences"]
    assert len(sentences) >= 2
    assert sentences[0]["start"] == 0


def test_sentences_empty() -> None:
    response = _client().post("/v1/sentences", json={"text": "   "})
    assert response.status_code == 422


def test_extra_fields_rejected() -> None:
    response = _client().post(
        "/v1/score",
        json={"text": "Hello there world.", "detector": "fake", "secret": "x"},
    )
    assert response.status_code == 422


def test_scripted_humanize_preserves_url() -> None:
    rewriter = ScriptedRewriter({})

    class KeepLocks(StripCueRewriter):
        name = "keep"

    client = _client(detector=CueDetector(), rewriter=KeepLocks())
    text = "Furthermore, see https://example.com/a in 2024."
    response = client.post(
        "/v1/humanize",
        json={
            "text": text,
            "detector": "cue",
            "min_semantic_similarity": 0.1,
            "semantic": "lexical",
            "allow_lexical_gate": True,
        },
    )
    assert response.status_code == 200
    assert "https://example.com/a" in response.json()["output_text"]
