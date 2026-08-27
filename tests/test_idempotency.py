from __future__ import annotations

from typing import get_args

from fastapi.testclient import TestClient

from adh.api import create_app
from adh.idempotency import IdempotencyStore
from adh.report import RunReport, StopReason
from adh.rewriter import IdentityRewriter
from adh.semantic import LexicalSemanticGate
from tests.conftest import CueDetector
from tests.test_engine import StripCueRewriter


def _payload(**overrides: object) -> dict:
    body = {
        "text": "Furthermore, the method is important to note in 2024.",
        "detector": "cue",
        "compact": True,
        "target_score": 30,
        "min_semantic_similarity": 0.2,
        "semantic": "lexical",
        "allow_lexical_gate": True,
    }
    body.update(overrides)
    return body


def test_compact_true_by_default() -> None:
    client = TestClient(
        create_app(
            detector=CueDetector(),
            rewriter=StripCueRewriter(),
            semantic_gate=LexicalSemanticGate(),
            default_detector="cue",
        )
    )
    response = client.post("/v1/humanize", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert "ai_score_before" in body
    assert "sentences" not in body
    assert "agent_hint" in body


def test_compact_false_returns_full_report() -> None:
    client = TestClient(
        create_app(
            detector=CueDetector(),
            rewriter=StripCueRewriter(),
            semantic_gate=LexicalSemanticGate(),
            default_detector="cue",
        )
    )
    response = client.post("/v1/humanize", json=_payload(compact=False))
    assert response.status_code == 200
    body = response.json()
    assert "sentences" in body
    assert "locks" in body
    assert body["report_id"].startswith("report_")


def test_agent_hint_present_on_compact() -> None:
    from adh.hints import agent_hint_for

    for reason in get_args(StopReason):
        report = RunReport(
            input_text="in",
            output_text="out",
            detector="fake",
            score_before=90.0,
            score_after=20.0,
            semantic_similarity=0.9,
            rounds=1,
            stop_reason=reason,
            report_id="report_0123456789abcdef",
        )
        hint = agent_hint_for(report)
        assert hint.strip()


def test_metadata_round_trip() -> None:
    client = TestClient(
        create_app(
            detector=CueDetector(),
            rewriter=IdentityRewriter(),
            semantic_gate=LexicalSemanticGate(),
            default_detector="cue",
        )
    )
    response = client.post(
        "/v1/humanize",
        json={
            "text": "This is a human sentence.",
            "detector": "cue",
            "allow_lexical_gate": True,
            "semantic": "lexical",
            "min_semantic_similarity": 0.2,
            "metadata": {"ticket": "JIRA-123"},
        },
    )
    assert response.status_code == 200
    assert response.json()["metadata"] == {"ticket": "JIRA-123"}


def test_metadata_max_keys_rejected() -> None:
    client = TestClient(
        create_app(
            detector=CueDetector(),
            rewriter=IdentityRewriter(),
            semantic_gate=LexicalSemanticGate(),
            default_detector="cue",
        )
    )
    metadata = {f"k{i}": "v" for i in range(51)}
    response = client.post(
        "/v1/humanize",
        json={
            "text": "This is a human sentence.",
            "detector": "cue",
            "metadata": metadata,
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


def test_idempotency_same_key_same_body(monkeypatch) -> None:
    calls = {"count": 0}
    store = IdempotencyStore()
    client = TestClient(
        create_app(
            detector=CueDetector(),
            rewriter=IdentityRewriter(),
            semantic_gate=LexicalSemanticGate(),
            default_detector="cue",
            idempotency_store=store,
        )
    )

    def spy(*args, **kwargs):
        calls["count"] += 1
        from adh.service import run_humanize as original

        return original(*args, **kwargs)

    monkeypatch.setattr("adh.api.run_humanize", spy)

    body = {
        "text": "This is a human sentence.",
        "detector": "cue",
        "allow_lexical_gate": True,
        "semantic": "lexical",
        "min_semantic_similarity": 0.2,
    }
    headers = {"Idempotency-Key": "idem_test_1"}
    first = client.post("/v1/humanize", json=body, headers=headers)
    second = client.post("/v1/humanize", json=body, headers=headers)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["report_id"] == second.json()["report_id"]
    assert calls["count"] == 1


def test_idempotency_same_key_different_body() -> None:
    store = IdempotencyStore()
    client = TestClient(
        create_app(
            detector=CueDetector(),
            rewriter=IdentityRewriter(),
            semantic_gate=LexicalSemanticGate(),
            default_detector="cue",
            idempotency_store=store,
        )
    )
    headers = {"Idempotency-Key": "idem_test_2"}
    first = client.post(
        "/v1/humanize",
        json={
            "text": "This is a human sentence.",
            "detector": "cue",
            "allow_lexical_gate": True,
            "semantic": "lexical",
            "min_semantic_similarity": 0.2,
        },
        headers=headers,
    )
    assert first.status_code == 200
    second = client.post(
        "/v1/humanize",
        json={
            "text": "Different sentence entirely here.",
            "detector": "cue",
            "allow_lexical_gate": True,
            "semantic": "lexical",
            "min_semantic_similarity": 0.2,
        },
        headers=headers,
    )
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "idempotency_key_reused"


def test_output_alias_equals_output_text() -> None:
    client = TestClient(
        create_app(
            detector=CueDetector(),
            rewriter=IdentityRewriter(),
            semantic_gate=LexicalSemanticGate(),
            default_detector="cue",
        )
    )
    response = client.post(
        "/v1/humanize",
        json={
            "text": "This is a human sentence.",
            "detector": "cue",
            "allow_lexical_gate": True,
            "semantic": "lexical",
            "min_semantic_similarity": 0.2,
        },
    )
    body = response.json()
    assert body["output"] == body["output_text"]


def test_openapi_lists_stop_reasons() -> None:
    response = TestClient(create_app(default_detector="fake")).get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()["components"]["schemas"]["CompactHumanizeResponse"]
    enum_values = schema["properties"]["stop_reason"]["enum"]
    assert set(enum_values) == set(get_args(StopReason))


def test_stop_reason_enum_frozen_in_backend_prd() -> None:
    from pathlib import Path

    text = Path(__file__).resolve().parents[1].joinpath("docs/BACKEND_PRD.md").read_text(
        encoding="utf-8"
    )
    for reason in get_args(StopReason):
        assert reason in text
