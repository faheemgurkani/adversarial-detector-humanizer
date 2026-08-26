from __future__ import annotations

import json

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from adh.api import create_app
from adh.cli import app
from adh.config import AdhConfig, resolve_adh_config
from adh.engine import EngineConfig
from adh.profiles import PROFILE_PRESETS
from adh.schemas import HumanizeRequest
from adh.service import (
    HUMANIZE_REQUEST_ENGINE_FIELDS,
    SERVICE_ONLY_HUMANIZE_FIELDS,
    adh_config_from_request,
    build_engine_config,
    run_humanize,
    run_score,
)

runner = CliRunner()
TEXT = "Furthermore, note this."


def test_fast_profile_preset_keys() -> None:
    fast = PROFILE_PRESETS["fast"]
    assert fast["detector"] == "fake"
    assert fast["rewriter"] == "identity"
    assert fast["max_rounds"] == 1
    assert fast["allow_lexical_gate"] is True
    assert fast["semantic"] == "lexical"
    assert fast["meaning_gate_mode"] == "lexical"


def test_resolve_fast_profile() -> None:
    config = resolve_adh_config(profile="fast")
    assert isinstance(config, AdhConfig)
    assert config.detector == "fake"
    assert config.rewriter == "identity"
    assert config.max_rounds == 1
    assert config.allow_lexical_gate is True


def test_run_score_fake() -> None:
    loaded, result = run_score(
        "A complete sentence for scoring.",
        detector_name="fake",
    )
    assert loaded.name == "fake"
    assert result.score == 80.0


def test_run_humanize_fast_profile() -> None:
    report = run_humanize(TEXT, config=resolve_adh_config(profile="fast"))
    assert report.detector == "fake"
    assert report.stop_reason
    assert report.output_text


def test_build_engine_config_from_adh() -> None:
    config = resolve_adh_config(profile="fast")
    engine = build_engine_config(config)
    assert isinstance(engine, EngineConfig)
    assert engine.detector == "fake"
    assert engine.max_rounds == 1
    assert engine.allow_lexical_gate is True


def test_request_profile_fast_overrides_defaults() -> None:
    request = HumanizeRequest(text=TEXT, profile="fast")
    config = adh_config_from_request(request)
    assert config.detector == "fake"
    assert config.rewriter == "identity"
    assert config.max_rounds == 1
    assert config.semantic == "lexical"


def test_service_produces_same_report_as_cli(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    config = resolve_adh_config(profile="fast")
    service_report = run_humanize(TEXT, config=config)
    result = runner.invoke(
        app,
        ["humanize", "--profile", "fast", "--text", TEXT, "--json"],
    )
    assert result.exit_code == 0, result.output
    cli_body = json.loads(result.stdout)
    assert cli_body["stop_reason"] == service_report.stop_reason
    assert cli_body["score_before"] == service_report.score_before
    assert cli_body["score_after"] == service_report.score_after
    assert cli_body["output_text"] == service_report.output_text


def test_service_produces_same_report_as_api(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    service_report = run_humanize(TEXT, config=resolve_adh_config(profile="fast"))
    client = TestClient(create_app())
    response = client.post(
        "/v1/humanize",
        json={"text": TEXT, "profile": "fast", "compact": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == service_report.stop_reason
    assert body["ai_score_before"] == service_report.score_before
    assert body["ai_score_after"] == service_report.score_after
    assert body["output_text"] == service_report.output_text
    assert body["detector"] == service_report.detector


def test_engine_config_fields_match_humanize_request() -> None:
    request_fields = set(HumanizeRequest.model_fields)
    mapped = set(HUMANIZE_REQUEST_ENGINE_FIELDS)
    leftover = request_fields - SERVICE_ONLY_HUMANIZE_FIELDS - mapped
    assert leftover == set()
    extra = mapped - (request_fields - SERVICE_ONLY_HUMANIZE_FIELDS)
    assert extra == set()

    request = HumanizeRequest(
        text=TEXT,
        detector="fake",
        target_score=21.0,
        verdict_score=33.0,
        max_rounds=2,
        sentence_threshold=40.0,
        min_semantic_similarity=0.5,
        max_rewrite_ratio=0.3,
        best_of_n=1,
        rewriter_model="gpt-test",
        allow_lexical_gate=True,
        meaning_gate_mode="lexical",
        verify=["pangram"],
        verify_threshold=41.0,
        deploy_detectors=["statistical"],
        enable_logprob_blend=False,
        logprob_blend_weight=0.2,
        hard_mode=True,
        hard_mode_max_sentences=2,
        prepass="structural",
        prepass_lang="de",
        prepass_max_paragraphs=1,
        prepass_backend="google",
    )
    engine = build_engine_config(request)
    assert isinstance(engine, EngineConfig)
    for request_field, engine_field in HUMANIZE_REQUEST_ENGINE_FIELDS.items():
        request_value = getattr(request, request_field)
        engine_value = getattr(engine, engine_field)
        if request_field == "rewriter_model" and request_value is None:
            assert engine_value == "gpt-4o-mini"
            continue
        if request_field == "verify":
            assert engine_value == list(request_value)
            continue
        assert engine_value == request_value, request_field
