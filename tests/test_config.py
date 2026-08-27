from __future__ import annotations

import json
from dataclasses import fields

import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from adh.api import create_app
from adh.cli import app
from adh.config import (
    YAML_TO_ADH,
    AdhConfig,
    config_from_mapping,
    init_config_path,
    load_config,
    load_yaml_file,
    parse_yaml_config,
    resolve_adh_config,
)
from adh.models import DEFAULT_MODEL
from adh.profiles import apply_profile, get_profile_preset
from adh.schemas import HumanizeRequest
from adh.service import HUMANIZE_REQUEST_ENGINE_FIELDS, build_engine_config

runner = CliRunner()
TEXT = "Furthermore, note this."


def test_init_writes_adh_yaml(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    target = tmp_path / "adh.yaml"
    assert target.is_file()
    parsed = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert parsed["profile"] == "standard"
    assert parsed["detector"] == DEFAULT_MODEL


def test_init_refuses_existing_without_force(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    init_config_path(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 1


def test_load_config_from_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "adh.yaml").write_text(
        "profile: fast\ndetector: fake\n",
        encoding="utf-8",
    )
    cfg = load_config()
    assert cfg is not None
    assert cfg.profile == "fast"
    assert cfg.detector == "fake"
    assert cfg.rewriter == "identity"
    assert cfg.max_rounds == 1


def test_load_config_from_adh_config_env(tmp_path, monkeypatch) -> None:
    config_file = tmp_path / "team.yaml"
    config_file.write_text(
        "profile: fast\ndetector: fake\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ADH_CONFIG", str(config_file))
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg is not None
    assert cfg.profile == "fast"
    assert cfg.detector == "fake"


def test_parse_nested_yaml_rewriter_and_humanize() -> None:
    flat = parse_yaml_config(
        """
        profile: standard
        detector: statistical
        rewriter:
          backend: identity
          model: gpt-test
        humanize:
          target_score: 21
          max_rounds: 2
        verify:
          detectors: [pangram]
          threshold: 40
        deploy_detectors: [statistical]
        """
    )
    assert flat["profile"] == "standard"
    assert flat["detector"] == "statistical"
    assert flat["rewriter"] == "identity"
    assert flat["rewriter_model"] == "gpt-test"
    assert flat["target_score"] == 21
    assert flat["max_rounds"] == 2
    assert flat["verify_detectors"] == ["pangram"]
    assert flat["verify_threshold"] == 40
    assert flat["deploy_detectors"] == ["statistical"]


def test_cli_flag_overrides_yaml(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "adh.yaml").write_text(
        "profile: fast\nhumanize:\n  target_score: 30\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["humanize", "--text", TEXT, "--target", "20", "--json"],
    )
    assert result.exit_code == 0, result.output
    engine = build_engine_config(
        resolve_adh_config(
            values={"target_score": 20.0},
            explicit={"target_score"},
            file=load_config(),
        )
    )
    assert engine.target_score == 20.0


def test_humanize_uses_yaml_without_repeating_flags(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "adh.yaml").write_text("profile: fast\n", encoding="utf-8")
    result = runner.invoke(app, ["humanize", "--text", TEXT, "--json"])
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["detector"] == "fake"


def test_serve_loads_same_config(tmp_path, monkeypatch) -> None:
    config_file = tmp_path / "adh.yaml"
    config_file.write_text("profile: fast\ndetector: fake\n", encoding="utf-8")
    application = create_app(config_path=config_file)
    client = TestClient(application)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["detector"] == "fake"


def test_profile_standard_fields() -> None:
    preset = apply_profile("standard")
    assert preset["detector"] == DEFAULT_MODEL
    assert preset["max_rounds"] == 5
    assert preset["rewriter"] == "openai"
    cfg = config_from_mapping(preset)
    assert cfg.detector == DEFAULT_MODEL
    assert cfg.max_rounds == 5


def test_profile_quality_fields() -> None:
    cfg = config_from_mapping(apply_profile("quality"))
    assert cfg.detector == "ensemble-local"
    assert cfg.max_rounds == 5


def test_profile_verify_only_fields() -> None:
    cfg = config_from_mapping(apply_profile("verify-only"))
    assert cfg.detector == DEFAULT_MODEL
    assert cfg.rewriter == "identity"
    assert cfg.max_rounds == 1


def test_unknown_profile_raises() -> None:
    try:
        get_profile_preset("nope")
    except Exception as error:
        assert "unknown profile" in str(error)
        assert "fast" in str(error)
        return
    raise AssertionError("expected InputError")


def test_api_accepts_profile_field(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client = TestClient(create_app())
    response = client.post(
        "/v1/humanize",
        json={"text": TEXT, "profile": "fast", "compact": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["detector"] == "fake"
    assert "stop_reason" in body


def test_config_field_names_match_humanize_request() -> None:
    adh_fields = {item.name for item in fields(AdhConfig)}
    mapped_yaml_targets = set(YAML_TO_ADH.values())
    for target in mapped_yaml_targets:
        assert target in adh_fields, target

    request_fields = set(HumanizeRequest.model_fields)
    for request_field, engine_field in HUMANIZE_REQUEST_ENGINE_FIELDS.items():
        assert engine_field in adh_fields, engine_field
        assert request_field in request_fields, request_field

    assert YAML_TO_ADH["humanize.target_score"] == "target_score"
    assert YAML_TO_ADH["verify.detectors"] == "verify_detectors"
    assert YAML_TO_ADH["rewriter.backend"] == "rewriter"


def test_load_yaml_file_round_trip(tmp_path) -> None:
    path = tmp_path / "adh.yaml"
    path.write_text("profile: fast\ndetector: fake\n", encoding="utf-8")
    flat = load_yaml_file(path)
    cfg = config_from_mapping(flat)
    assert cfg.detector == "fake"
    assert cfg.rewriter == "identity"
