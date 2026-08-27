from __future__ import annotations

import json

import yaml
from typer.testing import CliRunner

from adh.cli import app
from adh.config import resolve_adh_config
from adh.doctor import all_passed, run_checks
from adh.models import DEFAULT_MODEL

runner = CliRunner()


def test_doctor_fast_profile_all_green_no_keys(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    cfg = resolve_adh_config(profile="fast")
    results = run_checks(cfg)
    assert all_passed(results)
    names = {item.name for item in results}
    assert "python_version" in names
    assert "detector_registry" in names
    assert "rewriter" in names
    assert "local_models" in names
    assert "verify_keys" in names
    rewriter = next(item for item in results if item.name == "rewriter")
    assert rewriter.skipped
    assert "identity" in rewriter.message


def test_doctor_standard_fails_without_rewriter_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    cfg = resolve_adh_config(profile="standard")
    results = run_checks(cfg)
    assert not all_passed(results)
    key_check = next(item for item in results if item.name == "rewriter_api_key")
    assert key_check.ok is False
    assert "OPENAI_API_KEY" in key_check.message
    assert key_check.fix is not None
    assert "SETUP.md" in key_check.fix


def test_doctor_reports_missing_local_model(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    cfg = resolve_adh_config(
        profile="standard",
        values={"models_dir": tmp_path},
        explicit={"models_dir"},
    )
    results = run_checks(cfg)
    model_check = next(
        item for item in results if item.name == f"local_model_{DEFAULT_MODEL}"
    )
    assert model_check.ok is False
    assert "missing" in model_check.message.lower()
    assert model_check.fix is not None
    assert f"adh models fetch --model {DEFAULT_MODEL}" in model_check.fix


def test_doctor_exit_code_1_on_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    (tmp_path / "adh.yaml").write_text("profile: standard\n", encoding="utf-8")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "rewriter_api_key" in result.stdout
    assert "FAIL" in result.stdout


def test_doctor_exit_code_0_for_fast_profile(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    (tmp_path / "adh.yaml").write_text("profile: fast\n", encoding="utf-8")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "identity" in result.stdout


def test_doctor_json_output(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    (tmp_path / "adh.yaml").write_text("profile: fast\n", encoding="utf-8")
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload
    sample = payload[0]
    assert set(sample) == {"name", "ok", "message", "fix"}
    assert isinstance(sample["ok"], bool)
    assert isinstance(sample["message"], str)


def test_doctor_verify_keys_required(monkeypatch) -> None:
    monkeypatch.delenv("PANGRAM_API_KEY", raising=False)
    cfg = resolve_adh_config(
        profile="fast",
        values={"verify_detectors": ["pangram"]},
        explicit={"verify_detectors"},
    )
    results = run_checks(cfg)
    verify = next(item for item in results if item.name == "verify_keys")
    assert verify.ok is False
    assert "PANGRAM_API_KEY" in verify.message


def test_doctor_help() -> None:
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "--json" in result.stdout


def test_doctor_cli_profile_override(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    (tmp_path / "adh.yaml").write_text("profile: standard\n", encoding="utf-8")
    result = runner.invoke(app, ["doctor", "--profile", "fast"])
    assert result.exit_code == 0, result.output
    assert "identity" in result.stdout
