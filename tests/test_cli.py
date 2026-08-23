from __future__ import annotations

from typer.testing import CliRunner

from adh.cli import app

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "humanize" in result.stdout
    assert "score" in result.stdout


def test_models_list() -> None:
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "qwen3-variable" in result.stdout


def test_score_requires_input() -> None:
    result = runner.invoke(app, ["score"])
    assert result.exit_code == 1
    assert "error" in result.stdout.lower() or "error" in result.output.lower()


def test_score_fake_json() -> None:
    result = runner.invoke(
        app,
        ["score", "--detector", "fake", "--json", "--text", "A complete sentence for scoring."],
    )
    assert result.exit_code == 0
    assert "score" in result.stdout


def test_humanize_without_key_fails(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    result = runner.invoke(
        app,
        [
            "humanize",
            "--detector",
            "fake",
            "--allow-lexical-gate",
            "--semantic",
            "lexical",
            "--text",
            "Furthermore, the method is important to note.",
        ],
    )
    assert result.exit_code == 1
