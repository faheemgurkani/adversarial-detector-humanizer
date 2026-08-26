from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_env_example_documents_rewriter_and_reserved_keys() -> None:
    path = ROOT / ".env.example"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ADH_REWRITER_MODEL",
        "ADH_MODELS_DIR",
        "PANGRAM_API_KEY",
        "GPTZERO_API_KEY",
    ):
        assert key in text
    assert "not wired up yet" in text.lower() or "verification scoring" in text.lower()


def test_setup_doc_exists() -> None:
    path = ROOT / "docs" / "SETUP.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "python3.11 -m venv .venv" in text
    assert "cp .env.example .env" in text
    assert "adh models fetch" in text


def test_roadmap_doc_exists() -> None:
    path = ROOT / "docs" / "ROADMAP.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "adh.yaml" in text
    assert "entry_points" in text
    assert "Interface discipline" in text
    assert "Implementation playbook" in text
