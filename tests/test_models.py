from __future__ import annotations

from pathlib import Path

import pytest

from adh.exceptions import DetectorNotReadyError, InputError
from adh.models import (
    DEFAULT_MODEL,
    artifact_status,
    default_cache_dir,
    ensure_artifact_ready,
    fetch_models,
    list_models,
    model_kind,
    model_registry,
)


def test_default_model_and_kinds() -> None:
    assert DEFAULT_MODEL == "qwen3-variable"
    assert model_kind("logreg") == "sklearn"
    assert model_kind("distilbert") == "encoder"
    assert model_kind("qwen3-variable") == "causal"
    with pytest.raises(InputError):
        model_kind("nope")


def test_registry_paths(tmp_path: Path) -> None:
    registry = model_registry(tmp_path)
    assert "distilbert" in registry
    ready, status = artifact_status(registry["logreg"])
    assert ready is False
    assert "missing" in status


def test_ensure_not_ready(tmp_path: Path) -> None:
    spec = model_registry(tmp_path)["logreg"]
    with pytest.raises(DetectorNotReadyError):
        ensure_artifact_ready(spec)


def test_list_models_includes_hub_ids() -> None:
    rows = list_models()
    names = {row["name"] for row in rows}
    assert "qwen3-variable" in names
    assert all("hub" in row for row in rows)


def test_fetch_unknown_model() -> None:
    with pytest.raises(InputError):
        fetch_models(["not-a-model"])


def test_cache_dir_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ADH_MODELS_DIR", str(tmp_path))
    assert default_cache_dir() == tmp_path.resolve()
