"""Minimal run config. Yaml loading is added in a later step."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from adh.models import DEFAULT_MODEL
from adh.profiles import get_profile_preset

_ADH_FIELD_NAMES = None


def _adh_field_names() -> set[str]:
    global _ADH_FIELD_NAMES
    if _ADH_FIELD_NAMES is None:
        _ADH_FIELD_NAMES = {item.name for item in fields(AdhConfig)}
    return _ADH_FIELD_NAMES


@dataclass
class AdhConfig:
    """Caller-facing knobs shared by CLI, HTTP, and later doors."""

    profile: str | None = None
    detector: str = DEFAULT_MODEL
    device: str = "auto"
    models_dir: Path | str | None = None
    rewriter: str = "openai"
    rewriter_model: str | None = None
    target_score: float = 30.0
    verdict_score: float = 45.0
    max_rounds: int = 5
    sentence_threshold: float = 50.0
    min_semantic_similarity: float = 0.88
    max_rewrite_ratio: float = 0.4
    best_of_n: int = 3
    semantic: str = "auto"
    allow_lexical_gate: bool = False
    meaning_gate_mode: str = "auto"
    verify_detectors: list[str] = field(default_factory=list)
    verify_threshold: float = 45.0
    deploy_detectors: list[str] = field(default_factory=list)
    enable_logprob_blend: bool = True
    logprob_blend_weight: float = 0.15
    hard_mode: bool = False
    hard_mode_max_sentences: int = 1
    prepass: str = "none"
    prepass_lang: str = "fi"
    prepass_max_paragraphs: int = 2
    prepass_backend: str = "llm"


def _from_mapping(data: dict[str, Any]) -> AdhConfig:
    allowed = _adh_field_names()
    filtered = {key: value for key, value in data.items() if key in allowed}
    for key in ("verify_detectors", "deploy_detectors"):
        if key in filtered and filtered[key] is not None:
            filtered[key] = list(filtered[key])
    return AdhConfig(**filtered)


def resolve_adh_config(
    *,
    profile: str | None = None,
    values: dict[str, Any] | None = None,
    explicit: set[str] | frozenset[str] | None = None,
) -> AdhConfig:
    """Build config from defaults, an optional profile, then explicit overrides.

    When ``profile`` is set, ``values`` keys apply only if they appear in
    ``explicit`` (CLI flags / JSON fields the caller actually sent).
    When ``profile`` is omitted, ``values`` are applied in full.
    """
    values = dict(values or {})
    data: dict[str, Any] = {}
    if profile:
        data.update(get_profile_preset(profile))
        data["profile"] = profile
        chosen = explicit if explicit is not None else set(values)
        for key, value in values.items():
            if key in chosen:
                data[key] = value
    else:
        data.update(values)
    return _from_mapping(data)
