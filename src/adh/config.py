"""Load and merge AdhConfig from adh.yaml, profiles, CLI, and HTTP."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from adh.exceptions import InputError
from adh.models import DEFAULT_MODEL
from adh.profiles import PROFILE_PRESETS, apply_profile, get_profile_preset

DEFAULT_CONFIG_FILENAME = "adh.yaml"
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "adh.yaml"

# Nested yaml keys → flat AdhConfig field names (API review gate helper).
YAML_TO_ADH: dict[str, str] = {
    "profile": "profile",
    "detector": "detector",
    "device": "device",
    "models_dir": "models_dir",
    "deploy_detectors": "deploy_detectors",
    "rewriter.backend": "rewriter",
    "rewriter.model": "rewriter_model",
    "humanize.target_score": "target_score",
    "humanize.verdict_score": "verdict_score",
    "humanize.max_rounds": "max_rounds",
    "humanize.sentence_threshold": "sentence_threshold",
    "humanize.min_semantic_similarity": "min_semantic_similarity",
    "humanize.max_rewrite_ratio": "max_rewrite_ratio",
    "humanize.best_of_n": "best_of_n",
    "humanize.prepass": "prepass",
    "humanize.prepass_lang": "prepass_lang",
    "humanize.prepass_max_paragraphs": "prepass_max_paragraphs",
    "humanize.prepass_backend": "prepass_backend",
    "humanize.hard_mode": "hard_mode",
    "humanize.hard_mode_max_sentences": "hard_mode_max_sentences",
    "humanize.semantic": "semantic",
    "humanize.allow_lexical_gate": "allow_lexical_gate",
    "humanize.meaning_gate_mode": "meaning_gate_mode",
    "humanize.enable_logprob_blend": "enable_logprob_blend",
    "humanize.logprob_blend_weight": "logprob_blend_weight",
    "verify.detectors": "verify_detectors",
    "verify.threshold": "verify_threshold",
}


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
    allowed = {item.name for item in fields(AdhConfig)}
    filtered = {key: value for key, value in data.items() if key in allowed}
    for key in ("verify_detectors", "deploy_detectors"):
        if key in filtered and filtered[key] is not None:
            filtered[key] = list(filtered[key])
    if filtered.get("models_dir") in ("", "null", None):
        filtered["models_dir"] = None
    return AdhConfig(**filtered)


def _flatten_yaml(raw: dict[str, Any]) -> dict[str, Any]:
    """Map nested adh.yaml sections onto flat AdhConfig keys."""
    flat: dict[str, Any] = {}
    if not isinstance(raw, dict):
        raise InputError("adh.yaml root must be a mapping")

    for key, value in raw.items():
        if key in {"rewriter", "humanize", "verify"}:
            continue
        flat[key] = value

    rewriter = raw.get("rewriter")
    if isinstance(rewriter, dict):
        if "backend" in rewriter:
            flat["rewriter"] = rewriter["backend"]
        if "model" in rewriter:
            flat["rewriter_model"] = rewriter["model"]

    humanize = raw.get("humanize")
    if isinstance(humanize, dict):
        mapping = {
            "target_score": "target_score",
            "verdict_score": "verdict_score",
            "max_rounds": "max_rounds",
            "sentence_threshold": "sentence_threshold",
            "min_semantic_similarity": "min_semantic_similarity",
            "max_rewrite_ratio": "max_rewrite_ratio",
            "best_of_n": "best_of_n",
            "prepass": "prepass",
            "prepass_lang": "prepass_lang",
            "prepass_max_paragraphs": "prepass_max_paragraphs",
            "prepass_backend": "prepass_backend",
            "hard_mode": "hard_mode",
            "hard_mode_max_sentences": "hard_mode_max_sentences",
            "semantic": "semantic",
            "allow_lexical_gate": "allow_lexical_gate",
            "meaning_gate_mode": "meaning_gate_mode",
            "enable_logprob_blend": "enable_logprob_blend",
            "logprob_blend_weight": "logprob_blend_weight",
        }
        for yaml_key, adh_key in mapping.items():
            if yaml_key in humanize:
                flat[adh_key] = humanize[yaml_key]

    verify = raw.get("verify")
    if isinstance(verify, dict):
        if "detectors" in verify:
            flat["verify_detectors"] = list(verify["detectors"] or [])
        if "threshold" in verify:
            flat["verify_threshold"] = verify["threshold"]

    return flat


def find_config_path(*, cwd: Path | None = None) -> Path | None:
    """Resolve adh.yaml from ``ADH_CONFIG`` or the current working directory."""
    env_path = os.environ.get("ADH_CONFIG", "").strip()
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_file():
            return path.resolve()
        raise InputError(f"ADH_CONFIG points to a missing file: {path}")

    candidate = (cwd or Path.cwd()) / DEFAULT_CONFIG_FILENAME
    if candidate.is_file():
        return candidate.resolve()
    return None


def parse_yaml_config(text: str) -> dict[str, Any]:
    """Parse yaml text into flat AdhConfig kwargs."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise InputError(f"invalid adh.yaml: {error}") from error
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise InputError("adh.yaml root must be a mapping")
    return _flatten_yaml(raw)


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Read and flatten a config file."""
    if not path.is_file():
        raise InputError(f"config file not found: {path}")
    return parse_yaml_config(path.read_text(encoding="utf-8"))


def config_from_mapping(data: dict[str, Any]) -> AdhConfig:
    """Build AdhConfig applying profile preset then file/override values."""
    values = dict(data)
    profile = values.pop("profile", None)
    merged: dict[str, Any] = {}
    if profile:
        merged.update(get_profile_preset(str(profile)))
        merged["profile"] = profile
    merged.update(values)
    return _from_mapping(merged)


def load_config(path: Path | str | None = None, *, cwd: Path | None = None) -> AdhConfig | None:
    """Load adh.yaml if present. Returns ``None`` when no file is found."""
    chosen = Path(path).expanduser().resolve() if path is not None else find_config_path(cwd=cwd)
    if chosen is None:
        return None
    return config_from_mapping(load_yaml_file(chosen))


def init_config_path(destination: Path, *, force: bool = False) -> Path:
    """Write the bundled template to ``destination/adh.yaml``."""
    target = destination / DEFAULT_CONFIG_FILENAME
    if target.exists() and not force:
        raise InputError(f"{target.name} already exists (pass --force to overwrite)")
    if not TEMPLATE_PATH.is_file():
        raise InputError(f"template missing: {TEMPLATE_PATH}")
    target.write_text(TEMPLATE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def merge_cli_overrides(
    base: AdhConfig,
    overrides: dict[str, Any],
    *,
    explicit: set[str] | frozenset[str],
) -> AdhConfig:
    """Apply only explicitly provided CLI/API overrides onto ``base``."""
    data = asdict(base)
    for key in explicit:
        if key in overrides:
            data[key] = overrides[key]
    return _from_mapping(data)


def resolve_adh_config(
    *,
    profile: str | None = None,
    values: dict[str, Any] | None = None,
    explicit: set[str] | frozenset[str] | None = None,
    file: AdhConfig | None = None,
) -> AdhConfig:
    """Build config with precedence: profile preset → file → explicit overrides.

    When a profile is active, ``values`` keys apply only if they appear in
    ``explicit`` (CLI flags / JSON fields the caller actually sent).
    """
    values = dict(values or {})
    file_data = asdict(file) if file is not None else {}
    explicit_set = set(explicit or ())

    effective_profile = profile
    if effective_profile is None and file_data.get("profile"):
        effective_profile = str(file_data["profile"])
    if "profile" in explicit_set and profile is not None:
        effective_profile = profile

    merged: dict[str, Any] = {}
    if effective_profile:
        merged.update(get_profile_preset(effective_profile))
        merged["profile"] = effective_profile

    for key, value in file_data.items():
        if key == "profile":
            continue
        merged[key] = value

    chosen = explicit_set if effective_profile else explicit_set or set(values)
    for key, value in values.items():
        if key == "profile":
            if key in explicit_set:
                merged["profile"] = value
            continue
        if effective_profile:
            if key in chosen:
                merged[key] = value
        elif key in chosen or not explicit_set:
            merged[key] = value

    return _from_mapping(merged)


def list_profile_names() -> list[str]:
    return sorted(PROFILE_PRESETS)
