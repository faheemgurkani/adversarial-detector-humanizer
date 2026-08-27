"""Shared use-case layer. Doors parse input, this module runs the engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adh.config import AdhConfig, resolve_adh_config
from adh.detectors.base import Detector, ScoreResult
from adh.detectors.remote import assert_inner_loop_detector
from adh.engine import EngineConfig, humanize
from adh.factory import load_detector, load_gate, load_rewriter
from adh.ids import new_report_id
from adh.models import DEFAULT_MODEL
from adh.report import RunReport
from adh.rewriter import Rewriter
from adh.schemas import HumanizeRequest
from adh.semantic import SemanticGate

SERVICE_ONLY_HUMANIZE_FIELDS = frozenset(
    {
        "text",
        "compact",
        "device",
        "models_dir",
        "semantic",
        "profile",
        "metadata",
    }
)

HUMANIZE_REQUEST_ENGINE_FIELDS: dict[str, str] = {
    "detector": "detector",
    "target_score": "target_score",
    "verdict_score": "verdict_score",
    "max_rounds": "max_rounds",
    "sentence_threshold": "sentence_threshold",
    "min_semantic_similarity": "min_semantic_similarity",
    "max_rewrite_ratio": "max_rewrite_ratio",
    "best_of_n": "best_of_n",
    "rewriter_model": "rewriter_model",
    "allow_lexical_gate": "allow_lexical_gate",
    "meaning_gate_mode": "meaning_gate_mode",
    "verify": "verify_detectors",
    "verify_threshold": "verify_threshold",
    "deploy_detectors": "deploy_detectors",
    "enable_logprob_blend": "enable_logprob_blend",
    "logprob_blend_weight": "logprob_blend_weight",
    "hard_mode": "hard_mode",
    "hard_mode_max_sentences": "hard_mode_max_sentences",
    "prepass": "prepass",
    "prepass_lang": "prepass_lang",
    "prepass_max_paragraphs": "prepass_max_paragraphs",
    "prepass_backend": "prepass_backend",
}

_REQUEST_TO_ADH: dict[str, str] = {
    **HUMANIZE_REQUEST_ENGINE_FIELDS,
    "detector": "detector",
    "device": "device",
    "models_dir": "models_dir",
    "semantic": "semantic",
}


def adh_config_from_request(
    request: HumanizeRequest,
    *,
    file: AdhConfig | None = None,
) -> AdhConfig:
    """Map an HTTP body onto AdhConfig, honoring file defaults and explicit fields."""
    values: dict[str, Any] = {
        "detector": request.detector,
        "device": request.device,
        "models_dir": request.models_dir,
        "rewriter_model": request.rewriter_model,
        "target_score": request.target_score,
        "verdict_score": request.verdict_score,
        "max_rounds": request.max_rounds,
        "sentence_threshold": request.sentence_threshold,
        "min_semantic_similarity": request.min_semantic_similarity,
        "max_rewrite_ratio": request.max_rewrite_ratio,
        "best_of_n": request.best_of_n,
        "semantic": request.semantic,
        "allow_lexical_gate": request.allow_lexical_gate,
        "meaning_gate_mode": request.meaning_gate_mode,
        "verify_detectors": list(request.verify),
        "verify_threshold": request.verify_threshold,
        "deploy_detectors": list(request.deploy_detectors),
        "enable_logprob_blend": request.enable_logprob_blend,
        "logprob_blend_weight": request.logprob_blend_weight,
        "hard_mode": request.hard_mode,
        "hard_mode_max_sentences": request.hard_mode_max_sentences,
        "prepass": request.prepass,
        "prepass_lang": request.prepass_lang,
        "prepass_max_paragraphs": request.prepass_max_paragraphs,
        "prepass_backend": request.prepass_backend,
    }
    if request.profile is not None:
        values["profile"] = request.profile
    explicit: set[str] = set()
    for field_name in request.model_fields_set:
        if field_name in {"text", "compact"}:
            continue
        if field_name == "profile":
            explicit.add("profile")
            continue
        explicit.add(_REQUEST_TO_ADH.get(field_name, field_name))
    profile = request.profile if "profile" in request.model_fields_set else None
    return resolve_adh_config(
        profile=profile,
        values=values,
        explicit=explicit,
        file=file,
    )


def _engine_from_adh(config: AdhConfig) -> EngineConfig:
    return EngineConfig(
        target_score=config.target_score,
        verdict_score=config.verdict_score,
        max_rounds=config.max_rounds,
        sentence_threshold=config.sentence_threshold,
        min_semantic_similarity=config.min_semantic_similarity,
        max_rewrite_ratio=config.max_rewrite_ratio,
        best_of_n=config.best_of_n,
        rewriter_model=config.rewriter_model or "gpt-4o-mini",
        detector=config.detector,
        meaning_gate_mode=config.meaning_gate_mode,
        allow_lexical_gate=config.allow_lexical_gate,
        verify_detectors=list(config.verify_detectors),
        verify_threshold=config.verify_threshold,
        deploy_detectors=list(config.deploy_detectors),
        enable_logprob_blend=config.enable_logprob_blend,
        logprob_blend_weight=config.logprob_blend_weight,
        hard_mode=config.hard_mode,
        hard_mode_max_sentences=config.hard_mode_max_sentences,
        prepass=config.prepass,  # type: ignore[arg-type]
        prepass_lang=config.prepass_lang,
        prepass_max_paragraphs=config.prepass_max_paragraphs,
        prepass_backend=config.prepass_backend,
    )


def build_engine_config(
    source: HumanizeRequest | AdhConfig | EngineConfig,
    *,
    file: AdhConfig | None = None,
) -> EngineConfig:
    """Normalize door input into the engine's config object."""
    if isinstance(source, EngineConfig):
        return source
    if isinstance(source, HumanizeRequest):
        return _engine_from_adh(adh_config_from_request(source, file=file))
    return _engine_from_adh(source)


def _resolve_detector(
    name: str | None,
    *,
    injected: Detector | None,
    default_name: str,
    models_dir: Path | str | None,
    device: str,
) -> Detector:
    chosen = name or default_name
    if injected is not None and (name is None or name == injected.name):
        return injected
    return load_detector(chosen, models_dir=models_dir, device=device)


def run_score(
    text: str,
    *,
    detector_name: str | None = None,
    detector: Detector | None = None,
    device: str = "auto",
    models_dir: Path | str | None = None,
    default_detector: str = DEFAULT_MODEL,
) -> tuple[Detector, ScoreResult]:
    """Score text. ``detector`` is an optional injected adapter (tests / serve)."""
    loaded = _resolve_detector(
        detector_name,
        injected=detector,
        default_name=default_detector,
        models_dir=models_dir,
        device=device,
    )
    return loaded, loaded.score(text)


def run_humanize(
    text: str,
    *,
    config: AdhConfig | EngineConfig | HumanizeRequest | None = None,
    file: AdhConfig | None = None,
    detector: Detector | None = None,
    rewriter: Rewriter | None = None,
    semantic_gate: SemanticGate | None = None,
    hard_rewriter: Any = None,
    default_detector: str | None = None,
    device: str | None = None,
    models_dir: Path | str | None = None,
) -> RunReport:
    """Load adapters, build engine config, and run ``humanize()``."""
    adh: AdhConfig | None
    if isinstance(config, HumanizeRequest):
        adh = adh_config_from_request(config, file=file)
        settings = _engine_from_adh(adh)
    elif isinstance(config, EngineConfig):
        adh = None
        settings = config
    elif isinstance(config, AdhConfig):
        adh = config
        settings = _engine_from_adh(adh)
    else:
        adh = AdhConfig()
        settings = _engine_from_adh(adh)

    detector_name = adh.detector if adh is not None else settings.detector
    assert_inner_loop_detector(detector_name)

    loaded = _resolve_detector(
        detector_name,
        injected=detector,
        default_name=default_detector or detector_name,
        models_dir=(
            models_dir if models_dir is not None else (adh.models_dir if adh else None)
        ),
        device=device or (adh.device if adh else "auto"),
    )
    settings = settings.model_copy(update={"detector": loaded.name})

    if rewriter is not None:
        writer = rewriter
    else:
        backend = adh.rewriter if adh is not None else "openai"
        writer = load_rewriter(name=backend, model=adh.rewriter_model if adh else None)

    if semantic_gate is not None:
        gate = semantic_gate
    else:
        prefer = adh.semantic if adh is not None else "auto"
        gate = load_gate(prefer=prefer, allow_lexical=settings.allow_lexical_gate)

    guided = hard_rewriter
    if guided is None and settings.hard_mode:
        from adh.hard import HardModeRewriter

        guided = HardModeRewriter()

    return humanize(
        text,
        detector=loaded,
        rewriter=writer,
        semantic_gate=gate,
        hard_rewriter=guided,
        config=settings,
    ).model_copy(update={"report_id": new_report_id()})
