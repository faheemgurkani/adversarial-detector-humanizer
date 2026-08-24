"""Local FastAPI surface. Hosted SaaS must keep calling this same engine."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from adh import __version__
from adh.detectors.base import Detector, ScoreResult
from adh.engine import EngineConfig, humanize
from adh.exceptions import (
    AdhError,
    DetectorNotReadyError,
    InputError,
    PreserveLockError,
    RemoteDetectorUnavailableError,
    RewriterError,
    SemanticBackendError,
)
from adh.factory import load_detector, load_gate, load_rewriter
from adh.models import DEFAULT_MODEL, list_models
from adh.report import RunReport
from adh.rewriter import Rewriter
from adh.schemas import (
    CompactHumanizeResponse,
    HealthResponse,
    HumanizeRequest,
    ScoreRequest,
    ScoreResponse,
    SentenceSplitRequest,
    SentenceSplitResponse,
    compact_from_report,
)
from adh.semantic import SemanticGate
from adh.sentences import split_sentences

_STATUS = {
    InputError: 422,
    PreserveLockError: 422,
    RemoteDetectorUnavailableError: 501,
    RewriterError: 502,
    DetectorNotReadyError: 503,
    SemanticBackendError: 503,
}


def _http_status(error: AdhError) -> int:
    return _STATUS.get(type(error), 400)


def _windows(result: ScoreResult) -> list[dict]:
    return [
        asdict(window) if is_dataclass(window) else dict(window)  # type: ignore[arg-type]
        for window in result.windows
    ]


def create_app(
    *,
    detector: Detector | None = None,
    rewriter: Rewriter | None = None,
    semantic_gate: SemanticGate | None = None,
    default_detector: str = DEFAULT_MODEL,
    device: str = "auto",
    models_dir: Path | str | None = None,
) -> Any:
    """Build the ASGI app. Tests inject fakes; CLI injects loaded adapters."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as error:
        raise SemanticBackendError(
            "FastAPI is required for the HTTP API. "
            "Install extras: pip install 'adversarial-detector-humanizer[api]'"
        ) from error

    application = FastAPI(
        title="adversarial-detector-humanizer",
        version=__version__,
        description=(
            "Detector-verified, sentence-targeted, meaning-preserving humanizer. "
            "Scores are local-proxy estimates. No bypass guarantees."
        ),
    )
    application.state.detector = detector
    application.state.rewriter = rewriter
    application.state.semantic_gate = semantic_gate
    application.state.default_detector = default_detector
    application.state.device = device
    application.state.models_dir = models_dir

    def resolve_detector(name: str | None) -> Detector:
        if application.state.detector is not None and (
            name is None or name == application.state.detector.name
        ):
            return application.state.detector
        return load_detector(
            name or application.state.default_detector,
            models_dir=application.state.models_dir,
            device=application.state.device,
        )

    def resolve_rewriter(model: str | None) -> Rewriter:
        if application.state.rewriter is not None:
            return application.state.rewriter
        return load_rewriter(model=model)

    def resolve_gate(prefer: str, allow_lexical: bool) -> SemanticGate:
        if application.state.semantic_gate is not None:
            return application.state.semantic_gate
        return load_gate(prefer=prefer, allow_lexical=allow_lexical)

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        bound = application.state.detector
        name = bound.name if bound is not None else application.state.default_detector
        return HealthResponse(status="ok", version=__version__, detector=name)

    @application.get("/v1/models")
    def models() -> dict:
        return {"models": list_models(application.state.models_dir)}

    @application.post("/v1/score", response_model=ScoreResponse)
    def score_endpoint(payload: ScoreRequest) -> ScoreResponse:
        try:
            loaded = resolve_detector(payload.detector)
            result = loaded.score(payload.text)
        except AdhError as error:
            raise HTTPException(status_code=_http_status(error), detail=str(error)) from error
        return ScoreResponse(
            detector=loaded.name,
            score=result.score,
            label=result.label,
            windows=_windows(result),
        )

    @application.post("/v1/humanize")
    def humanize_endpoint(
        payload: HumanizeRequest,
    ) -> RunReport | CompactHumanizeResponse:
        try:
            loaded = resolve_detector(payload.detector)
            gate = resolve_gate(payload.semantic, payload.allow_lexical_gate)
            writer = resolve_rewriter(payload.rewriter_model)
            report = humanize(
                payload.text,
                detector=loaded,
                rewriter=writer,
                semantic_gate=gate,
                config=EngineConfig(
                    target_score=payload.target_score,
                    max_rounds=payload.max_rounds,
                    sentence_threshold=payload.sentence_threshold,
                    min_semantic_similarity=payload.min_semantic_similarity,
                    max_rewrite_ratio=payload.max_rewrite_ratio,
                    best_of_n=payload.best_of_n,
                    rewriter_model=payload.rewriter_model or "gpt-4o-mini",
                    detector=loaded.name,
                ),
            )
        except AdhError as error:
            raise HTTPException(status_code=_http_status(error), detail=str(error)) from error
        if payload.compact:
            return compact_from_report(report)
        return report

    @application.post("/v1/sentences", response_model=SentenceSplitResponse)
    def sentences_endpoint(payload: SentenceSplitRequest) -> SentenceSplitResponse:
        try:
            spans = split_sentences(payload.text)
        except AdhError as error:
            raise HTTPException(status_code=_http_status(error), detail=str(error)) from error
        return SentenceSplitResponse(
            sentences=[
                {"i": index, "text": span.text, "start": span.start, "end": span.end}
                for index, span in enumerate(spans)
            ]
        )

    return application


app = None


def get_app():
    global app
    if app is None:
        app = create_app()
    return app
