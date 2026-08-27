"""Local FastAPI surface. Hosted SaaS must keep calling this same engine."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from adh import __version__
from adh.config import AdhConfig, load_config
from adh.detectors.base import Detector, ScoreResult
from adh.exceptions import (
    AdhError,
    DetectorNotReadyError,
    InputError,
    PreserveLockError,
    RemoteDetectorError,
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
from adh.service import run_humanize, run_score

_STATUS = {
    InputError: 422,
    PreserveLockError: 422,
    RemoteDetectorUnavailableError: 501,
    RemoteDetectorError: 502,
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
    server_config: AdhConfig | None = None,
    config_path: Path | str | None = None,
    default_detector: str | None = None,
    device: str | None = None,
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

    if server_config is None:
        server_config = load_config(config_path) or AdhConfig()

    resolved_detector = default_detector or server_config.detector
    resolved_device = device or server_config.device
    resolved_models_dir = (
        models_dir if models_dir is not None else server_config.models_dir
    )

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
    application.state.server_config = server_config
    application.state.default_detector = resolved_detector
    application.state.device = resolved_device
    application.state.models_dir = resolved_models_dir

    if detector is None or rewriter is None or semantic_gate is None:
        try:
            if detector is None:
                application.state.detector = load_detector(
                    resolved_detector,
                    models_dir=resolved_models_dir,
                    device=resolved_device,
                )
            if rewriter is None:
                application.state.rewriter = load_rewriter(
                    name=server_config.rewriter,
                    model=server_config.rewriter_model,
                )
            if semantic_gate is None:
                application.state.semantic_gate = load_gate(
                    prefer=server_config.semantic,
                    allow_lexical=server_config.allow_lexical_gate,
                )
        except AdhError:
            pass

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        bound = application.state.detector
        name = bound.name if bound is not None else application.state.default_detector
        return HealthResponse(status="ok", version=__version__, detector=name)

    @application.get("/v1/models")
    def models() -> dict:
        return {"models": list_models(application.state.models_dir)}

    def _request_device(payload: ScoreRequest | HumanizeRequest) -> str:
        if "device" in payload.model_fields_set:
            return payload.device
        return application.state.device

    def _request_models_dir(payload: ScoreRequest | HumanizeRequest) -> Path | str | None:
        if "models_dir" in payload.model_fields_set:
            return payload.models_dir
        return application.state.models_dir

    def _request_detector(payload: ScoreRequest | HumanizeRequest) -> str:
        if "detector" in payload.model_fields_set:
            return payload.detector
        return application.state.default_detector

    @application.post("/v1/score", response_model=ScoreResponse)
    def score_endpoint(payload: ScoreRequest) -> ScoreResponse:
        try:
            loaded, result = run_score(
                payload.text,
                detector_name=_request_detector(payload),
                detector=application.state.detector,
                device=_request_device(payload),
                models_dir=_request_models_dir(payload),
                default_detector=application.state.default_detector,
            )
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
            report = run_humanize(
                payload.text,
                config=payload,
                file=application.state.server_config,
                detector=application.state.detector,
                rewriter=application.state.rewriter,
                semantic_gate=application.state.semantic_gate,
                default_detector=application.state.default_detector,
                device=_request_device(payload),
                models_dir=_request_models_dir(payload),
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
