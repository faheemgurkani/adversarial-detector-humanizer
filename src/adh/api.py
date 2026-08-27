"""Local FastAPI surface. Hosted SaaS must keep calling this same engine."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, get_args

from adh import __version__
from adh.config import AdhConfig, load_config
from adh.detectors.base import Detector, ScoreResult
from adh.errors import error_response
from adh.exceptions import (
    AdhError,
    DetectorNotReadyError,
    IdempotencyConflictError,
    InputError,
    PreserveLockError,
    RemoteDetectorError,
    RemoteDetectorUnavailableError,
    RewriterError,
    SemanticBackendError,
)
from adh.factory import load_detector, load_gate, load_rewriter
from adh.idempotency import IdempotencyStore
from adh.ids import new_request_id
from adh.models import list_models
from adh.report import RunReport, StopReason
from adh.rewriter import Rewriter
from adh.schemas import (
    CompactHumanizeResponse,
    HealthResponse,
    HumanizeRequest,
    ScoreRequest,
    ScoreResponse,
    SentenceSplitRequest,
    SentenceSplitResponse,
    StructuredErrorResponse,
    compact_from_report,
)
from adh.semantic import SemanticGate
from adh.sentences import split_sentences
from adh.service import run_humanize, run_score

_STATUS = {
    InputError: 422,
    PreserveLockError: 422,
    IdempotencyConflictError: 409,
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


def _request_id(request: Any) -> str:
    return getattr(request.state, "request_id", new_request_id())


def _error_json(request: Any, error: AdhError, *, code: str | None = None) -> dict[str, Any]:
    return {"error": error_response(error, _request_id(request), code=code)}


def _serialize_humanize(report: RunReport, payload: HumanizeRequest) -> dict[str, Any]:
    if payload.compact:
        return compact_from_report(report, metadata=payload.metadata).model_dump()
    body = report.model_dump()
    if payload.metadata:
        body["metadata"] = dict(payload.metadata)
    return body


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
    idempotency_store: IdempotencyStore | None = None,
) -> Any:
    """Build the ASGI app. Tests inject fakes; CLI injects loaded adapters."""
    try:
        from fastapi import FastAPI, Header, Request
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse
        from starlette.middleware.base import BaseHTTPMiddleware
    except ImportError as error:
        raise SemanticBackendError(
            "FastAPI is required for the HTTP API. "
            "Install extras: pip install 'adversarial-detector-humanizer[api]'"
        ) from error

    file_cfg = load_config(config_path)
    if server_config is None:
        server_config = file_cfg or AdhConfig()

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
    application.state.file_config = file_cfg
    application.state.default_detector = resolved_detector
    application.state.device = resolved_device
    application.state.models_dir = resolved_models_dir
    application.state.idempotency_store = idempotency_store or IdempotencyStore()

    class RequestIdMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            request.state.request_id = new_request_id()
            response = await call_next(request)
            response.headers["X-Request-Id"] = request.state.request_id
            return response

    application.add_middleware(RequestIdMiddleware)

    @application.exception_handler(AdhError)
    async def adh_error_handler(request: Request, error: AdhError) -> JSONResponse:
        request_id = _request_id(request)
        return JSONResponse(
            status_code=_http_status(error),
            content=_error_json(request, error),
            headers={"X-Request-Id": request_id},
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        request_id = _request_id(request)
        message = "invalid request"
        for item in error.errors():
            message = str(item.get("msg", message))
            break
        payload = _error_json(request, InputError(message), code="invalid_input")
        return JSONResponse(
            status_code=422,
            content=payload,
            headers={"X-Request-Id": request_id},
        )

    if file_cfg is not None or config_path is not None:
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
        loaded, result = run_score(
            payload.text,
            detector_name=_request_detector(payload),
            detector=application.state.detector,
            device=_request_device(payload),
            models_dir=_request_models_dir(payload),
            default_detector=application.state.default_detector,
        )
        return ScoreResponse(
            detector=loaded.name,
            score=result.score,
            label=result.label,
            windows=_windows(result),
        )

    @application.post(
        "/v1/humanize",
        responses={
            200: {"description": "Humanize result (compact by default)."},
            409: {"model": StructuredErrorResponse},
            422: {"model": StructuredErrorResponse},
            502: {"model": StructuredErrorResponse},
        },
    )
    def humanize_endpoint(
        payload: HumanizeRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> RunReport | CompactHumanizeResponse | dict[str, Any]:
        store: IdempotencyStore = application.state.idempotency_store
        body_hash = IdempotencyStore.hash_body(payload.model_dump(mode="json"))
        if idempotency_key:
            cached = store.lookup(idempotency_key, body_hash)
            if cached is not None:
                return cached

        report = run_humanize(
            payload.text,
            config=payload,
            file=application.state.file_config,
            detector=application.state.detector,
            rewriter=application.state.rewriter,
            semantic_gate=application.state.semantic_gate,
            default_detector=application.state.default_detector,
            device=_request_device(payload),
            models_dir=_request_models_dir(payload),
        )
        response_data = _serialize_humanize(report, payload)
        if idempotency_key and report.report_id:
            store.store(
                idempotency_key,
                body_hash=body_hash,
                response=response_data,
                report_id=report.report_id,
            )
        return response_data

    @application.post("/v1/sentences", response_model=SentenceSplitResponse)
    def sentences_endpoint(payload: SentenceSplitRequest) -> SentenceSplitResponse:
        spans = split_sentences(payload.text)
        return SentenceSplitResponse(
            sentences=[
                {"i": index, "text": span.text, "start": span.start, "end": span.end}
                for index, span in enumerate(spans)
            ]
        )

    _annotate_openapi(application)
    return application


def _annotate_openapi(application: Any) -> None:
    """Document stop reasons and error codes in OpenAPI metadata."""
    stop_reasons = ", ".join(
        [
            "passed",
            "max_rounds",
            "no_flagged_sentences",
            "all_candidates_rejected",
            "max_rewrite_ratio",
            "already_below_target",
        ]
    )
    application.openapi_tags = [
        {
            "name": "humanize",
            "description": (
                f"Compact responses include stop_reason values: {stop_reasons}. "
                "Errors use {{error: {{code, message, retryable, doc_url, request_id}}}}."
            ),
        }
    ]

    original_openapi = application.openapi

    def custom_openapi() -> dict[str, Any]:
        if application.openapi_schema:
            return application.openapi_schema
        schema = original_openapi()
        humanize_schema = (
            schema.get("components", {})
            .get("schemas", {})
            .get("CompactHumanizeResponse", {})
        )
        properties = humanize_schema.get("properties", {})
        if "stop_reason" in properties:
            properties["stop_reason"]["enum"] = list(get_args(StopReason))
        application.openapi_schema = schema
        return schema

    application.openapi = custom_openapi  # type: ignore[method-assign]


app = None


def get_app():
    global app
    if app is None:
        app = create_app()
    return app
