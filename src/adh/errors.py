"""Structured HTTP/CLI error envelopes."""

from __future__ import annotations

from typing import Any

from adh.exceptions import (
    AdhError,
    DetectorNotReadyError,
    HardModeUnavailableError,
    IdempotencyConflictError,
    InputError,
    PreserveLockError,
    RemoteDetectorError,
    RemoteDetectorUnavailableError,
    RewriterError,
    SemanticBackendError,
)

DOC_BASE = (
    "https://github.com/faheemgurkani/adversarial-detector-humanizer/"
    "blob/main/docs/BACKEND_PRD.md"
)

ERROR_CODES: dict[str, dict[str, Any]] = {
    "invalid_input": {
        "retryable": False,
        "doc_url": f"{DOC_BASE}#error-codes",
    },
    "unknown_detector": {
        "retryable": False,
        "doc_url": f"{DOC_BASE}#error-codes",
    },
    "preserve_lock_failed": {
        "retryable": False,
        "doc_url": f"{DOC_BASE}#error-codes",
    },
    "remote_detector_unsupported": {
        "retryable": False,
        "doc_url": f"{DOC_BASE}#error-codes",
    },
    "remote_detector_error": {
        "retryable": True,
        "doc_url": f"{DOC_BASE}#error-codes",
    },
    "rewriter_unavailable": {
        "retryable": False,
        "doc_url": f"{DOC_BASE}#error-codes",
    },
    "detector_not_ready": {
        "retryable": False,
        "doc_url": f"{DOC_BASE}#error-codes",
    },
    "semantic_backend_unavailable": {
        "retryable": False,
        "doc_url": f"{DOC_BASE}#error-codes",
    },
    "hard_mode_unavailable": {
        "retryable": False,
        "doc_url": f"{DOC_BASE}#error-codes",
    },
    "idempotency_key_reused": {
        "retryable": False,
        "doc_url": f"{DOC_BASE}#idempotency",
    },
    "internal_error": {
        "retryable": True,
        "doc_url": f"{DOC_BASE}#error-codes",
    },
}


def resolve_error_code(error: AdhError) -> str:
    if isinstance(error, InputError) and "unknown plugin" in str(error).lower():
        return "unknown_detector"
    explicit = getattr(error, "code", None)
    if explicit and explicit != "internal_error":
        return str(explicit)
    mapping: dict[type[AdhError], str] = {
        InputError: "invalid_input",
        PreserveLockError: "preserve_lock_failed",
        RemoteDetectorUnavailableError: "remote_detector_unsupported",
        RemoteDetectorError: "remote_detector_error",
        RewriterError: "rewriter_unavailable",
        DetectorNotReadyError: "detector_not_ready",
        SemanticBackendError: "semantic_backend_unavailable",
        HardModeUnavailableError: "hard_mode_unavailable",
        IdempotencyConflictError: "idempotency_key_reused",
    }
    for error_type, code in mapping.items():
        if isinstance(error, error_type):
            return code
    return "internal_error"


def error_response(
    error: AdhError,
    request_id: str,
    *,
    code: str | None = None,
) -> dict[str, Any]:
    """Build the public ``error`` object shared by HTTP and CLI JSON."""
    resolved = code or resolve_error_code(error)
    spec = ERROR_CODES.get(resolved, ERROR_CODES["internal_error"])
    return {
        "code": resolved,
        "message": str(error),
        "retryable": bool(spec["retryable"]),
        "doc_url": str(spec["doc_url"]),
        "request_id": request_id,
    }
