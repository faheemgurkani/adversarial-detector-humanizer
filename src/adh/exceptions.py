"""Shared exception types for the humanizer engine."""


class AdhError(Exception):
    """Base error for the adversarial-detector-humanizer package."""

    code: str = "internal_error"


class InputError(AdhError, ValueError):
    """The caller provided empty, invalid, or conflicting input."""

    code = "invalid_input"


class DetectorNotReadyError(AdhError, RuntimeError):
    """A local detector artifact is missing, incomplete, or unloadable."""

    code = "detector_not_ready"


class RemoteDetectorUnavailableError(AdhError, RuntimeError):
    """A remote detector was requested for an unsupported code path."""

    code = "remote_detector_unsupported"


class RemoteDetectorError(AdhError, RuntimeError):
    """A remote detector API call failed or returned an unusable response."""

    code = "remote_detector_error"


class PreserveLockError(AdhError, ValueError):
    """Locked spans could not be extracted or restored safely."""

    code = "preserve_lock_failed"


class SemanticBackendError(AdhError, RuntimeError):
    """The semantic similarity backend is unavailable or misconfigured."""

    code = "semantic_backend_unavailable"


class RewriterError(AdhError, RuntimeError):
    """The rewriter provider rejected the request or returned unusable text."""

    code = "rewriter_unavailable"


class HardModeUnavailableError(AdhError, RuntimeError):
    """Token-guided hard mode requires GPU extras that are not installed."""

    code = "hard_mode_unavailable"


class IdempotencyConflictError(AdhError, RuntimeError):
    """An idempotency key was reused with a different request body."""

    code = "idempotency_key_reused"
