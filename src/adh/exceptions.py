"""Shared exception types for the humanizer engine."""


class AdhError(Exception):
    """Base error for the adversarial-detector-humanizer package."""


class InputError(AdhError, ValueError):
    """The caller provided empty, invalid, or conflicting input."""


class DetectorNotReadyError(AdhError, RuntimeError):
    """A local detector artifact is missing, incomplete, or unloadable."""


class RemoteDetectorUnavailableError(AdhError, RuntimeError):
    """A remote detector was requested for an unsupported code path."""


class RemoteDetectorError(AdhError, RuntimeError):
    """A remote detector API call failed or returned an unusable response."""


class PreserveLockError(AdhError, ValueError):
    """Locked spans could not be extracted or restored safely."""


class SemanticBackendError(AdhError, RuntimeError):
    """The semantic similarity backend is unavailable or misconfigured."""


class RewriterError(AdhError, RuntimeError):
    """The rewriter provider rejected the request or returned unusable text."""


class HardModeUnavailableError(AdhError, RuntimeError):
    """Token-guided hard mode requires GPU extras that are not installed."""
