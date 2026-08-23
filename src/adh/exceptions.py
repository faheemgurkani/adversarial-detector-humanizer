"""Shared exception types for the humanizer engine."""


class AdhError(Exception):
    """Base error for the adversarial-detector-humanizer package."""


class InputError(AdhError, ValueError):
    """The caller provided empty, invalid, or conflicting input."""


class DetectorNotReadyError(AdhError, RuntimeError):
    """A local detector artifact is missing, incomplete, or unloadable."""


class RemoteDetectorUnavailableError(AdhError, RuntimeError):
    """A later-phase remote detector was requested but is not implemented."""


class PreserveLockError(AdhError, ValueError):
    """Locked spans could not be extracted or restored safely."""


class SemanticBackendError(AdhError, RuntimeError):
    """The semantic similarity backend is unavailable or misconfigured."""


class RewriterError(AdhError, RuntimeError):
    """The rewriter provider rejected the request or returned unusable text."""
