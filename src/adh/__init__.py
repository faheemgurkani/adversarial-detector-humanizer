"""Detector-verified, sentence-targeted, meaning-preserving humanizer."""

from adh.engine import EngineConfig, humanize
from adh.exceptions import (
    AdhError,
    DetectorNotReadyError,
    InputError,
    PreserveLockError,
    RemoteDetectorUnavailableError,
    RemoteDetectorError,
    RewriterError,
    SemanticBackendError,
)
from adh.report import RunReport, SentenceReport, score_to_label
from adh.sentences import SentenceSpan, split_sentences

__all__ = [
    "AdhError",
    "DetectorNotReadyError",
    "EngineConfig",
    "InputError",
    "PreserveLockError",
    "RemoteDetectorUnavailableError",
    "RemoteDetectorError",
    "RewriterError",
    "RunReport",
    "SemanticBackendError",
    "SentenceReport",
    "SentenceSpan",
    "humanize",
    "score_to_label",
    "split_sentences",
]

__version__ = "0.1.0"
