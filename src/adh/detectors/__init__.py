"""Detector adapters used by the humanize loop."""

from adh.detectors.base import Detector, ScoreResult, Window
from adh.detectors.fake import FakeDetector
from adh.detectors.local_raschka import LocalRaschkaDetector
from adh.detectors.remote import EnsembleDetector, GPTZeroDetector, PangramDetector

__all__ = [
    "Detector",
    "EnsembleDetector",
    "FakeDetector",
    "GPTZeroDetector",
    "LocalRaschkaDetector",
    "PangramDetector",
    "ScoreResult",
    "Window",
]
