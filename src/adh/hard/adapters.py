"""Detector adapter for token-guided decoding."""

from __future__ import annotations

from adh.detectors.base import Detector


class DetectorScoreAdapter:
    """Map ADH detector scores to AP-style get_scores (lower = more human)."""

    def __init__(self, detector: Detector) -> None:
        self.detector = detector

    def get_scores(self, texts: list[str]) -> list[float]:
        return [self.detector.score(text).score for text in texts]
