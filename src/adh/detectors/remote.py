"""Stubs for later-phase commercial detectors.

The open-core loop never calls these. They exist so a hosted SaaS layer can
swap detectors without rewriting ``humanize()``.
"""

from __future__ import annotations

from adh.detectors.base import Detector, ScoreResult, require_text
from adh.exceptions import RemoteDetectorUnavailableError


def _unavailable(name: str, text: str) -> ScoreResult:
    require_text(text)
    raise RemoteDetectorUnavailableError(
        f"{name} is a later-phase adapter. The open-core loop uses local "
        "Raschka detectors only. After a local loop converges, a hosted "
        "tier may call this API once for a before/after report."
    )


class PangramDetector:
    """Pangram 4 adapter. Not implemented in the open-core MVP."""

    name = "pangram"

    def __init__(self, api_key: str | None = None, *, model: str = "pangram-4") -> None:
        self.api_key = api_key
        self.model = model

    def score(self, text: str) -> ScoreResult:
        return _unavailable(self.name, text)

    def score_spans(self, texts: list[str]) -> list[ScoreResult]:
        if not texts:
            return []
        return [_unavailable(self.name, texts[0])]


class GPTZeroDetector:
    """GPTZero v2 adapter. Not implemented in the open-core MVP."""

    name = "gptzero"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def score(self, text: str) -> ScoreResult:
        return _unavailable(self.name, text)

    def score_spans(self, texts: list[str]) -> list[ScoreResult]:
        if not texts:
            return []
        return [_unavailable(self.name, texts[0])]


class EnsembleDetector:
    """Average scores from multiple detectors. Requires ready members."""

    name = "ensemble"

    def __init__(self, detectors: list[Detector], *, weights: list[float] | None = None) -> None:
        if not detectors:
            raise ValueError("ensemble requires at least one detector")
        if weights is not None and len(weights) != len(detectors):
            raise ValueError("weights must match the number of detectors")
        if weights is not None and any(weight < 0 for weight in weights):
            raise ValueError("weights must be non-negative")
        self.detectors = list(detectors)
        if weights is None:
            self.weights = [1.0] * len(detectors)
        else:
            total = sum(weights)
            if total <= 0:
                raise ValueError("weights must sum to a positive value")
            self.weights = [weight / total for weight in weights]

    def score(self, text: str) -> ScoreResult:
        require_text(text)
        from adh.report import score_to_label

        blended = 0.0
        for detector, weight in zip(self.detectors, self.weights, strict=True):
            blended += detector.score(text).score * weight
        score = round(blended, 4)
        return ScoreResult(score=score, label=score_to_label(score))

    def score_spans(self, texts: list[str]) -> list[ScoreResult]:
        from adh.report import score_to_label

        if not texts:
            return []
        matrices = [detector.score_spans(texts) for detector in self.detectors]
        results: list[ScoreResult] = []
        for index in range(len(texts)):
            blended = 0.0
            for row, weight in zip(matrices, self.weights, strict=True):
                blended += row[index].score * weight
            score = round(blended, 4)
            results.append(ScoreResult(score=score, label=score_to_label(score)))
        return results
