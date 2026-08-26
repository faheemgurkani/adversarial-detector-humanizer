"""Third-party detector used only in registry integration tests."""

from __future__ import annotations

from adh.detectors.base import ScoreResult, require_text
from adh.report import score_to_label


class AcmeDetector:
    """Stand-in for an external package registering ``adh.detectors``."""

    name = "acme-nlp"

    def score(self, text: str) -> ScoreResult:
        require_text(text)
        return ScoreResult(score=10.0, label=score_to_label(10.0))

    def score_spans(self, texts: list[str]) -> list[ScoreResult]:
        return [self.score(text) for text in texts]
