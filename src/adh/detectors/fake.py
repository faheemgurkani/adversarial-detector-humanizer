"""Deterministic detector for tests and CLI dry-runs."""

from __future__ import annotations

from adh.detectors.base import Detector, ScoreResult, probability_to_result, require_text
from adh.report import score_to_label


class FakeDetector:
    """Score text from caller-supplied per-sentence values or lexical cues.

    If ``sentence_scores`` is provided, each span is looked up by exact text,
    then by a 0-based index fallback. Whole-document score is the mean of
    the span scores when ``score_spans`` has been used, otherwise a default.
    """

    name = "fake"

    def __init__(
        self,
        sentence_scores: dict[str, float] | None = None,
        *,
        default_score: float = 80.0,
        document_score: float | None = None,
    ) -> None:
        if not 0.0 <= default_score <= 100.0:
            raise ValueError("default_score must be between 0 and 100")
        if document_score is not None and not 0.0 <= document_score <= 100.0:
            raise ValueError("document_score must be between 0 and 100")
        self.sentence_scores = dict(sentence_scores or {})
        self.default_score = default_score
        self.document_score = document_score

    def score(self, text: str) -> ScoreResult:
        require_text(text)
        stripped = text.strip()
        if stripped in self.sentence_scores:
            value = self.sentence_scores[stripped]
            return ScoreResult(score=value, label=score_to_label(value))
        if self.document_score is not None:
            return ScoreResult(
                score=self.document_score,
                label=score_to_label(self.document_score),
            )
        return ScoreResult(
            score=self.default_score,
            label=score_to_label(self.default_score),
        )

    def score_spans(self, texts: list[str]) -> list[ScoreResult]:
        results: list[ScoreResult] = []
        for index, text in enumerate(texts):
            require_text(text)
            key = text.strip()
            if key in self.sentence_scores:
                value = self.sentence_scores[key]
            elif str(index) in self.sentence_scores:
                value = self.sentence_scores[str(index)]
            elif self.document_score is not None:
                value = self.document_score
            else:
                value = self.default_score
            results.append(ScoreResult(score=value, label=score_to_label(value)))
        return results

    def set_document_score(self, score: float) -> None:
        if not 0.0 <= score <= 100.0:
            raise ValueError("score must be between 0 and 100")
        self.document_score = score


def lexical_ai_score(text: str) -> ScoreResult:
    """Cheap heuristic used only when no detector artifact is available."""
    require_text(text)
    cues = (
        "furthermore",
        "moreover",
        "in conclusion",
        "it is important to note",
        "delve",
        "tapestry",
        "in today's fast-paced",
        "not only",
        "landscape",
    )
    lowered = text.lower()
    hits = sum(1 for cue in cues if cue in lowered)
    density = hits / max(1, len(text.split()))
    probability = min(0.99, 0.35 + density * 8.0 + (0.15 if "—" in text else 0.0))
    return probability_to_result(probability)
