"""CPU statistical AI-likeness heuristic (humanize-text Method 3 signal)."""

from __future__ import annotations

import math
import re

from adh.detectors.base import ScoreResult, probability_to_result, require_text
from adh.report import score_to_label


def _split_sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]


def ai_likeness_probability(text: str) -> float:
    """Return 0–1 AI-likeness from TTR, sentence-length CV, and hapax ratio."""
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return 0.5

    words = text.split()
    if not words:
        return 0.5

    ttr = len({word.lower() for word in words}) / len(words)

    lengths = [len(sentence.split()) for sentence in sentences]
    mean_len = sum(lengths) / len(lengths)
    variance = sum((length - mean_len) ** 2 for length in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean_len if mean_len > 0 else 0.0

    word_counts: dict[str, int] = {}
    for word in words:
        lowered = word.lower()
        word_counts[lowered] = word_counts.get(lowered, 0) + 1
    hapax_ratio = (
        sum(1 for count in word_counts.values() if count == 1) / len(word_counts)
        if word_counts
        else 0.0
    )

    ttr_score = max(0.0, min(1.0, (0.7 - ttr) / 0.3))
    cv_score = max(0.0, min(1.0, (0.5 - cv) / 0.3))
    hapax_score = max(0.0, min(1.0, (0.6 - hapax_ratio) / 0.3))
    return (ttr_score + cv_score + hapax_score) / 3.0


class StatisticalDetector:
    """Weak proxy for uniform AI rhythm; best used inside a max ensemble."""

    name = "statistical"

    def score(self, text: str) -> ScoreResult:
        cleaned = require_text(text)
        return probability_to_result(ai_likeness_probability(cleaned))

    def score_spans(self, texts: list[str]) -> list[ScoreResult]:
        results: list[ScoreResult] = []
        for text in texts:
            cleaned = text.strip()
            if not cleaned:
                raise ValueError("span text cannot be empty")
            if len(_split_sentences(cleaned)) < 2:
                results.append(ScoreResult(score=50.0, label=score_to_label(50.0)))
                continue
            results.append(probability_to_result(ai_likeness_probability(cleaned)))
        return results
