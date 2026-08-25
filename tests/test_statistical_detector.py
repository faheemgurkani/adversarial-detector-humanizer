from __future__ import annotations

import pytest

from adh.detectors.fake import FakeDetector
from adh.detectors.remote import EnsembleDetector
from adh.detectors.statistical import StatisticalDetector, ai_likeness_probability
from adh.factory import load_detector


def test_uniform_sentences_score_high() -> None:
    text = (
        "The result is clear. The result is clear. The result is clear. "
        "The result is clear."
    )
    assert ai_likeness_probability(text) > 0.5
    assert StatisticalDetector().score(text).score > 50.0


def test_varied_paragraph_scores_lower() -> None:
    text = (
        "Yesterday I walked through the rain-soaked market and bought oranges. "
        "The vendor laughed when I asked for a discount on the last crate. "
        "By evening the streets were empty except for a lone cyclist weaving "
        "between puddles under flickering lamps."
    )
    assert ai_likeness_probability(text) < 0.5
    assert StatisticalDetector().score(text).score < 50.0


def test_single_sentence_is_neutral() -> None:
    result = StatisticalDetector().score("One lone sentence without a partner.")
    assert result.score == pytest.approx(50.0)


def test_span_scoring_single_sentence_neutral() -> None:
    spans = StatisticalDetector().score_spans(["Only one sentence here."])
    assert spans[0].score == pytest.approx(50.0)


def test_ensemble_max_with_statistical() -> None:
    low = FakeDetector(document_score=30.0)
    high = StatisticalDetector()
    uniform = "Same length. Same length. Same length."
    ensemble = EnsembleDetector([low, high], aggregate="max")
    result = ensemble.score(uniform)
    assert result.score == pytest.approx(max(30.0, high.score(uniform).score))


def test_load_statistical_detector() -> None:
    loaded = load_detector("statistical")
    assert loaded.name == "statistical"
