from __future__ import annotations

import pytest

from adh.detectors.base import ScoreResult, probability_to_result
from adh.detectors.fake import FakeDetector
from adh.detectors.remote import EnsembleDetector
from adh.exceptions import InputError
from adh.factory import load_detector
from adh.report import score_to_label


def test_score_to_label_bounds() -> None:
    assert score_to_label(0) == "human-leaning"
    assert score_to_label(50) == "uncertain"
    assert score_to_label(90) == "ai-leaning"
    with pytest.raises(ValueError):
        score_to_label(120)


def test_probability_to_result_rejects_nan() -> None:
    with pytest.raises(ValueError):
        probability_to_result(float("nan"))


def test_fake_detector_lookup() -> None:
    detector = FakeDetector({"Hello.": 12.0}, default_score=80.0)
    assert detector.score("Hello.").score == 12.0
    spans = detector.score_spans(["Hello.", "Other."])
    assert spans[0].score == 12.0
    assert spans[1].score == 80.0


def test_fake_detector_empty_raises() -> None:
    with pytest.raises(InputError):
        FakeDetector().score(" ")



def test_ensemble_max_default() -> None:
    low = FakeDetector(document_score=10.0)
    high = FakeDetector(document_score=90.0)
    ensemble = EnsembleDetector([low, high], weights=[1.0, 1.0])
    result = ensemble.score("A normal sentence for blending.")
    assert result.score == pytest.approx(90.0)


def test_ensemble_mean_mode() -> None:
    low = FakeDetector(document_score=10.0)
    high = FakeDetector(document_score=90.0)
    ensemble = EnsembleDetector([low, high], weights=[1.0, 1.0], aggregate="mean")
    result = ensemble.score("A normal sentence for blending.")
    assert result.score == pytest.approx(50.0)
    spans = ensemble.score_spans(["A normal sentence for blending."])
    assert spans[0].score == pytest.approx(50.0)


def test_ensemble_rejects_bad_weights() -> None:
    with pytest.raises(ValueError):
        EnsembleDetector([])
    with pytest.raises(ValueError):
        EnsembleDetector([FakeDetector()], weights=[1.0, 2.0])


def test_load_detector_unknown() -> None:
    with pytest.raises(InputError):
        load_detector("nope")


def test_score_result_bounds() -> None:
    with pytest.raises(ValueError):
        ScoreResult(score=140, label="x")
