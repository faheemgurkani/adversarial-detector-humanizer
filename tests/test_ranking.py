from __future__ import annotations

from adh.ranking import blend_score


def test_blend_prefers_better_logprob_when_detector_close() -> None:
    low_detector = blend_score(
        detector_score=40.0,
        mean_logprob=-0.2,
        source="The method works well today.",
        candidate="The method works well today in practice.",
        logprob_blend_weight=0.5,
        detector_blend_weight=1.0,
    )
    high_detector_low_logprob = blend_score(
        detector_score=39.0,
        mean_logprob=-8.0,
        source="The method works well today.",
        candidate="Purple elephants dance nightly under stars.",
        logprob_blend_weight=0.5,
        detector_blend_weight=1.0,
    )
    assert low_detector < high_detector_low_logprob


def test_blend_weight_zero_matches_detector_only() -> None:
    score = blend_score(
        detector_score=42.0,
        mean_logprob=-3.0,
        source="A",
        candidate="B",
        logprob_blend_weight=0.0,
    )
    assert score == 42.0
