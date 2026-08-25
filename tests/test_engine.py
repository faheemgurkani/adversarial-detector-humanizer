from __future__ import annotations

import pytest

from adh.engine import EngineConfig, humanize
from adh.exceptions import InputError
from adh.rewriter import IdentityRewriter, ScriptedRewriter
from tests.conftest import CueDetector


class StripCueRewriter:
    """Keep lock sentinels, drop stock AI openers."""

    name = "strip-cue"

    def rewrite(self, sentence: str, *, n: int = 1, history=None) -> list[str]:
        rewritten = sentence
        for cue in (
            "Furthermore, ",
            "furthermore, ",
            "it is important to note ",
        ):
            rewritten = rewritten.replace(cue, "")
        rewritten = rewritten.strip() or sentence
        return [rewritten] * n


def test_empty_input_raises(lexical_gate) -> None:
    with pytest.raises(InputError):
        humanize(
            "  ",
            detector=CueDetector(),
            rewriter=IdentityRewriter(),
            semantic_gate=lexical_gate,
        )


def test_already_below_target(lexical_gate) -> None:
    report = humanize(
        "This is a human sentence.",
        detector=CueDetector(),
        rewriter=IdentityRewriter(),
        semantic_gate=lexical_gate,
        config=EngineConfig(target_score=30),
    )
    assert report.stop_reason == "already_below_target"
    assert report.rounds == 0
    assert report.output_text == report.input_text


def test_loop_rewrites_flagged_sentence(lexical_gate) -> None:
    report = humanize(
        "Furthermore, the method is important to note in 2024.",
        detector=CueDetector(),
        rewriter=StripCueRewriter(),
        semantic_gate=lexical_gate,
        config=EngineConfig(
            target_score=30,
            max_rounds=3,
            sentence_threshold=50,
            min_semantic_similarity=0.2,
        ),
    )
    assert report.score_after < report.score_before
    assert report.stop_reason == "passed"
    assert "Furthermore" not in report.output_text


def test_max_rounds_keeps_best(lexical_gate) -> None:
    rewriter = IdentityRewriter()
    report = humanize(
        "Furthermore, the landscape is important to note.",
        detector=CueDetector(),
        rewriter=rewriter,
        semantic_gate=lexical_gate,
        config=EngineConfig(target_score=5, max_rounds=2, min_semantic_similarity=0.1),
    )
    assert report.stop_reason in {"max_rounds", "all_candidates_rejected"}
    assert report.rounds >= 1


def test_all_candidates_rejected_on_meaning_drift(lexical_gate) -> None:
    rewriter = ScriptedRewriter(
        {
            "Furthermore, budgets were approved tonight.": [
                "Purple elephants invented jazz yesterday."
            ]
        }
    )
    report = humanize(
        "Furthermore, budgets were approved tonight.",
        detector=CueDetector(),
        rewriter=rewriter,
        semantic_gate=lexical_gate,
        config=EngineConfig(
            target_score=10,
            max_rounds=2,
            min_semantic_similarity=0.85,
        ),
    )
    assert report.stop_reason == "all_candidates_rejected"
    assert report.output_text == report.input_text


def test_no_flagged_sentences_when_top_k_is_zero(lexical_gate) -> None:
    report = humanize(
        "Furthermore, the method is important to note.",
        detector=CueDetector(),
        rewriter=IdentityRewriter(),
        semantic_gate=lexical_gate,
        config=EngineConfig(
            target_score=5,
            sentence_threshold=99.9,
            top_k_fallback=0,
            min_semantic_similarity=0.1,
        ),
    )
    assert report.stop_reason == "no_flagged_sentences"


def test_max_rewrite_ratio_zero_stops(lexical_gate) -> None:
    report = humanize(
        "Furthermore, the method is important to note.",
        detector=CueDetector(),
        rewriter=StripCueRewriter(),
        semantic_gate=lexical_gate,
        config=EngineConfig(
            target_score=5,
            max_rewrite_ratio=0,
            min_semantic_similarity=0.1,
        ),
    )
    assert report.stop_reason == "max_rewrite_ratio"
    assert report.output_text == report.input_text
