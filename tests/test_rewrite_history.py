from __future__ import annotations

import pytest

from adh.engine import EngineConfig, humanize
from adh.rewriter import RewriteCandidate, RewriteHistory, _build_chat_messages
from tests.conftest import CueDetector


class RecordingRewriter:
    name = "recording"

    def __init__(self) -> None:
        self.calls: list[tuple[str, RewriteHistory | None]] = []

    def rewrite(self, sentence: str, *, n: int = 1, history: RewriteHistory | None = None) -> list[str]:
        return [c.text for c in self.rewrite_candidates(sentence, n=n, history=history)]

    def rewrite_candidates(
        self,
        sentence: str,
        *,
        n: int = 1,
        history: RewriteHistory | None = None,
    ) -> list[RewriteCandidate]:
        self.calls.append((sentence, history))
        rewritten = sentence.replace("Furthermore, ", "")
        return [RewriteCandidate(text=rewritten or sentence, mean_logprob=-0.1)] * n


class SlowImproveRewriter:
    """First pass keeps cue; second pass with history removes it."""

    name = "slow-improve"

    def __init__(self) -> None:
        self.calls: list[RewriteHistory | None] = []

    def rewrite_candidates(
        self,
        sentence: str,
        *,
        n: int = 1,
        history: RewriteHistory | None = None,
    ) -> list[RewriteCandidate]:
        self.calls.append(history)
        if history:
            text = sentence.replace("Furthermore, ", "").replace("furthermore, ", "")
        else:
            text = sentence.replace("Furthermore", "Further")
        return [RewriteCandidate(text=text, mean_logprob=-0.1)]


def test_build_chat_messages_first_round() -> None:
    messages = _build_chat_messages("Rewrite me.")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_build_chat_messages_includes_prior_hop() -> None:
    messages = _build_chat_messages(
        "Rewrite again.",
        history=[("Original.", "First pass.")],
    )
    roles = [message["role"] for message in messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert "Do not revert prior improvements" in messages[0]["content"]


def test_engine_passes_history_on_second_round(lexical_gate) -> None:
    rewriter = SlowImproveRewriter()
    report = humanize(
        "Furthermore, the method is important to note.",
        detector=CueDetector(),
        rewriter=rewriter,
        semantic_gate=lexical_gate,
        config=EngineConfig(
            target_score=5,
            max_rounds=3,
            sentence_threshold=40,
            min_semantic_similarity=0.2,
        ),
    )
    assert len(rewriter.calls) >= 2
    assert rewriter.calls[0] is None
    assert rewriter.calls[1] is not None
    assert report.sentences
    assert report.sentences[0].round >= 1


def test_history_does_not_leak_across_sentence_indices(lexical_gate) -> None:
    recorder = RecordingRewriter()
    humanize(
        "Furthermore, one sentence here. Furthermore, another sentence there.",
        detector=CueDetector(),
        rewriter=recorder,
        semantic_gate=lexical_gate,
        config=EngineConfig(
            target_score=5,
            max_rounds=2,
            sentence_threshold=40,
            min_semantic_similarity=0.2,
            best_of_n=1,
        ),
    )
    first_histories = [history for _sentence, history in recorder.calls if history is None]
    assert len(first_histories) >= 2
