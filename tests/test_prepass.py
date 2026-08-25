from __future__ import annotations

from adh.engine import EngineConfig, humanize
from adh.gates.stack import MeaningGateStack
from adh.prepass.structural import StructuralPrepass, split_paragraphs
from adh.prepass.translate import IdentityTranslator, round_trip_translate
from adh.rewriter import IdentityRewriter
from tests.conftest import CueDetector


class ShuffleTranslator:
    name = "shuffle"

    def translate(self, text: str, *, source: str, target: str) -> str:
        if target == "en":
            return text
        return " ".join(reversed(text.split()))


class LockPreservingTranslator:
    name = "lock-preserving"

    def translate(self, text: str, *, source: str, target: str) -> str:
        if target != "en":
            return text.replace("Furthermore", "Moreover")
        return text.replace("Moreover", "Also")


def test_prepass_none_skips_translation(lexical_gate) -> None:
    calls = {"count": 0}

    class CountingTranslator:
        name = "counting"

        def translate(self, text: str, *, source: str, target: str) -> str:
            calls["count"] += 1
            return text

    report = humanize(
        "Furthermore, the method is important to note.\n\nSecond paragraph stays.",
        detector=CueDetector(),
        rewriter=IdentityRewriter(),
        semantic_gate=lexical_gate,
        config=EngineConfig(prepass="none", min_semantic_similarity=0.2),
        prepass_translator=CountingTranslator(),
    )
    assert calls["count"] == 0
    assert report.prepass_applied is False


def test_identity_round_trip_is_noop() -> None:
    text = "Hello world."
    assert round_trip_translate(text, lang="fi", translator=IdentityTranslator()) == text


def test_lock_token_survives_mock_round_trip(lexical_gate) -> None:
    gate_stack = MeaningGateStack(
        semantic_gate=lexical_gate,
        strict_semantic_similarity=0.2,
        relaxed_semantic_similarity=0.1,
    )
    prepass = StructuralPrepass(translator=LockPreservingTranslator(), lang="fi")
    paragraph = "Furthermore, revenue hit __LOCK_num_0__ in 2024."
    result = prepass.apply_paragraph(
        paragraph,
        original_paragraph=paragraph,
        gate_stack=gate_stack,
    )
    assert result is not None
    assert "__LOCK_num_0__" in result


def test_shuffle_may_be_rejected_by_gates(lexical_gate) -> None:
    gate_stack = MeaningGateStack(
        semantic_gate=lexical_gate,
        strict_semantic_similarity=0.85,
        relaxed_semantic_similarity=0.1,
    )
    prepass = StructuralPrepass(translator=ShuffleTranslator(), lang="fi")
    paragraph = (
        "Furthermore, the quarterly revenue increased substantially in 2024. "
        "The board approved the budget unanimously."
    )
    result = prepass.apply_paragraph(
        paragraph,
        original_paragraph=paragraph,
        gate_stack=gate_stack,
    )
    assert result is None or result != paragraph


def test_split_paragraphs_offsets() -> None:
    text = "First paragraph.\n\nSecond paragraph."
    spans = split_paragraphs(text)
    assert len(spans) == 2
    assert text[spans[0].start : spans[0].end] == "First paragraph."
    assert text[spans[1].start : spans[1].end] == "Second paragraph."
