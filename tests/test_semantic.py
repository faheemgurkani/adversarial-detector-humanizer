from __future__ import annotations

import pytest

from adh.exceptions import InputError
from adh.semantic import LexicalSemanticGate, build_semantic_gate, passes_gate


def test_identical_texts_are_similar(lexical_gate: LexicalSemanticGate) -> None:
    score = lexical_gate.similarity("The cat sat on the mat.", "The cat sat on the mat.")
    assert score == pytest.approx(1.0)


def test_meaning_flip_is_low(lexical_gate: LexicalSemanticGate) -> None:
    score = lexical_gate.similarity(
        "The committee approved the budget tonight.",
        "Purple elephants invented jazz yesterday.",
    )
    assert score < 0.3


def test_empty_pair_raises(lexical_gate: LexicalSemanticGate) -> None:
    with pytest.raises(InputError):
        lexical_gate.similarity("hello", "   ")


def test_gate_threshold(lexical_gate: LexicalSemanticGate) -> None:
    ok, score = passes_gate(
        "cats sit outside",
        "cats sit outside",
        lexical_gate,
        0.88,
    )
    assert ok
    assert score >= 0.88


def test_build_lexical_explicitly() -> None:
    gate = build_semantic_gate(prefer="lexical")
    assert gate.name == "lexical"


def test_unknown_prefer_raises() -> None:
    with pytest.raises(InputError):
        build_semantic_gate(prefer="magic")
