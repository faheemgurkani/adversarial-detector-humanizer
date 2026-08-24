from __future__ import annotations

import pytest

from adh.gates import build_meaning_gate_stack
from adh.gates.hedges import certainty_kept, polarity_kept
from adh.gates.numerals import numbers_kept
from adh.semantic import LexicalSemanticGate


@pytest.fixture
def stack() -> object:
    return build_meaning_gate_stack(prefer="lexical")


def test_negation_flip_rejected(stack) -> None:
    result = stack.evaluate("The build runs significantly faster.", "The build runs significantly slower.")
    assert not result.preserved
    assert "polarity" in result.vetoes or "similarity" in result.vetoes


def test_hedge_drop_rejected() -> None:
    assert not certainty_kept("The drug may cause drowsiness.", "The drug causes drowsiness.")


def test_numerals_drop_rejected() -> None:
    assert not numbers_kept("Only 7 of the 19 tests passed.", "Only a few of the 19 tests passed.")


def test_faithful_register_shift_accepted(stack) -> None:
    result = stack.evaluate(
        "The committee approved the budget tonight.",
        "The committee approved the budget tonight after debate.",
    )
    assert result.preserved


def test_polarity_kept() -> None:
    assert polarity_kept("It does not run.", "It doesn't run.")
