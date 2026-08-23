from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adh.detectors.fake import FakeDetector
from adh.rewriter import ScriptedRewriter
from adh.semantic import LexicalSemanticGate


@pytest.fixture
def lexical_gate() -> LexicalSemanticGate:
    return LexicalSemanticGate()


@pytest.fixture
def cue_detector() -> FakeDetector:
    return CueDetector()


class CueDetector:
    """High score when AI-ish cues are present, low otherwise."""

    name = "cue"

    def score(self, text: str):
        from adh.detectors.base import probability_to_result, require_text

        require_text(text)
        lowered = text.lower()
        probability = 0.92 if "furthermore" in lowered else 0.18
        return probability_to_result(probability)

    def score_spans(self, texts: list[str]):
        return [self.score(text) for text in texts]


@pytest.fixture
def scripted_rewriter() -> ScriptedRewriter:
    return ScriptedRewriter(
        {
            "Furthermore, the method is important to note in 2024.": [
                "The method mattered in 2024."
            ],
            "This is a human sentence.": ["This is a human sentence."],
        }
    )
