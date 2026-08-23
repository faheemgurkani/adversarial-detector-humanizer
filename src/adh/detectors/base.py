"""Detector protocol and score types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from adh.exceptions import InputError
from adh.report import score_to_label


@dataclass(frozen=True)
class Window:
    text: str
    score: float
    label: str
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class ScoreResult:
    score: float
    label: str
    windows: list[Window] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 100.0:
            raise ValueError("score must be between 0 and 100")


@runtime_checkable
class Detector(Protocol):
    name: str

    def score(self, text: str) -> ScoreResult:
        """Return a 0-100 AI score for a whole document."""

    def score_spans(self, texts: list[str]) -> list[ScoreResult]:
        """Score each span independently."""


def require_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise InputError("text cannot be empty")
    return text


def probability_to_result(probability: float, *, windows: list[Window] | None = None) -> ScoreResult:
    if probability != probability or probability < 0.0 or probability > 1.0:
        raise ValueError("AI probability must be between 0 and 1")
    score = round(float(probability) * 100.0, 4)
    return ScoreResult(
        score=score,
        label=score_to_label(score),
        windows=windows or [],
    )
