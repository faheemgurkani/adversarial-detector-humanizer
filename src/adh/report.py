"""Public run-report contracts used by the CLI and later HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

StopReason = Literal[
    "passed",
    "max_rounds",
    "no_flagged_sentences",
    "all_candidates_rejected",
    "max_rewrite_ratio",
    "already_below_target",
]


def score_to_label(score: float) -> str:
    """Map a 0-100 AI score to a coarse label."""
    if score < 0 or score > 100:
        raise ValueError("score must be between 0 and 100")
    if score < 30:
        return "human-leaning"
    if score < 70:
        return "uncertain"
    return "ai-leaning"


class LockRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    ok: bool = True


class SentenceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    i: int
    original: str
    rewritten: str
    score_before: float
    score_after: float
    kept: bool
    start: int
    end: int


class WindowScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    score: float
    label: str
    start: int | None = None
    end: int | None = None


class RunReport(BaseModel):
    """Full engine result. The future REST body is a subset of this model."""

    model_config = ConfigDict(extra="forbid")

    input_text: str
    output_text: str
    detector: str
    score_before: float
    score_after: float
    semantic_similarity: float
    rounds: int
    stop_reason: StopReason
    sentences: list[SentenceReport] = Field(default_factory=list)
    locks: list[LockRecord] = Field(default_factory=list)
    flagged_count: int = 0
    rewrite_ratio: float = 0.0

    def to_public_dict(self) -> dict[str, float | str]:
        return {
            "ai_score_before": self.score_before,
            "ai_score_after": self.score_after,
            "semantic_score": self.semantic_similarity,
            "stop_reason": self.stop_reason,
            "detector": self.detector,
        }
