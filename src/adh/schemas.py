"""HTTP request and response models for the open-core API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from adh.models import DEFAULT_MODEL
from adh.report import RunReport


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    detector: str = Field(default=DEFAULT_MODEL)
    device: str = Field(default="auto")
    models_dir: str | None = None


class ScoreResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detector: str
    score: float
    label: str
    windows: list[dict] = Field(default_factory=list)


class HumanizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    detector: str = Field(default=DEFAULT_MODEL)
    device: str = Field(default="auto")
    models_dir: str | None = None
    target_score: float = Field(default=30.0, ge=0.0, le=100.0)
    verdict_score: float = Field(default=45.0, ge=0.0, le=100.0)
    max_rounds: int = Field(default=5, ge=1, le=20)
    sentence_threshold: float = Field(default=50.0, ge=0.0, le=100.0)
    min_semantic_similarity: float = Field(default=0.88, ge=0.0, le=1.0)
    max_rewrite_ratio: float = Field(default=0.4, ge=0.0, le=1.0)
    best_of_n: int = Field(default=3, ge=1, le=8)
    rewriter_model: str | None = None
    semantic: str = Field(default="auto")
    allow_lexical_gate: bool = False
    meaning_gate_mode: str = Field(default="auto")
    verify: list[str] = Field(default_factory=list)
    verify_threshold: float = Field(default=45.0, ge=0.0, le=100.0)
    deploy_detectors: list[str] = Field(default_factory=list)
    enable_logprob_blend: bool = True
    logprob_blend_weight: float = Field(default=0.15, ge=0.0)
    hard_mode: bool = False
    hard_mode_max_sentences: int = Field(default=1, ge=0, le=5)
    prepass: str = Field(default="none")
    prepass_lang: str = Field(default="fi")
    prepass_max_paragraphs: int = Field(default=2, ge=0, le=10)
    prepass_backend: str = Field(default="llm")
    compact: bool = False


class CompactHumanizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ai_score_before: float
    ai_score_after: float
    semantic_score: float
    stop_reason: str
    detector: str
    output_text: str


class SentenceSplitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)


class SentenceSplitResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentences: list[dict]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    version: str
    detector: str


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str
    code: str


def compact_from_report(report: RunReport) -> CompactHumanizeResponse:
    public = report.to_public_dict()
    return CompactHumanizeResponse(
        ai_score_before=float(public["ai_score_before"]),
        ai_score_after=float(public["ai_score_after"]),
        semantic_score=float(public["semantic_score"]),
        stop_reason=str(public["stop_reason"]),
        detector=str(public["detector"]),
        output_text=report.output_text,
    )
