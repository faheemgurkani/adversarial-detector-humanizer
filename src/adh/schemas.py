"""HTTP request and response models for the open-core API."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field, field_validator

from adh.hints import agent_hint_for
from adh.models import DEFAULT_MODEL
from adh.report import RunReport, StopReason

METADATA_MAX_KEYS = 50


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
    profile: str | None = Field(
        default=None,
        description="Preset bundle. Use 'fast' for zero-key test mode.",
    )
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
    compact: bool = True
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > METADATA_MAX_KEYS:
            raise ValueError(
                f"metadata accepts at most {METADATA_MAX_KEYS} key/value pairs"
            )
        return value


class CompactHumanizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str
    ai_score_before: float
    ai_score_after: float
    semantic_score: float
    stop_reason: StopReason
    detector: str
    output_text: str
    output: str
    agent_hint: str
    metadata: dict[str, str] = Field(default_factory=dict)
    input: str | None = Field(
        default=None,
        description="SHA-256 fingerprint (first 16 hex chars) of the source text.",
    )


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


class StructuredError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    retryable: bool
    doc_url: str
    request_id: str


class StructuredErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: StructuredError


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str
    code: str


def input_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def compact_from_report(
    report: RunReport,
    *,
    metadata: dict[str, str] | None = None,
    include_input_fingerprint: bool = True,
) -> CompactHumanizeResponse:
    public = report.to_public_dict()
    if not report.report_id:
        raise ValueError("compact response requires report.report_id")
    output_text = report.output_text
    return CompactHumanizeResponse(
        report_id=report.report_id,
        ai_score_before=float(public["ai_score_before"]),
        ai_score_after=float(public["ai_score_after"]),
        semantic_score=float(public["semantic_score"]),
        stop_reason=report.stop_reason,
        detector=str(public["detector"]),
        output_text=output_text,
        output=output_text,
        agent_hint=agent_hint_for(report),
        metadata=dict(metadata or {}),
        input=input_fingerprint(report.input_text) if include_input_fingerprint else None,
    )
