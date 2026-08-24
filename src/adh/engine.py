"""Closed-loop detector-guided humanizer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from adh.detectors.base import Detector
from adh.exceptions import InputError, PreserveLockError
from adh.preserve import extract_locks, lock_records, restore_locks
from adh.report import LockRecord, RunReport, SentenceReport, StopReason
from adh.rewriter import Rewriter
from adh.semantic import SemanticGate, passes_gate
from adh.sentences import SentenceSpan, reassemble, split_sentences


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_score: float = Field(default=30.0, ge=0.0, le=100.0)
    max_rounds: int = Field(default=5, ge=1, le=20)
    sentence_threshold: float = Field(default=50.0, ge=0.0, le=100.0)
    min_semantic_similarity: float = Field(default=0.88, ge=0.0, le=1.0)
    max_rewrite_ratio: float = Field(default=0.4, ge=0.0, le=1.0)
    best_of_n: int = Field(default=2, ge=1, le=8)
    top_k_fallback: int = Field(default=1, ge=0, le=20)
    rewriter_model: str = "gpt-4o-mini"
    detector: str = "qwen3-variable"


@dataclass
class _Candidate:
    index: int
    original: str
    rewritten: str
    score_before: float
    score_after: float
    kept: bool
    start: int
    end: int
    locks: list[LockRecord]


def _word_count(text: str) -> int:
    return len(text.split())


def _flag_indices(
    spans: list[SentenceSpan],
    scores: list[float],
    *,
    threshold: float,
    top_k: int,
    max_rewrite_ratio: float,
) -> list[int]:
    flagged = [index for index, score in enumerate(scores) if score >= threshold]
    if not flagged and top_k > 0:
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        flagged = ranked[: min(top_k, len(ranked))]
    total_words = max(1, sum(_word_count(span.text) for span in spans))
    selected: list[int] = []
    used_words = 0
    for index in sorted(flagged, key=lambda item: scores[item], reverse=True):
        next_words = used_words + _word_count(spans[index].text)
        if selected and next_words / total_words > max_rewrite_ratio:
            continue
        if not selected and max_rewrite_ratio == 0:
            break
        selected.append(index)
        used_words = next_words
    return sorted(selected)


def _pick_candidate(
    original: str,
    *,
    rewriter: Rewriter,
    detector: Detector,
    gate: SemanticGate,
    threshold: float,
    best_of_n: int,
) -> tuple[str, float, float, list[LockRecord], bool]:
    locked, lock = extract_locks(original)
    lock_meta = [
        LockRecord(id=identifier, text=text, ok=True)
        for identifier, text, _ok in lock_records(lock, original)
    ]
    before = detector.score(original).score
    best_text = original
    best_score = before
    kept = False
    try:
        candidates = rewriter.rewrite(locked, n=best_of_n)
    except Exception:
        candidates = []
    for raw in candidates:
        try:
            restored = restore_locks(raw, lock, strict=True)
        except PreserveLockError:
            continue
        ok, _similarity = passes_gate(original, restored, gate, threshold)
        if not ok:
            continue
        score = detector.score(restored).score
        if score < best_score or (score == best_score and restored != original):
            best_text = restored
            best_score = score
            kept = restored != original
            lock_meta = [
                LockRecord(id=identifier, text=text, ok=present)
                for identifier, text, present in lock_records(lock, restored)
            ]
    return best_text, before, best_score, lock_meta, kept


def humanize(
    text: str,
    *,
    detector: Detector,
    rewriter: Rewriter,
    semantic_gate: SemanticGate,
    config: EngineConfig | None = None,
) -> RunReport:
    """Run the score → flag → lock → rewrite → gate → rescore loop."""
    if not isinstance(text, str) or not text.strip():
        raise InputError("text cannot be empty")
    settings = config or EngineConfig()

    spans = split_sentences(text)
    score_before = detector.score(text).score
    current = text
    current_score = score_before
    best_text = text
    best_score = score_before
    best_similarity = 1.0
    sentence_reports: list[SentenceReport] = []
    lock_report: list[LockRecord] = []
    flagged_count = 0
    rewrite_ratio = 0.0
    stop_reason: StopReason = "max_rounds"
    rounds = 0

    if score_before <= settings.target_score:
        return RunReport(
            input_text=text,
            output_text=text,
            detector=detector.name,
            score_before=score_before,
            score_after=score_before,
            semantic_similarity=1.0,
            rounds=0,
            stop_reason="already_below_target",
            sentences=[],
            locks=[],
            flagged_count=0,
            rewrite_ratio=0.0,
        )

    for _round in range(settings.max_rounds):
        rounds += 1
        spans = split_sentences(current)
        span_scores = [result.score for result in detector.score_spans([span.text for span in spans])]
        flagged = _flag_indices(
            spans,
            span_scores,
            threshold=settings.sentence_threshold,
            top_k=settings.top_k_fallback,
            max_rewrite_ratio=settings.max_rewrite_ratio,
        )
        flagged_count = len(flagged)
        total_words = max(1, _word_count(current))
        rewrite_ratio = sum(_word_count(spans[index].text) for index in flagged) / total_words

        if not flagged:
            stop_reason = (
                "max_rewrite_ratio" if settings.max_rewrite_ratio == 0 else "no_flagged_sentences"
            )
            break

        replacements: dict[int, str] = {}
        round_reports: list[SentenceReport] = []
        accepted = 0
        for index in flagged:
            span = spans[index]
            rewritten, before, after, locks, kept = _pick_candidate(
                span.text,
                rewriter=rewriter,
                detector=detector,
                gate=semantic_gate,
                threshold=settings.min_semantic_similarity,
                best_of_n=settings.best_of_n,
            )
            lock_report.extend(locks)
            if kept:
                replacements[index] = rewritten
                accepted += 1
            round_reports.append(
                SentenceReport(
                    i=index,
                    original=span.text,
                    rewritten=rewritten,
                    score_before=before,
                    score_after=after,
                    kept=kept,
                    start=span.start,
                    end=span.end,
                )
            )

        if accepted == 0:
            sentence_reports = round_reports
            stop_reason = "all_candidates_rejected"
            break

        current = reassemble(current, replacements)
        ok, similarity = passes_gate(text, current, semantic_gate, settings.min_semantic_similarity)
        if not ok:
            # Whole-document meaning drifted; keep the previous best and stop.
            stop_reason = "all_candidates_rejected"
            sentence_reports = round_reports
            break

        current_score = detector.score(current).score
        sentence_reports = round_reports
        if current_score < best_score:
            best_text = current
            best_score = current_score
            best_similarity = similarity
        if current_score <= settings.target_score:
            stop_reason = "passed"
            break
    else:
        stop_reason = "max_rounds"

    return RunReport(
        input_text=text,
        output_text=best_text,
        detector=detector.name,
        score_before=score_before,
        score_after=best_score,
        semantic_similarity=round(best_similarity, 4),
        rounds=rounds,
        stop_reason=stop_reason,
        sentences=sentence_reports,
        locks=lock_report,
        flagged_count=flagged_count,
        rewrite_ratio=round(rewrite_ratio, 4),
    )
