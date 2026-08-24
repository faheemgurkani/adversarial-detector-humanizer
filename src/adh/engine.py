"""Closed-loop detector-guided humanizer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from adh.detectors.base import Detector
from adh.exceptions import InputError, PreserveLockError
from adh.gates.stack import MeaningGateStack
from adh.preserve import extract_locks, lock_records, restore_locks, sentinels_preserved
from adh.report import LockRecord, RunReport, SentenceReport, StopReason
from adh.rewriter import Rewriter
from adh.scrub import scrub_text
from adh.semantic import SemanticGate
from adh.sentences import SentenceSpan, reassemble, split_sentences
from adh.tells import score_tells
from adh.verify import run_verification

TELLS_EPS_SCORE = 2.0


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_score: float = Field(default=30.0, ge=0.0, le=100.0)
    verdict_score: float = Field(default=45.0, ge=0.0, le=100.0)
    max_rounds: int = Field(default=5, ge=1, le=20)
    sentence_threshold: float = Field(default=50.0, ge=0.0, le=100.0)
    min_semantic_similarity: float = Field(default=0.88, ge=0.0, le=1.0)
    relaxed_semantic_similarity: float = Field(default=0.30, ge=0.0, le=1.0)
    max_rewrite_ratio: float = Field(default=0.4, ge=0.0, le=1.0)
    best_of_n: int = Field(default=3, ge=1, le=8)
    top_k_fallback: int = Field(default=1, ge=0, le=20)
    rewriter_model: str = "gpt-4o-mini"
    detector: str = "qwen3-variable"
    meaning_gate_mode: str = "auto"
    allow_lexical_gate: bool = False
    enable_tells_tiebreak: bool = True
    tells_tiebreak_epsilon: float = Field(default=TELLS_EPS_SCORE, ge=0.0, le=100.0)
    scrub_input: bool = True
    verify_detectors: list[str] = Field(default_factory=list)
    verify_threshold: float = Field(default=45.0, ge=0.0, le=100.0)
    verify_on_input: bool = True


def _word_count(text: str) -> int:
    return len(text.split())


def _build_gate_stack(config: EngineConfig, semantic_gate: SemanticGate | None) -> MeaningGateStack:
    if semantic_gate is not None:
        return MeaningGateStack(
            semantic_gate=semantic_gate,
            strict_semantic_similarity=config.min_semantic_similarity,
            relaxed_semantic_similarity=config.relaxed_semantic_similarity,
        )
    from adh.gates import build_meaning_gate_stack

    return build_meaning_gate_stack(
        prefer=config.meaning_gate_mode,
        allow_lexical=config.allow_lexical_gate,
        strict_semantic_similarity=config.min_semantic_similarity,
        relaxed_semantic_similarity=config.relaxed_semantic_similarity,
    )


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
    gate_stack: MeaningGateStack,
    best_of_n: int,
    enable_tells_tiebreak: bool,
    tells_epsilon: float,
) -> tuple[str, float, float, list[LockRecord], bool, int | None, list[str]]:
    locked, lock = extract_locks(original)
    lock_meta = [
        LockRecord(id=identifier, text=text, ok=True)
        for identifier, text, _ok in lock_records(lock, original)
    ]
    before = detector.score(original).score
    best_text = original
    best_score = before
    kept = False
    best_tells: int | None = None
    best_vetoes: list[str] = []
    try:
        candidates = rewriter.rewrite(locked, n=best_of_n)
    except Exception:
        candidates = []

    valid: list[tuple[str, float, int, list[LockRecord], float, list[str]]] = []
    for raw in candidates:
        if not sentinels_preserved(locked, raw):
            continue
        try:
            restored = restore_locks(raw, lock, strict=True)
        except PreserveLockError:
            continue
        result = gate_stack.evaluate(original, restored)
        if not result.preserved:
            continue
        score = detector.score(restored).score
        tells = int(score_tells(restored)["tells"]) if enable_tells_tiebreak else 0
        valid.append((restored, score, tells, lock_meta, result.similarity, result.vetoes))

    if valid:
        best_score_value = min(item[1] for item in valid)
        pool = [item for item in valid if item[1] <= best_score_value + tells_epsilon]
        chosen = min(pool, key=lambda item: (item[2], item[1]))
        best_text, best_score, best_tells, lock_meta, _similarity, best_vetoes = chosen
        kept = best_text != original
        lock_meta = [
            LockRecord(id=identifier, text=text, ok=present)
            for identifier, text, present in lock_records(lock, best_text)
        ]

    return best_text, before, best_score, lock_meta, kept, best_tells, best_vetoes


def humanize(
    text: str,
    *,
    detector: Detector,
    rewriter: Rewriter,
    semantic_gate: SemanticGate | None = None,
    meaning_gate_stack: MeaningGateStack | None = None,
    config: EngineConfig | None = None,
) -> RunReport:
    """Run the score → flag → lock → rewrite → gate → rescore loop."""
    if not isinstance(text, str) or not text.strip():
        raise InputError("text cannot be empty")
    settings = config or EngineConfig()
    original_input = text
    hidden_removed = 0
    if settings.scrub_input:
        text, scrub_report = scrub_text(text)
        hidden_removed = scrub_report.hidden_removed
        if not text.strip():
            raise InputError("text cannot be empty after scrub")

    gate_stack = meaning_gate_stack or _build_gate_stack(settings, semantic_gate)

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
    passed_verdict = score_before <= settings.verdict_score

    if score_before <= settings.target_score:
        report = RunReport(
            input_text=original_input,
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
            meaning_gate=gate_stack.name,
            passed_verdict=passed_verdict,
            flagged=not passed_verdict,
            hidden_removed=hidden_removed,
        )
        if settings.verify_detectors:
            report.verification = run_verification(
                input_text=original_input,
                output_text=report.output_text,
                detectors=settings.verify_detectors,
                threshold=settings.verify_threshold,
                verify_on_input=settings.verify_on_input,
            )
        return report

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
            rewritten, before, after, locks, kept, tells_after, vetoes = _pick_candidate(
                span.text,
                rewriter=rewriter,
                detector=detector,
                gate_stack=gate_stack,
                best_of_n=settings.best_of_n,
                enable_tells_tiebreak=settings.enable_tells_tiebreak,
                tells_epsilon=settings.tells_tiebreak_epsilon,
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
                    tells_after=tells_after,
                    gate_vetoes=vetoes,
                )
            )

        if accepted == 0:
            sentence_reports = round_reports
            stop_reason = "all_candidates_rejected"
            break

        current = reassemble(current, replacements)
        doc_result = gate_stack.evaluate(text, current)
        if not doc_result.preserved:
            stop_reason = "all_candidates_rejected"
            sentence_reports = round_reports
            break

        current_score = detector.score(current).score
        sentence_reports = round_reports
        if current_score < best_score:
            best_text = current
            best_score = current_score
            best_similarity = doc_result.similarity
        if current_score <= settings.target_score:
            stop_reason = "passed"
            break
    else:
        stop_reason = "max_rounds"

    passed_verdict = best_score <= settings.verdict_score
    report = RunReport(
        input_text=original_input,
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
        meaning_gate=gate_stack.name,
        passed_verdict=passed_verdict,
        flagged=not passed_verdict,
        hidden_removed=hidden_removed,
    )
    if settings.verify_detectors:
        report.verification = run_verification(
            input_text=original_input,
            output_text=report.output_text,
            detectors=settings.verify_detectors,
            threshold=settings.verify_threshold,
            verify_on_input=settings.verify_on_input,
        )
    return report
