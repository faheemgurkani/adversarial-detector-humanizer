"""Closed-loop detector-guided humanizer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from adh.audit import build_detector_breakdown
from adh.detectors.base import Detector
from adh.exceptions import InputError, PreserveLockError
from adh.factory import load_detector
from adh.gates.stack import MeaningGateStack
from adh.preserve import extract_locks, lock_records, restore_locks, sentinels_preserved
from adh.ranking import blend_score
from adh.report import CandidateScoreDebug, LockRecord, RunReport, SentenceReport, StopReason
from adh.rewriter import Rewriter, rewrite_candidates_for
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
    deploy_detectors: list[str] = Field(default_factory=list)
    run_detector_breakdown: bool = True
    enable_logprob_blend: bool = True
    logprob_blend_weight: float = Field(default=0.15, ge=0.0)
    detector_blend_weight: float = Field(default=1.0, ge=0.0)
    blend_epsilon: float = Field(default=2.0, ge=0.0, le=100.0)
    hard_mode: bool = False
    hard_mode_max_sentences: int = Field(default=1, ge=0, le=5)
    prepass: Literal["none", "structural"] = "none"
    prepass_lang: str = "fi"
    prepass_max_paragraphs: int = Field(default=2, ge=0, le=10)
    prepass_backend: str = "llm"


def _maybe_structural_prepass(
    text: str,
    *,
    settings: EngineConfig,
    detector: Detector,
    gate_stack: MeaningGateStack,
    translator=None,
) -> tuple[str, bool, int]:
    if settings.prepass != "structural" or settings.prepass_max_paragraphs == 0:
        return text, False, 0

    spans = split_sentences(text)
    span_scores = [result.score for result in detector.score_spans([span.text for span in spans])]
    flagged = _flag_indices(
        spans,
        span_scores,
        threshold=settings.sentence_threshold,
        top_k=settings.top_k_fallback,
        max_rewrite_ratio=settings.max_rewrite_ratio,
    )
    if not flagged:
        return text, False, 0

    from adh.prepass import StructuralPrepass, load_translator

    prepass = StructuralPrepass(
        translator=translator or load_translator(settings.prepass_backend),
        lang=settings.prepass_lang,
        max_paragraphs=settings.prepass_max_paragraphs,
    )
    sentence_spans = [(span.start, span.end) for span in spans]
    updated, changed, _reset = prepass.apply_document(
        text,
        flagged_sentence_indices=flagged,
        sentence_spans=sentence_spans,
        gate_stack=gate_stack,
    )
    if changed == 0:
        return text, False, 0
    doc_result = gate_stack.evaluate(text, updated)
    if not doc_result.preserved:
        return text, False, 0
    return updated, True, changed


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


def _load_deploy_detectors(names: list[str]) -> list[Detector]:
    return [load_detector(name) for name in names]


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


def _try_hard_rewrite(
    original: str,
    *,
    detector: Detector,
    gate_stack: MeaningGateStack,
    hard_rewriter,
) -> tuple[str, float, bool] | None:
    restored = hard_rewriter.rewrite_sentence(original, detector=detector, gate_stack=gate_stack)
    if restored is None:
        return None
    after = detector.score(restored).score
    return restored, after, True


def _pick_candidate(
    original: str,
    *,
    rewriter: Rewriter,
    detector: Detector,
    gate_stack: MeaningGateStack,
    settings: EngineConfig,
    hard_rewriter=None,
    hard_budget: int = 0,
    history: list[tuple[str, str]] | None = None,
) -> tuple[
    str,
    float,
    float,
    list[LockRecord],
    bool,
    int | None,
    list[str],
    Literal["api", "hard", "none"],
    list[CandidateScoreDebug],
]:
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
    rewrite_mode: Literal["api", "hard", "none"] = "none"
    debug_scores: list[CandidateScoreDebug] = []

    try:
        candidates = rewrite_candidates_for(
            rewriter,
            locked,
            n=settings.best_of_n,
            history=history,
        )
    except Exception:
        candidates = []

    valid: list[tuple[str, float, float, int, float, list[str]]] = []
    for candidate in candidates:
        raw = candidate.text
        if not sentinels_preserved(locked, raw):
            continue
        try:
            restored = restore_locks(raw, lock, strict=True)
        except PreserveLockError:
            continue
        result = gate_stack.evaluate(original, restored)
        if not result.preserved:
            continue
        detector_score = detector.score(restored).score
        blended = blend_score(
            detector_score=detector_score,
            mean_logprob=candidate.mean_logprob,
            source=original,
            candidate=restored,
            detector_blend_weight=settings.detector_blend_weight,
            logprob_blend_weight=settings.logprob_blend_weight,
            enable_logprob_blend=settings.enable_logprob_blend,
        )
        tells = int(score_tells(restored)["tells"]) if settings.enable_tells_tiebreak else 0
        valid.append((restored, detector_score, blended, tells, result.similarity, result.vetoes))
        debug_scores.append(
            CandidateScoreDebug(
                text=restored,
                detector=detector_score,
                logprob=candidate.mean_logprob,
                blend=round(blended, 4),
            )
        )

    if valid:
        best_blend = min(item[2] for item in valid)
        pool = [item for item in valid if item[2] <= best_blend + settings.blend_epsilon]
        chosen = min(pool, key=lambda item: (item[3], item[2], item[1]))
        best_text, best_score, _blend, best_tells, _similarity, best_vetoes = chosen
        kept = best_text != original
        rewrite_mode = "api" if kept else "none"
        lock_meta = [
            LockRecord(id=identifier, text=text, ok=present)
            for identifier, text, present in lock_records(lock, best_text)
        ]

    if not kept and settings.hard_mode and hard_budget > 0 and hard_rewriter is not None:
        hard_result = _try_hard_rewrite(
            original,
            detector=detector,
            gate_stack=gate_stack,
            hard_rewriter=hard_rewriter,
        )
        if hard_result is not None:
            best_text, best_score, kept = hard_result
            rewrite_mode = "hard"
            lock_meta = [
                LockRecord(id=identifier, text=text, ok=present)
                for identifier, text, present in lock_records(lock, best_text)
            ]

    return (
        best_text,
        before,
        best_score,
        lock_meta,
        kept,
        best_tells,
        best_vetoes,
        rewrite_mode,
        debug_scores,
    )


def _attach_post_run_reports(
    report: RunReport,
    *,
    settings: EngineConfig,
    original_input: str,
    guidance: Detector,
) -> RunReport:
    if settings.verify_detectors:
        report.verification = run_verification(
            input_text=original_input,
            output_text=report.output_text,
            detectors=settings.verify_detectors,
            threshold=settings.verify_threshold,
            verify_on_input=settings.verify_on_input,
        )
    if settings.deploy_detectors and settings.run_detector_breakdown:
        deploy = _load_deploy_detectors(settings.deploy_detectors)
        report.detector_breakdown = build_detector_breakdown(
            original_input,
            report.output_text,
            guidance=guidance,
            guidance_before=report.score_before,
            guidance_after=report.score_after,
            deploy=deploy,
        )
    return report


def humanize(
    text: str,
    *,
    detector: Detector,
    rewriter: Rewriter,
    semantic_gate: SemanticGate | None = None,
    meaning_gate_stack: MeaningGateStack | None = None,
    config: EngineConfig | None = None,
    hard_rewriter=None,
    prepass_translator=None,
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
    hard_budget = settings.hard_mode_max_sentences if settings.hard_mode else 0
    prepass_applied = False
    prepass_paragraphs = 0
    rewrite_history: dict[int, list[tuple[str, str]]] = {}

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
            prepass_applied=False,
            prepass_paragraphs=0,
        )
        return _attach_post_run_reports(
            report,
            settings=settings,
            original_input=original_input,
            guidance=detector,
        )

    text, prepass_applied, prepass_paragraphs = _maybe_structural_prepass(
        text,
        settings=settings,
        detector=detector,
        gate_stack=gate_stack,
        translator=prepass_translator,
    )
    initial_score_before = score_before
    current = text
    if prepass_applied:
        rewrite_history.clear()
        current = text
        best_text = text
        current_score = detector.score(text).score
        best_score = current_score
        best_similarity = round(gate_stack.evaluate(original_input, text).similarity, 4)
        if current_score <= settings.target_score:
            passed_verdict = current_score <= settings.verdict_score
            report = RunReport(
                input_text=original_input,
                output_text=text,
                detector=detector.name,
                score_before=initial_score_before,
                score_after=current_score,
                semantic_similarity=best_similarity,
                rounds=0,
                stop_reason="passed",
                sentences=[],
                locks=[],
                flagged_count=0,
                rewrite_ratio=0.0,
                meaning_gate=gate_stack.name,
                passed_verdict=passed_verdict,
                flagged=not passed_verdict,
                hidden_removed=hidden_removed,
                prepass_applied=True,
                prepass_paragraphs=prepass_paragraphs,
            )
            return _attach_post_run_reports(
                report,
                settings=settings,
                original_input=original_input,
                guidance=detector,
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
            prior = rewrite_history.get(index, [])
            history = prior[-1:] if prior else None
            (
                rewritten,
                before,
                after,
                locks,
                kept,
                tells_after,
                vetoes,
                rewrite_mode,
                candidate_scores,
            ) = _pick_candidate(
                span.text,
                rewriter=rewriter,
                detector=detector,
                gate_stack=gate_stack,
                settings=settings,
                hard_rewriter=hard_rewriter,
                hard_budget=hard_budget,
                history=history,
            )
            if rewrite_mode == "hard" and kept:
                hard_budget -= 1
            lock_report.extend(locks)
            if kept:
                replacements[index] = rewritten
                accepted += 1
                rewrite_history.setdefault(index, []).append((span.text, rewritten))
            round_reports.append(
                SentenceReport(
                    i=index,
                    round=rounds,
                    original=span.text,
                    rewritten=rewritten,
                    score_before=before,
                    score_after=after,
                    kept=kept,
                    start=span.start,
                    end=span.end,
                    tells_after=tells_after,
                    gate_vetoes=vetoes,
                    rewrite_mode=rewrite_mode,
                    candidate_scores=candidate_scores,
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
        score_before=initial_score_before,
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
        prepass_applied=prepass_applied,
        prepass_paragraphs=prepass_paragraphs,
    )
    return _attach_post_run_reports(
        report,
        settings=settings,
        original_input=original_input,
        guidance=detector,
    )
