"""Candidate ranking helpers (detector + logprob blend)."""

from __future__ import annotations

import math


def normalize_logprob(mean_logprob: float | None) -> float:
    if mean_logprob is None:
        return 0.0
    # Map typical logprobs (-0..-8) to a 0-100 penalty scale for blending.
    return max(0.0, min(100.0, -mean_logprob * 12.5))


def length_penalty(source: str, candidate: str) -> float:
    source_words = max(1, len(source.split()))
    candidate_words = max(1, len(candidate.split()))
    ratio = candidate_words / source_words
    if 0.75 <= ratio <= 1.35:
        return 0.0
    return min(20.0, abs(ratio - 1.0) * 20.0)


def blend_score(
    *,
    detector_score: float,
    mean_logprob: float | None,
    source: str,
    candidate: str,
    detector_blend_weight: float = 1.0,
    logprob_blend_weight: float = 0.15,
    enable_logprob_blend: bool = True,
) -> float:
    if not enable_logprob_blend or mean_logprob is None or logprob_blend_weight == 0.0:
        return detector_score + length_penalty(source, candidate)
    logprob_penalty = normalize_logprob(mean_logprob)
    return (
        detector_blend_weight * detector_score
        + logprob_blend_weight * logprob_penalty
        + length_penalty(source, candidate)
    )


def mean_token_logprob(choice: dict) -> float | None:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return None
    values: list[float] = []
    for item in content:
        if isinstance(item, dict) and item.get("logprob") is not None:
            values.append(float(item["logprob"]))
            continue
        top = item.get("top_logprobs") if isinstance(item, dict) else None
        if isinstance(top, list) and top and isinstance(top[0], dict):
            if top[0].get("logprob") is not None:
                values.append(float(top[0]["logprob"]))
    if not values:
        return None
    return sum(values) / len(values)
