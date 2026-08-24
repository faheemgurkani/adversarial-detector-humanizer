"""Meaning gate stack factory."""

from __future__ import annotations

from adh.exceptions import InputError
from adh.gates.stack import MeaningGateStack
from adh.semantic import SemanticGate, build_semantic_gate


def build_meaning_gate_stack(
    *,
    prefer: str = "auto",
    allow_lexical: bool = False,
    strict_semantic_similarity: float = 0.88,
    relaxed_semantic_similarity: float = 0.30,
    contradiction_bar: float = 0.5,
    entailment_floor: float = 0.005,
    enable_nli: bool | None = None,
    enable_roles: bool | None = None,
) -> MeaningGateStack:
    if prefer == "lexical":
        gate: SemanticGate = build_semantic_gate(prefer="lexical")
    elif prefer == "minilm":
        gate = build_semantic_gate(prefer="minilm")
    elif prefer == "auto":
        gate = build_semantic_gate(prefer="auto", allow_lexical=allow_lexical)
    elif prefer == "full":
        gate = build_semantic_gate(prefer="auto", allow_lexical=allow_lexical)
        return MeaningGateStack(
            semantic_gate=gate,
            strict_semantic_similarity=strict_semantic_similarity,
            relaxed_semantic_similarity=relaxed_semantic_similarity,
            contradiction_bar=contradiction_bar,
            entailment_floor=entailment_floor,
            enable_nli=True if enable_nli is None else enable_nli,
            enable_roles=True if enable_roles is None else enable_roles,
        )
    else:
        raise InputError("meaning gate must be auto, minilm, lexical, or full")
    return MeaningGateStack(
        semantic_gate=gate,
        strict_semantic_similarity=strict_semantic_similarity,
        relaxed_semantic_similarity=relaxed_semantic_similarity,
        contradiction_bar=contradiction_bar,
        entailment_floor=entailment_floor,
        enable_nli=enable_nli,
        enable_roles=enable_roles,
    )
