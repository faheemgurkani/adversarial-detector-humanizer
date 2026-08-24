"""Shared types for the meaning-gate stack."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GateResult:
    preserved: bool
    similarity: float = 1.0
    vetoes: list[str] = field(default_factory=list)
    meaning_gate: str = "mechanical"
