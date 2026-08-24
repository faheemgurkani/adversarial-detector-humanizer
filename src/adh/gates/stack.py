"""Composite meaning gate stack."""

from __future__ import annotations

from dataclasses import dataclass

from adh.gates.base import GateResult
from adh.gates.deletion import deletion_kept
from adh.gates.entailment import (
    DEFAULT_CONTRADICTION_BAR,
    DEFAULT_ENTAILMENT_FLOOR,
    nli_available,
    nli_passes,
)
from adh.gates.hedges import certainty_kept, polarity_kept
from adh.gates.numerals import numbers_kept
from adh.gates.roles import role_swap, roles_available
from adh.gates.scaffolding import strip_scaffolding
from adh.semantic import SemanticGate


@dataclass
class MeaningGateStack:
    semantic_gate: SemanticGate
    strict_semantic_similarity: float = 0.88
    relaxed_semantic_similarity: float = 0.30
    contradiction_bar: float = DEFAULT_CONTRADICTION_BAR
    entailment_floor: float = DEFAULT_ENTAILMENT_FLOOR
    enable_nli: bool | None = None
    enable_roles: bool | None = None

    @property
    def name(self) -> str:
        if self._use_nli():
            return "full" if self._use_roles() else "nli"
        return self.semantic_gate.name

    def _use_nli(self) -> bool:
        if self.enable_nli is False:
            return False
        if self.enable_nli is True:
            return nli_available()
        return nli_available()

    def _use_roles(self) -> bool:
        if self.enable_roles is False:
            return False
        if self.enable_roles is True:
            return roles_available()
        return roles_available()

    def evaluate(self, source: str, candidate: str) -> GateResult:
        left = strip_scaffolding(source)
        right = strip_scaffolding(candidate)
        vetoes: list[str] = []

        if not numbers_kept(left, right):
            vetoes.append("numerals")
        if not certainty_kept(left, right):
            vetoes.append("hedges")
        if not polarity_kept(left, right):
            vetoes.append("polarity")
        if not deletion_kept(left, right):
            vetoes.append("deletion")

        if vetoes:
            return GateResult(
                preserved=False,
                similarity=self.semantic_gate.similarity(left, right),
                vetoes=vetoes,
                meaning_gate="mechanical",
            )

        swapped = role_swap(left, right) if self._use_roles() else None
        if swapped is True:
            return GateResult(
                preserved=False,
                similarity=self.semantic_gate.similarity(left, right),
                vetoes=["roles"],
                meaning_gate="roles",
            )

        similarity = self.semantic_gate.similarity(left, right)
        if self._use_nli():
            nli_ok = nli_passes(
                left,
                right,
                contradiction_bar=self.contradiction_bar,
                entailment_floor=self.entailment_floor,
            )
            if nli_ok is False:
                return GateResult(
                    preserved=False,
                    similarity=similarity,
                    vetoes=["nli"],
                    meaning_gate="nli",
                )
            if nli_ok is True:
                if similarity < self.relaxed_semantic_similarity:
                    return GateResult(
                        preserved=False,
                        similarity=similarity,
                        vetoes=["similarity"],
                        meaning_gate="nli",
                    )
                return GateResult(
                    preserved=True,
                    similarity=similarity,
                    meaning_gate="nli",
                )

        if similarity < self.strict_semantic_similarity:
            return GateResult(
                preserved=False,
                similarity=similarity,
                vetoes=["similarity"],
                meaning_gate=self.semantic_gate.name,
            )
        return GateResult(
            preserved=True,
            similarity=similarity,
            meaning_gate=self.semantic_gate.name,
        )
