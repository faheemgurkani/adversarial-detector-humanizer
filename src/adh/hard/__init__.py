"""Token-guided hard mode rewriter."""

from __future__ import annotations

from adh.detectors.base import Detector
from adh.exceptions import HardModeUnavailableError, PreserveLockError
from adh.gates.stack import MeaningGateStack
from adh.hard.adapters import DetectorScoreAdapter
from adh.hard.guided_decode import guided_decode_text, hard_mode_available
from adh.preserve import PreserveLock, extract_locks, restore_locks, sentinels_preserved

HARD_PROMPT = """Rephrase the sentence below. Preserve meaning and every __LOCK_...__ token exactly.
Output only the rewritten sentence.

Sentence: {sentence}
"""


class HardModeRewriter:
    name = "hard-guided"

    def __init__(self, *, model_id: str | None = None, top_k: int = 16) -> None:
        if not hard_mode_available():
            raise HardModeUnavailableError(
                "hard mode is unavailable. Install [hard] extras and GPU support."
            )
        self.model_id = model_id
        self.top_k = top_k

    def rewrite_sentence(
        self,
        original: str,
        *,
        detector: Detector,
        gate_stack: MeaningGateStack,
    ) -> str | None:
        locked, lock = extract_locks(original)
        prompt = HARD_PROMPT.format(sentence=locked)
        adapter = DetectorScoreAdapter(detector)
        raw = guided_decode_text(
            prompt,
            detector_adapter=adapter,
            model_id=self.model_id,
            top_k=self.top_k,
        )
        if not sentinels_preserved(locked, raw):
            return None
        try:
            restored = restore_locks(raw, lock, strict=True)
        except PreserveLockError:
            return None
        if not gate_stack.evaluate(original, restored).preserved:
            return None
        if restored.strip() == original.strip():
            return None
        return restored
