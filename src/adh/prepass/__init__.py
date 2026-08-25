"""Optional pre-loop structural translation on flagged paragraphs."""

from adh.prepass.structural import ParagraphSpan, StructuralPrepass, split_paragraphs
from adh.prepass.translate import (
    GoogleTranslator,
    IdentityTranslator,
    LLMTranslator,
    Translator,
    load_translator,
    round_trip_translate,
)

__all__ = [
    "GoogleTranslator",
    "IdentityTranslator",
    "LLMTranslator",
    "ParagraphSpan",
    "StructuralPrepass",
    "Translator",
    "load_translator",
    "round_trip_translate",
    "split_paragraphs",
]
