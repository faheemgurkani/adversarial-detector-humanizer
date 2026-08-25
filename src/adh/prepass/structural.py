"""Structural translation pre-pass on flagged paragraph windows."""

from __future__ import annotations

import re
from dataclasses import dataclass

from adh.exceptions import PreserveLockError
from adh.gates.stack import MeaningGateStack
from adh.prepass.translate import Translator, round_trip_translate
from adh.preserve import extract_locks, restore_locks, sentinels_preserved


@dataclass(frozen=True)
class ParagraphSpan:
    text: str
    start: int
    end: int


def split_paragraphs(text: str) -> list[ParagraphSpan]:
    """Split on blank lines while preserving document offsets."""
    if not text.strip():
        return []
    pattern = re.compile(r"\n\s*\n")
    spans: list[ParagraphSpan] = []
    cursor = 0
    for match in pattern.finditer(text):
        chunk = text[cursor : match.start()]
        if chunk.strip():
            start = cursor + len(chunk) - len(chunk.lstrip())
            end = cursor + len(chunk.rstrip())
            spans.append(ParagraphSpan(text=text[start:end], start=start, end=end))
        cursor = match.end()
    tail = text[cursor:]
    if tail.strip():
        start = cursor + len(tail) - len(tail.lstrip())
        end = cursor + len(tail.rstrip())
        spans.append(ParagraphSpan(text=text[start:end], start=start, end=end))
    if not spans:
        spans.append(ParagraphSpan(text=text.strip(), start=0, end=len(text.rstrip())))
    return spans


def paragraph_indices_for_sentences(
    paragraphs: list[ParagraphSpan],
    sentence_spans: list[tuple[int, int]],
    flagged: list[int],
) -> list[int]:
    """Return paragraph indices that contain at least one flagged sentence."""
    hits: list[int] = []
    for para_index, paragraph in enumerate(paragraphs):
        for sentence_index in flagged:
            start, end = sentence_spans[sentence_index]
            if start >= paragraph.start and end <= paragraph.end:
                hits.append(para_index)
                break
    return hits


def replace_paragraph(text: str, paragraph: ParagraphSpan, replacement: str) -> str:
    return f"{text[: paragraph.start]}{replacement}{text[paragraph.end :]}"


class StructuralPrepass:
    """EN→lang→EN round trip on flagged paragraphs with lock + gate safety."""

    def __init__(
        self,
        *,
        translator: Translator,
        lang: str = "fi",
        max_paragraphs: int = 2,
    ) -> None:
        self.translator = translator
        self.lang = lang
        self.max_paragraphs = max_paragraphs

    def apply_paragraph(
        self,
        paragraph: str,
        *,
        original_paragraph: str,
        gate_stack: MeaningGateStack,
    ) -> str | None:
        locked, lock = extract_locks(paragraph)
        try:
            translated = round_trip_translate(
                locked,
                lang=self.lang,
                translator=self.translator,
            )
        except Exception:
            return None
        if not sentinels_preserved(locked, translated):
            return None
        try:
            restored = restore_locks(translated, lock, strict=True)
        except PreserveLockError:
            return None
        result = gate_stack.evaluate(original_paragraph, restored)
        if not result.preserved:
            return None
        if restored.strip() == original_paragraph.strip():
            return None
        return restored

    def apply_document(
        self,
        text: str,
        *,
        flagged_sentence_indices: list[int],
        sentence_spans: list[tuple[int, int]],
        gate_stack: MeaningGateStack,
    ) -> tuple[str, int, set[int]]:
        """Return updated text, paragraphs changed, and sentence indices to reset."""
        paragraphs = split_paragraphs(text)
        if not paragraphs or not flagged_sentence_indices:
            return text, 0, set()

        target_paragraphs = paragraph_indices_for_sentences(
            paragraphs,
            sentence_spans,
            flagged_sentence_indices,
        )[: self.max_paragraphs]
        if not target_paragraphs:
            return text, 0, set()

        current = text
        changed = 0
        reset_indices: set[int] = set()

        for para_index in target_paragraphs:
            paragraphs = split_paragraphs(current)
            if para_index >= len(paragraphs):
                continue
            paragraph = paragraphs[para_index]
            replacement = self.apply_paragraph(
                paragraph.text,
                original_paragraph=paragraph.text,
                gate_stack=gate_stack,
            )
            if replacement is None:
                continue
            current = replace_paragraph(current, paragraph, replacement)
            changed += 1
            for sentence_index, (start, end) in enumerate(sentence_spans):
                if start >= paragraph.start and end <= paragraph.end:
                    reset_indices.add(sentence_index)

        return current, changed, reset_indices
