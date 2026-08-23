"""Sentence segmentation that preserves character offsets."""

from __future__ import annotations

from dataclasses import dataclass

from adh.exceptions import InputError

_SEGMENTER = None


@dataclass(frozen=True)
class SentenceSpan:
    text: str
    start: int
    end: int

    @property
    def stripped(self) -> str:
        return self.text.strip()


def _segmenter():
    global _SEGMENTER
    if _SEGMENTER is None:
        import pysbd

        _SEGMENTER = pysbd.Segmenter(language="en", clean=False)
    return _SEGMENTER


def split_sentences(text: str) -> list[SentenceSpan]:
    """Split ``text`` into sentence spans with offsets into the original string.

    Whitespace-only input raises ``InputError``. A document with no terminal
    punctuation is treated as a single sentence.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise InputError("text cannot be empty")

    raw_sentences = [item for item in _segmenter().segment(text) if item.strip()]
    if not raw_sentences:
        raise InputError("no sentences could be extracted")

    spans: list[SentenceSpan] = []
    cursor = 0
    for sentence in raw_sentences:
        start = text.find(sentence, cursor)
        if start < 0:
            # pysbd can normalize whitespace; fall back to a stripped search.
            start = text.find(sentence.strip(), cursor)
            if start < 0:
                start = cursor
                end = min(len(text), start + len(sentence))
            else:
                end = start + len(sentence.strip())
        else:
            end = start + len(sentence)
        spans.append(SentenceSpan(text=text[start:end], start=start, end=end))
        cursor = end
    return spans


def reassemble(original: str, replacements: dict[int, str]) -> str:
    """Rebuild ``original`` after substituting sentences by index."""
    spans = split_sentences(original)
    if not spans:
        return original
    if any(index < 0 or index >= len(spans) for index in replacements):
        raise InputError("replacement index is out of range")

    pieces: list[str] = []
    cursor = 0
    for index, span in enumerate(spans):
        pieces.append(original[cursor : span.start])
        pieces.append(replacements.get(index, span.text))
        cursor = span.end
    pieces.append(original[cursor:])
    return "".join(pieces)
