"""Extract and restore factual spans so a rewriter cannot mutate them."""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field

from adh.exceptions import PreserveLockError

_CODE_FENCE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_URL = re.compile(r"https?://[^\s<>)'\"]+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_ISBN = re.compile(r"\b(?:ISBN(?:-1[03])?:?\s*)?(?:97[89][-\s]?)?(?:\d[-\s]?){9}[\dX]\b", re.IGNORECASE)
_QUOTED = re.compile(r"(?<!\w)([\"“])(.+?)\1")
_PERCENT = re.compile(r"\b\d+(?:\.\d+)?%")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?(?![\w.%])")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
_PROPER_NOUN = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

_PATTERNS: tuple[re.Pattern[str], ...] = (
    _CODE_FENCE,
    _INLINE_CODE,
    _URL,
    _EMAIL,
    _DOI,
    _ISBN,
    _QUOTED,
    _PERCENT,
    _YEAR,
    _NUMBER,
    _ACRONYM,
    _PROPER_NOUN,
)

_SENTINEL_RE = re.compile(r"__LOCK_[A-Z0-9]+_\d{3,}__")


@dataclass(frozen=True)
class LockedSpan:
    sentinel: str
    text: str
    start: int
    end: int


@dataclass
class PreserveLock:
    token: str
    spans: list[LockedSpan] = field(default_factory=list)

    @property
    def mapping(self) -> dict[str, str]:
        return {span.sentinel: span.text for span in self.spans}


def _choose_token(text: str) -> str:
    for _ in range(8):
        token = secrets.token_hex(4).upper()
        if token not in text and f"__LOCK_{token}_" not in text:
            return token
    raise PreserveLockError("could not allocate a unique lock token")


def _overlaps(start: int, end: int, taken: list[tuple[int, int]]) -> bool:
    for taken_start, taken_end in taken:
        if start < taken_end and end > taken_start:
            return True
    return False


def extract_locks(text: str) -> tuple[str, PreserveLock]:
    """Replace lockable spans with unique sentinels.

    Empty input is returned unchanged. Overlapping matches are skipped so
    longer, earlier patterns (code fences, URLs) win.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if text == "":
        return "", PreserveLock(token="")

    token = _choose_token(text)
    taken: list[tuple[int, int]] = []
    candidates: list[tuple[int, int, str]] = []

    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if end <= start:
                continue
            if _overlaps(start, end, taken):
                continue
            taken.append((start, end))
            candidates.append((start, end, match.group(0)))

    candidates.sort(key=lambda item: item[0])
    spans: list[LockedSpan] = []
    pieces: list[str] = []
    cursor = 0
    for index, (start, end, original) in enumerate(candidates, start=1):
        sentinel = f"__LOCK_{token}_{index:03d}__"
        pieces.append(text[cursor:start])
        pieces.append(sentinel)
        spans.append(LockedSpan(sentinel=sentinel, text=original, start=start, end=end))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), PreserveLock(token=token, spans=spans)


def restore_locks(text: str, lock: PreserveLock, *, strict: bool = True) -> str:
    """Put locked spans back. Reject missing or mutated sentinels when strict."""
    if lock.token == "" and not lock.spans:
        return text

    restored = text
    missing: list[str] = []
    for span in lock.spans:
        if span.sentinel not in restored:
            missing.append(span.sentinel)
            continue
        restored = restored.replace(span.sentinel, span.text, 1)

    leftover = _SENTINEL_RE.findall(restored)
    if leftover:
        raise PreserveLockError(
            f"rewriter introduced or mutated lock sentinels: {leftover}"
        )
    if missing and strict:
        raise PreserveLockError(
            f"rewriter dropped locked spans: {missing}"
        )
    return restored


def lock_records(lock: PreserveLock, restored_text: str) -> list[tuple[str, str, bool]]:
    """Return (id, original, still_present) tuples for reporting."""
    records = []
    for span in lock.spans:
        identifier = span.sentinel.split("_")[-1].rstrip("_")
        records.append((identifier, span.text, span.text in restored_text))
    return records
