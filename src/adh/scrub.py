"""Pre-loop Unicode scrub before preserve locks."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_WATERMARK_CHARS = re.compile(
    "[\u200b\u200c\u200d\u2060\ufeff\u00ad\u2061-\u2064\u202a-\u202e\u2066-\u2069]"
)
_EXOTIC_SPACES = re.compile("[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")


@dataclass(frozen=True)
class ScrubReport:
    hidden_removed: int
    changed: bool


def count_hidden(text: str) -> int:
    return len(_WATERMARK_CHARS.findall(text)) + len(_EXOTIC_SPACES.findall(text))


def scrub_text(text: str) -> tuple[str, ScrubReport]:
    before = count_hidden(text)
    cleaned = _WATERMARK_CHARS.sub("", text)
    cleaned = _EXOTIC_SPACES.sub(" ", cleaned)
    cleaned = unicodedata.normalize("NFC", cleaned)
    after = count_hidden(cleaned)
    return cleaned, ScrubReport(hidden_removed=max(0, before - after + (1 if cleaned != text else 0)), changed=cleaned != text)
