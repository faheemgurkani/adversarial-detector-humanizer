"""Quantity retention: source numerals must survive in rewrites."""

from __future__ import annotations

import re

from adh.preserve import SENTINEL_RE

_NUMBER_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")

_WORDS: dict[str, tuple[str, ...]] = {
    "0": ("zero", "no", "none"),
    "1": ("one", "a single"),
    "2": ("two", "both", "a pair"),
    "3": ("three",),
    "4": ("four",),
    "5": ("five",),
    "6": ("six",),
    "7": ("seven",),
    "8": ("eight",),
    "9": ("nine",),
    "10": ("ten",),
    "11": ("eleven",),
    "12": ("twelve", "a dozen"),
    "13": ("thirteen",),
    "14": ("fourteen",),
    "15": ("fifteen",),
    "16": ("sixteen",),
    "17": ("seventeen",),
    "18": ("eighteen",),
    "19": ("nineteen",),
    "20": ("twenty",),
}

_LIST_MARKER_RE = re.compile(r"(?m)^[ \t]*\d{1,2}[.)](?=\s)")

_VAGUE_QUANTIFIERS = re.compile(
    r"(?<!\w)(?:few|several|some|many|a few|a couple|couple|handful|numerous)(?!\w)",
    re.IGNORECASE,
)


def _canonical(value: str) -> str:
    text = value.replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return str(number)


def _numbers(text: str) -> list[str]:
    cleaned = _LIST_MARKER_RE.sub(" ", SENTINEL_RE.sub(" ", text))
    return [_canonical(match.replace(",", "")) for match in _NUMBER_RE.findall(cleaned)]


def _says_word(text: str, word: str) -> bool:
    pattern = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)
    return bool(pattern.search(text))


def missing_numbers(source: str, candidate: str) -> list[str]:
    candidate_values = set(_numbers(candidate))
    candidate_lower = candidate.lower()
    seen: set[str] = set()
    missing: list[str] = []
    for value in _numbers(source):
        if value in seen:
            continue
        seen.add(value)
        if value in candidate_values:
            continue
        if any(_says_word(candidate_lower, word) for word in _WORDS.get(value, ())):
            continue
        missing.append(value)
    return missing


def numbers_kept(source: str, candidate: str) -> bool:
    if missing_numbers(source, candidate):
        return False
    source_values = set(_numbers(source))
    if not source_values:
        return True
    if _VAGUE_QUANTIFIERS.search(candidate) and not _numbers(candidate):
        return False
    return True
