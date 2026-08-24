"""Deletion allowance gate."""

from __future__ import annotations

import re

_LENGTH_WORD = re.compile(r"[A-Za-z0-9']+")
_LENGTH_SLACK_WORDS = 10
_LENGTH_SLACK_SHARE = 0.10


def words_lost(source: str, candidate: str) -> int:
    return len(_LENGTH_WORD.findall(source)) - len(_LENGTH_WORD.findall(candidate))


def deletion_allowance(source: str) -> float:
    return max(_LENGTH_SLACK_WORDS, _LENGTH_SLACK_SHARE * len(_LENGTH_WORD.findall(source)))


def deletion_kept(source: str, candidate: str) -> bool:
    return words_lost(source, candidate) <= deletion_allowance(source)
