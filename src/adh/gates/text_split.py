"""Chunk alignment for long-text NLI scoring."""

from __future__ import annotations

import difflib


def aligned_chunks(left: str, right: str, *, max_chars: int = 900) -> list[tuple[str, str]]:
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return [(left, right)]
    if len(left) <= max_chars and len(right) <= max_chars:
        return [(left, right)]

    left_words = left.split()
    right_words = right.split()
    matcher = difflib.SequenceMatcher(None, left_words, right_words, autojunk=False)
    pairs: list[tuple[str, str]] = []
    left_buf: list[str] = []
    right_buf: list[str] = []
    left_len = 0
    right_len = 0

    def flush() -> None:
        nonlocal left_len, right_len
        if left_buf or right_buf:
            pairs.append((" ".join(left_buf), " ".join(right_buf)))
        left_buf.clear()
        right_buf.clear()
        left_len = 0
        right_len = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        segment_left = " ".join(left_words[i1:i2])
        segment_right = " ".join(right_words[j1:j2])
        next_left = left_len + len(segment_left)
        next_right = right_len + len(segment_right)
        if left_buf and (next_left > max_chars or next_right > max_chars):
            flush()
        if segment_left or segment_right:
            left_buf.append(segment_left)
            right_buf.append(segment_right)
            left_len += len(segment_left)
            right_len += len(segment_right)
    flush()
    return pairs or [(left, right)]
