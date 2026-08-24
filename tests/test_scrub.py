from __future__ import annotations

from adh.scrub import scrub_text


def test_scrub_removes_zero_width() -> None:
    dirty = "hello\u200bworld"
    cleaned, report = scrub_text(dirty)
    assert "\u200b" not in cleaned
    assert report.hidden_removed >= 1
