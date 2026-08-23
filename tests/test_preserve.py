from __future__ import annotations

import pytest

from adh.exceptions import PreserveLockError
from adh.preserve import extract_locks, restore_locks


def test_empty_text_is_noop() -> None:
    locked, lock = extract_locks("")
    assert locked == ""
    assert lock.spans == []
    assert restore_locks(locked, lock) == ""


def test_locks_url_email_number_and_year() -> None:
    text = "See https://example.com/a and write 42.5% in 2024 to ada@example.com."
    locked, lock = extract_locks(text)
    assert "https://example.com/a" not in locked
    assert "42.5%" not in locked
    assert "2024" not in locked
    assert "ada@example.com" not in locked
    assert restore_locks(locked, lock) == text
    assert all(span.sentinel.startswith("__LOCK_") for span in lock.spans)


def test_code_fence_wins_over_inner_numbers() -> None:
    text = "Before\n```\nvalue = 99\n```\nAfter"
    locked, lock = extract_locks(text)
    assert "99" not in locked
    assert restore_locks(locked, lock) == text
    assert any("```" in span.text for span in lock.spans)


def test_rejects_missing_sentinel() -> None:
    text = "Rate was 12%."
    locked, lock = extract_locks(text)
    with pytest.raises(PreserveLockError, match="dropped"):
        restore_locks("Rate was gone.", lock, strict=True)


def test_rejects_mutated_sentinel() -> None:
    text = "Rate was 12%."
    _locked, lock = extract_locks(text)
    with pytest.raises(PreserveLockError, match="mutated"):
        restore_locks("Rate was __LOCK_DEAD_001__.", lock, strict=True)


def test_quoted_span_is_locked() -> None:
    text = 'He said "keep this quote" today.'
    locked, lock = extract_locks(text)
    assert "keep this quote" not in locked
    assert restore_locks(locked, lock) == text
