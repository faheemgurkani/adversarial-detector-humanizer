from __future__ import annotations

from adh.preserve import extract_locks, sentinels_preserved


def test_sentinels_preserved_requires_exact_multiset() -> None:
    source = "Visit https://example.com today."
    locked, _lock = extract_locks(source)
    assert sentinels_preserved(locked, locked)

    mutated = locked.replace("_001__", "_002__")
    assert not sentinels_preserved(locked, mutated)
