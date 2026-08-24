from __future__ import annotations

from adh.tells import score_tells


def test_score_tells_counts_formulaic_transition() -> None:
    result = score_tells("Furthermore, it is important to note the result.")
    assert result["tells"] > 0
