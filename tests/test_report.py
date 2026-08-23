from __future__ import annotations

from adh.report import RunReport


def test_public_dict() -> None:
    report = RunReport(
        input_text="in",
        output_text="out",
        detector="fake",
        score_before=90.0,
        score_after=20.0,
        semantic_similarity=0.91,
        rounds=2,
        stop_reason="passed",
    )
    public = report.to_public_dict()
    assert public["ai_score_before"] == 90.0
    assert public["ai_score_after"] == 20.0
    assert public["semantic_score"] == 0.91
