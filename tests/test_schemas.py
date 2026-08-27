from __future__ import annotations

from adh.ids import new_report_id
from adh.report import RunReport
from adh.schemas import compact_from_report


def test_compact_from_report() -> None:
    report = RunReport(
        input_text="in",
        output_text="out",
        detector="fake",
        score_before=90.0,
        score_after=20.0,
        semantic_similarity=0.91,
        rounds=2,
        stop_reason="passed",
        report_id=new_report_id(),
    )
    compact = compact_from_report(report, metadata={"ticket": "1"})
    assert compact.ai_score_before == 90.0
    assert compact.output_text == "out"
    assert compact.output == "out"
    assert compact.stop_reason == "passed"
    assert compact.agent_hint
    assert compact.metadata == {"ticket": "1"}
