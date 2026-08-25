from __future__ import annotations

from adh.audit import build_detector_breakdown
from adh.detectors.fake import FakeDetector


def test_transfer_ok_when_guidance_and_deploy_drop() -> None:
    guidance = FakeDetector(document_score=80.0)
    deploy = FakeDetector(document_score=70.0)

    class DroppingDeploy(FakeDetector):
        def score(self, text: str):
            from adh.detectors.base import probability_to_result

            if "ORIGINAL" in text:
                return probability_to_result(0.7)
            return probability_to_result(0.35)

    breakdown = build_detector_breakdown(
        "ORIGINAL text",
        "rewritten text",
        guidance=guidance,
        guidance_before=80.0,
        guidance_after=40.0,
        deploy=[DroppingDeploy()],
    )
    assert breakdown.transfer_ok is True
    assert len(breakdown.entries) == 2


def test_transfer_not_ok_when_deploy_rises() -> None:
    guidance = FakeDetector(document_score=80.0)

    class RisingDeploy(FakeDetector):
        def score(self, text: str):
            from adh.detectors.base import probability_to_result

            if "ORIGINAL" in text:
                return probability_to_result(0.5)
            return probability_to_result(0.8)

    breakdown = build_detector_breakdown(
        "ORIGINAL text",
        "rewritten text",
        guidance=guidance,
        guidance_before=80.0,
        guidance_after=40.0,
        deploy=[RisingDeploy()],
    )
    assert breakdown.transfer_ok is False


def test_empty_deploy_list_transfer_none() -> None:
    guidance = FakeDetector(document_score=80.0)
    breakdown = build_detector_breakdown(
        "input",
        "output",
        guidance=guidance,
        guidance_before=80.0,
        guidance_after=40.0,
        deploy=[],
    )
    assert breakdown.transfer_ok is None
