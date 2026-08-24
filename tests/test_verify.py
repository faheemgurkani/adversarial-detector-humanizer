from __future__ import annotations

from unittest.mock import patch

from adh.engine import EngineConfig, humanize
from adh.rewriter import IdentityRewriter
from adh.semantic import LexicalSemanticGate
from adh.report import VerificationReport
from adh.verify import run_verification
from tests.conftest import CueDetector


def test_run_verification_passes_all(monkeypatch) -> None:
    class FakeRemote:
        name = "pangram"

        def score(self, text: str):
            from adh.detectors.base import probability_to_result

            return probability_to_result(0.2)

    with patch("adh.verify.load_detector", return_value=FakeRemote()):
        report = run_verification(
            input_text="Before text.",
            output_text="After text.",
            detectors=["pangram"],
            threshold=45.0,
        )
    assert report.passes_all
    assert report.results[0].name == "pangram"


def test_humanize_attaches_verification(monkeypatch) -> None:
    class FakeRemote:
        name = "pangram"

        def score(self, text: str):
            from adh.detectors.base import probability_to_result

            return probability_to_result(0.2)

    with patch("adh.engine.run_verification") as verify:
        verify.return_value = VerificationReport(threshold=45.0, results=[], passes_all=True)
        humanize(
            "This is a human sentence.",
            detector=CueDetector(),
            rewriter=IdentityRewriter(),
            semantic_gate=LexicalSemanticGate(),
            config=EngineConfig(verify_detectors=["pangram"]),
        )
        verify.assert_called_once()
