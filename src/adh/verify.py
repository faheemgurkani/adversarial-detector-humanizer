"""Post-loop commercial detector verification."""

from __future__ import annotations

from adh.detectors.base import Detector
from adh.factory import load_detector
from adh.report import DetectorVerifyResult, VerificationReport, score_to_label


def run_verification(
    *,
    input_text: str,
    output_text: str,
    detectors: list[str],
    threshold: float = 45.0,
    verify_on_input: bool = True,
) -> VerificationReport:
    results: list[DetectorVerifyResult] = []
    passes_all = True
    for name in detectors:
        try:
            detector: Detector = load_detector(name)
            before = detector.score(input_text).score if verify_on_input else 0.0
            after = detector.score(output_text).score
            passed = after <= threshold
            if not passed:
                passes_all = False
            results.append(
                DetectorVerifyResult(
                    name=detector.name,
                    score_before=before,
                    score_after=after,
                    label_before=score_to_label(before),
                    label_after=score_to_label(after),
                    passed=passed,
                )
            )
        except Exception as error:
            passes_all = False
            results.append(
                DetectorVerifyResult(
                    name=name,
                    score_before=0.0,
                    score_after=0.0,
                    label_before="error",
                    label_after="error",
                    passed=False,
                    error=str(error),
                )
            )
    return VerificationReport(threshold=threshold, results=results, passes_all=passes_all)
