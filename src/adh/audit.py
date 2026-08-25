"""Post-loop detector breakdown for guidance vs deploy transfer reporting."""

from __future__ import annotations

from adh.detectors.base import Detector
from adh.report import DetectorBreakdown, DetectorBreakdownEntry, score_to_label


def _entry(
    name: str,
    role: str,
    *,
    before: float,
    after: float,
) -> DetectorBreakdownEntry:
    return DetectorBreakdownEntry(
        name=name,
        role=role,  # type: ignore[arg-type]
        score_before=before,
        score_after=after,
        label_before=score_to_label(before),
        label_after=score_to_label(after),
        delta=round(after - before, 4),
    )


def build_detector_breakdown(
    input_text: str,
    output_text: str,
    *,
    guidance: Detector,
    guidance_before: float | None = None,
    guidance_after: float | None = None,
    deploy: list[Detector],
) -> DetectorBreakdown:
    before_guidance = (
        guidance_before
        if guidance_before is not None
        else guidance.score(input_text).score
    )
    after_guidance = (
        guidance_after
        if guidance_after is not None
        else guidance.score(output_text).score
    )
    entries = [
        _entry(guidance.name, "guidance", before=before_guidance, after=after_guidance)
    ]
    guidance_dropped = after_guidance < before_guidance
    transfer_ok = guidance_dropped if deploy else None

    for detector in deploy:
        before = detector.score(input_text).score
        after = detector.score(output_text).score
        entries.append(_entry(detector.name, "deploy", before=before, after=after))
        if transfer_ok is not None:
            if not (after < before and guidance_dropped):
                transfer_ok = False

    return DetectorBreakdown(
        guidance=guidance.name,
        entries=entries,
        transfer_ok=transfer_ok,
    )
