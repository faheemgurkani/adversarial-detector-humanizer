"""Agent-facing guidance derived from engine stop reasons."""

from __future__ import annotations

from adh.report import RunReport, StopReason

AGENT_HINTS: dict[StopReason, str] = {
    "passed": (
        "Local score reached target. Run verify if the user asked about "
        "Pangram or GPTZero."
    ),
    "max_rounds": (
        "Best effort returned after max rounds. Consider manual edits or "
        "raising max_rounds."
    ),
    "no_flagged_sentences": (
        "No sentences exceeded the threshold; output is unchanged."
    ),
    "all_candidates_rejected": (
        "Rewrites failed semantic or detector gates. Review flagged sentences "
        "manually."
    ),
    "max_rewrite_ratio": (
        "Stopped to preserve document length. Tighten targeting or edit manually."
    ),
    "already_below_target": (
        "Input already below target score; no rewrite was needed."
    ),
}


def agent_hint_for(report: RunReport) -> str:
    return AGENT_HINTS.get(report.stop_reason, "Review the run report for next steps.")
