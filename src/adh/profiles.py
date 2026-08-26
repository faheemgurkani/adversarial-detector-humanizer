"""Named setting bundles. Full yaml profiles arrive in a later step."""

from __future__ import annotations

from typing import Any

from adh.exceptions import InputError

TRY_SAMPLE_TEXT = "Furthermore, note this."

FAST_PROFILE: dict[str, Any] = {
    "detector": "fake",
    "rewriter": "identity",
    "max_rounds": 1,
    "allow_lexical_gate": True,
    "semantic": "lexical",
    "meaning_gate_mode": "lexical",
}

PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "fast": FAST_PROFILE,
}


def get_profile_preset(name: str) -> dict[str, Any]:
    """Return a copy of a named profile preset."""
    if name not in PROFILE_PRESETS:
        available = ", ".join(sorted(PROFILE_PRESETS))
        raise InputError(f"unknown profile {name!r}. Available: {available}")
    return dict(PROFILE_PRESETS[name])
