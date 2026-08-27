"""Named setting bundles shared by CLI, HTTP, and adh.yaml."""

from __future__ import annotations

from typing import Any

from adh.exceptions import InputError
from adh.models import DEFAULT_MODEL

TRY_SAMPLE_TEXT = "Furthermore, note this."

FAST_PROFILE: dict[str, Any] = {
    "detector": "fake",
    "rewriter": "identity",
    "max_rounds": 1,
    "allow_lexical_gate": True,
    "semantic": "lexical",
    "meaning_gate_mode": "lexical",
}

STANDARD_PROFILE: dict[str, Any] = {
    "detector": DEFAULT_MODEL,
    "rewriter": "openai",
    "max_rounds": 5,
}

QUALITY_PROFILE: dict[str, Any] = {
    "detector": "ensemble-local",
    "rewriter": "openai",
    "max_rounds": 5,
    "allow_lexical_gate": True,
    "semantic": "lexical",
}

VERIFY_ONLY_PROFILE: dict[str, Any] = {
    "detector": DEFAULT_MODEL,
    "rewriter": "identity",
    "max_rounds": 1,
}

PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "fast": FAST_PROFILE,
    "standard": STANDARD_PROFILE,
    "quality": QUALITY_PROFILE,
    "verify-only": VERIFY_ONLY_PROFILE,
}


def get_profile_preset(name: str) -> dict[str, Any]:
    """Return a copy of a named profile preset."""
    if name not in PROFILE_PRESETS:
        available = ", ".join(sorted(PROFILE_PRESETS))
        raise InputError(f"unknown profile {name!r}. Available: {available}")
    return dict(PROFILE_PRESETS[name])


def apply_profile(name: str) -> dict[str, Any]:
    """Apply a profile preset and include the profile name."""
    preset = get_profile_preset(name)
    preset["profile"] = name
    return preset
