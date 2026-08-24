"""Mechanical AI-tells counter for best-of-N tie-breaking."""

from __future__ import annotations

import re
from collections import Counter

_FORMULAIC_TRANSITION = re.compile(
    r"(?<!\w)(?:furthermore|moreover|in addition|additionally|in conclusion|"
    r"to summarize|overall|it is important to note|it should be noted)(?!\w)",
    re.IGNORECASE,
)
_EM_DASH = re.compile(r"—|(?:\s-\s){2,}")
_AI_VOCAB = re.compile(
    r"(?<!\w)(?:leverage|delve|robust|seamless|tapestry|landscape|utilize|"
    r"facilitate|comprehensive|multifaceted|myriad|plethora)(?!\w)",
    re.IGNORECASE,
)
_META_CLOSER = re.compile(
    r"(?<!\w)(?:let me know if you need|hope this helps|feel free to ask)(?!\w)",
    re.IGNORECASE,
)


def score_tells(text: str) -> dict[str, int | dict[str, int]]:
    categories = {
        "formulaic_transition": len(_FORMULAIC_TRANSITION.findall(text)),
        "em_dash": len(_EM_DASH.findall(text)),
        "ai_vocab": len(_AI_VOCAB.findall(text)),
        "meta_closer": len(_META_CLOSER.findall(text)),
    }
    total = sum(categories.values())
    return {
        "tells": total,
        "by_category": categories,
    }
