"""Strip reader-directed scaffolding before meaning checks."""

from __future__ import annotations

import re

_META_CLOSER_RE = re.compile(
    r"(?<!\w)(?:in conclusion|to summarize|overall|in summary|let me know if you need"
    r"|hope this helps|feel free to ask)(?!\w)[^.!?]*[.!?]?",
    re.IGNORECASE,
)
_STANCE_FRAME_RE = re.compile(
    r"(?<!\w)(?:it is important to note that|it should be noted that|"
    r"it is worth noting that)(?!\w)\s*",
    re.IGNORECASE,
)


def strip_scaffolding(text: str) -> str:
    stripped = _META_CLOSER_RE.sub(" ", text)
    stripped = _STANCE_FRAME_RE.sub(" ", stripped)
    return " ".join(stripped.split())
