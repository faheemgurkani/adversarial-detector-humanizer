"""Certainty and polarity retention checks."""

from __future__ import annotations

import re

_CLASSES: dict[str, tuple[str, ...]] = {
    "modality": (
        "may",
        "might",
        "could",
        "can",
        "would",
        "possibly",
        "perhaps",
        "potentially",
        "likely",
        "unlikely",
        "probably",
        "maybe",
        "if",
        "unless",
        "assuming",
    ),
    "evidential": (
        "suggests",
        "suggest",
        "indicates",
        "indicate",
        "appears",
        "appear",
        "seems",
        "seem",
        "reportedly",
        "allegedly",
        "alleged",
        "allege",
        "accuse",
        "accuses",
        "accused",
        "claims",
        "claim",
        "claimed",
        "according to",
        "evidence",
    ),
    "frequency": (
        "usually",
        "often",
        "sometimes",
        "typically",
        "generally",
        "frequently",
        "occasionally",
        "rarely",
        "mostly",
        "tends",
        "tend",
    ),
    "quantifier": (
        "some",
        "several",
        "many",
        "most",
        "a few",
        "few",
        "certain",
        "various",
        "a number of",
        "not all",
    ),
    "degree": (
        "slightly",
        "marginally",
        "somewhat",
        "a bit",
        "moderately",
        "partially",
        "partly",
        "slight",
        "small",
        "minor",
        "modest",
        "moderate",
        "minimal",
        "partial",
    ),
    "intention": (
        "plans",
        "plan",
        "planned",
        "aims",
        "aim",
        "intends",
        "intend",
        "expects",
        "expect",
        "hopes",
        "hope",
        "proposes",
        "proposed",
    ),
}

_CLASS_RES: dict[str, re.Pattern[str]] = {
    name: re.compile(
        r"(?<!\w)(?:"
        + "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True))
        + r")(?!\w)",
        re.IGNORECASE,
    )
    for name, terms in _CLASSES.items()
}

_ASSOCIATION_RE = re.compile(
    r"(?<!\w)(?:correlat\w*|associat\w*|linked|link\s+between|related\s+to|coincid\w*|"
    r"tied\s+to|goes?\s+together|tracks?\s+with)(?!\w)",
    re.IGNORECASE,
)
_CAUSAL_RE = re.compile(
    r"(?<!\w)(?:caus\w*|leads?\s+to|led\s+to|results?\s+in|resulted\s+in|produc\w*|"
    r"triggers?|triggered|drives?|responsible\s+for|because\s+of|owing\s+to)(?!\w)",
    re.IGNORECASE,
)
_NEGATOR_RE = re.compile(
    r"(?<!\w)(?:not|no|nothing|never|neither|nor|n't|cannot|can't|isn't|is\s+not|without)(?!\w)",
    re.IGNORECASE,
)
_SENT_START_RE = re.compile(r"(?:^|[.!?]\s+)")
_INTENSIFIER_RE = re.compile(
    r"(?<!\w)(?:large|huge|massive|dramatic|dramatically|sharp|sharply|significant|"
    r"significantly|substantial|substantially|major|severe|drastic|drastically|"
    r"considerable|considerably|collapsed|plummeted|surged)(?!\w)",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"n't|\bnot\b(?!\s+only\b)|\b(?:never|no|none|neither|nor|cannot|without|unable to)\b",
    re.IGNORECASE,
)


def _classes_present(text: str) -> set[str]:
    return {name for name, pattern in _CLASS_RES.items() if pattern.search(text)}


def _asserts_causation(text: str) -> bool:
    for match in _CAUSAL_RE.finditer(text):
        start = 0
        for boundary in _SENT_START_RE.finditer(text, 0, match.start()):
            start = boundary.end()
        if not _NEGATOR_RE.search(text[start : match.start()]):
            return True
    return False


def _causal_upgrade(source: str, candidate: str) -> bool:
    if not _ASSOCIATION_RE.search(source) or _asserts_causation(source):
        return False
    return _asserts_causation(candidate)


def _intensifier_added(source: str, candidate: str) -> bool:
    return bool(_INTENSIFIER_RE.search(candidate)) and not _INTENSIFIER_RE.search(source)


def dropped_hedges(source: str, candidate: str) -> list[str]:
    found = sorted(_classes_present(source) - _classes_present(candidate))
    if _causal_upgrade(source, candidate):
        found.append("causal_upgrade")
    if _intensifier_added(source, candidate):
        found.append("intensifier_added")
    return found


def certainty_kept(source: str, candidate: str) -> bool:
    return not dropped_hedges(source, candidate)


def negation_count(text: str) -> int:
    return len(_NEGATION_RE.findall(text))


def polarity_kept(source: str, candidate: str) -> bool:
    return negation_count(source) == negation_count(candidate)
