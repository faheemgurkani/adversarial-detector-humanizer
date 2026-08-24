"""Optional predicate-argument role swap detection."""

from __future__ import annotations

import os
from functools import lru_cache

_SUBJ = {"nsubj", "csubj", "expl", "nsubjpass", "nsubj:pass", "csubjpass"}
_OBJ = {"dobj", "obj", "attr", "oprd", "dative", "xcomp", "ccomp"}


class _NLP:
    pipe = None
    dead = False


def roles_available() -> bool:
    if _NLP.dead:
        return False
    if os.environ.get("ADH_DISABLE_ROLES") == "1":
        return False
    try:
        import spacy  # noqa: F401
    except Exception:
        return False
    return True


def _load():
    if _NLP.pipe is None:
        import spacy

        try:
            _NLP.pipe = spacy.load("en_core_web_sm")
        except OSError:
            _NLP.dead = True
            raise
    return _NLP.pipe


def _triples(text: str) -> set[tuple[str, str, str]]:
    nlp = _load()
    triples: set[tuple[str, str, str]] = set()
    doc = nlp(text)
    for sent in doc.sents:
        for token in sent:
            if token.pos_ != "VERB":
                continue
            subjects = [child.lemma_.lower() for child in token.children if child.dep_ in _SUBJ]
            objects = [child.lemma_.lower() for child in token.children if child.dep_ in _OBJ]
            if not subjects or not objects:
                continue
            for subject in subjects:
                for obj in objects:
                    triples.add((subject, token.lemma_.lower(), obj))
    return triples


@lru_cache(maxsize=32)
def _cached_triples(text: str) -> frozenset[tuple[str, str, str]]:
    return frozenset(_triples(text))


def role_swap(source: str, candidate: str) -> bool | None:
    if not roles_available():
        return None
    try:
        source_triples = _cached_triples(source.strip())
        candidate_triples = _cached_triples(candidate.strip())
    except Exception:
        _NLP.dead = True
        return None
    if not source_triples or not candidate_triples:
        return False
    swapped = {
        (obj, verb, subj)
        for subj, verb, obj in source_triples
        if (obj, verb, subj) in candidate_triples and (subj, verb, obj) not in candidate_triples
    }
    return bool(swapped)
