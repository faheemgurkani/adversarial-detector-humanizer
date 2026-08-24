"""Optional NLI contradiction and entailment checks."""

from __future__ import annotations

import os
from functools import lru_cache

from adh.gates.text_split import aligned_chunks

_MODEL_ID = "cross-encoder/nli-distilroberta-base"
DEFAULT_CONTRADICTION_BAR = 0.5
DEFAULT_ENTAILMENT_FLOOR = 0.005


class _NLI:
    tok = None
    model = None
    label_idx: dict[str, int] | None = None
    dead = False


def nli_available() -> bool:
    if _NLI.dead:
        return False
    if os.environ.get("ADH_DISABLE_NLI") == "1":
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def _load():
    if _NLI.model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(_MODEL_ID).eval()
        _NLI.label_idx = {str(value).lower(): int(key) for key, value in model.config.id2label.items()}
        _NLI.tok, _NLI.model = tok, model
    return _NLI.tok, _NLI.model


@lru_cache(maxsize=16)
def _pair_probs(premise: str, hypothesis: str) -> tuple[float, float, float]:
    import torch

    tok, model = _load()
    enc = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]
    labels = _NLI.label_idx or {}
    entail = float(probs[labels.get("entailment", 0)])
    neutral = float(probs[labels.get("neutral", 1)])
    contra = float(probs[labels.get("contradiction", 2)])
    return entail, neutral, contra


def _chunk_scores(left: str, right: str) -> tuple[float, float]:
    max_contra = 0.0
    min_entail = 1.0
    for chunk_left, chunk_right in aligned_chunks(left, right):
        if chunk_left.strip() == chunk_right.strip():
            continue
        for premise, hypothesis in ((chunk_left, chunk_right), (chunk_right, chunk_left)):
            entail, _neutral, contra = _pair_probs(premise, hypothesis)
            max_contra = max(max_contra, contra)
            min_entail = min(min_entail, entail)
    if min_entail == 1.0:
        return 0.0, 1.0
    return max_contra, min_entail


def contradiction_score(left: str, right: str) -> float | None:
    if not nli_available() or not left.strip() or not right.strip():
        return None
    try:
        contra, _entail = _chunk_scores(left, right)
        return contra
    except Exception:
        _NLI.dead = True
        return None


def entailment_score(left: str, right: str) -> float | None:
    if not nli_available() or not left.strip() or not right.strip():
        return None
    try:
        _contra, entail = _chunk_scores(left, right)
        return entail
    except Exception:
        _NLI.dead = True
        return None


def nli_passes(
    left: str,
    right: str,
    *,
    contradiction_bar: float = DEFAULT_CONTRADICTION_BAR,
    entailment_floor: float = DEFAULT_ENTAILMENT_FLOOR,
) -> bool | None:
    contra = contradiction_score(left, right)
    entail = entailment_score(left, right)
    if contra is None or entail is None:
        return None
    return contra < contradiction_bar and entail >= entailment_floor
