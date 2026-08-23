"""Semantic similarity gates used to reject meaning-drifting rewrites."""

from __future__ import annotations

import math
import re
from typing import Protocol, runtime_checkable

from adh.exceptions import InputError, SemanticBackendError

_TOKEN = re.compile(r"[A-Za-z0-9']+")
_MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"
_minilm_model = None


@runtime_checkable
class SemanticGate(Protocol):
    name: str

    def similarity(self, left: str, right: str) -> float:
        """Return cosine-like similarity in [0, 1]."""


def _require_pair(left: str, right: str) -> tuple[str, str]:
    if not isinstance(left, str) or not isinstance(right, str):
        raise TypeError("both texts must be strings")
    if not left.strip() or not right.strip():
        raise InputError("cannot score similarity of empty text")
    return left, right


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


class LexicalSemanticGate:
    """Token-overlap cosine. Used in tests and when MiniLM is unavailable."""

    name = "lexical"

    def similarity(self, left: str, right: str) -> float:
        left, right = _require_pair(left, right)
        left_tokens = _tokenize(left)
        right_tokens = _tokenize(right)
        if not left_tokens or not right_tokens:
            return 0.0
        vocab = sorted(set(left_tokens) | set(right_tokens))
        left_vec = [left_tokens.count(token) for token in vocab]
        right_vec = [right_tokens.count(token) for token in vocab]
        dot = sum(a * b for a, b in zip(left_vec, right_vec, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left_vec))
        right_norm = math.sqrt(sum(value * value for value in right_vec))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (left_norm * right_norm)))


class MiniLMSemanticGate:
    """Cosine similarity via sentence-transformers/all-MiniLM-L6-v2."""

    name = "minilm"

    def __init__(self, model_id: str = _MINILM_ID) -> None:
        self.model_id = model_id
        self._model = None

    def _load(self):
        global _minilm_model
        if self._model is not None:
            return self._model
        if _minilm_model is not None and self.model_id == _MINILM_ID:
            self._model = _minilm_model
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise SemanticBackendError(
                "sentence-transformers is required for the MiniLM gate. "
                "Install extras: pip install 'adversarial-detector-humanizer[local]'"
            ) from error
        self._model = SentenceTransformer(self.model_id)
        if self.model_id == _MINILM_ID:
            _minilm_model = self._model
        return self._model

    def similarity(self, left: str, right: str) -> float:
        left, right = _require_pair(left, right)
        model = self._load()
        embeddings = model.encode([left, right])
        scores = model.similarity(embeddings[:1], embeddings[1:])
        value = float(scores[0][0])
        return max(0.0, min(1.0, value))


def build_semantic_gate(*, prefer: str = "auto", allow_lexical: bool = False) -> SemanticGate:
    """Resolve a gate. ``auto`` tries MiniLM and falls back only when allowed."""
    if prefer == "lexical":
        return LexicalSemanticGate()
    if prefer == "minilm":
        return MiniLMSemanticGate()
    if prefer != "auto":
        raise InputError("semantic gate must be auto, minilm, or lexical")
    try:
        return MiniLMSemanticGate()
    except SemanticBackendError:
        if allow_lexical:
            return LexicalSemanticGate()
        raise


def passes_gate(left: str, right: str, gate: SemanticGate, threshold: float) -> tuple[bool, float]:
    if not 0.0 <= threshold <= 1.0:
        raise InputError("semantic threshold must be between 0 and 1")
    score = gate.similarity(left, right)
    return score >= threshold, score
