"""OpenAI-compatible register-shift rewriter. No regex humanizer fallback."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx

from adh.exceptions import InputError, RewriterError
from adh.ranking import mean_token_logprob

REGISTER_SHIFT_SYSTEM_PROMPT = """You are a register-shift editor, not a synonym swapper.

Rewrite the user's sentence so it keeps the exact meaning while sounding less
template-like. Follow every rule:

- Preserve facts, citations, names, numbers, and any __LOCK_...__ sentinels exactly.
- Vary sentence rhythm. Mix short and longer clauses. Avoid metronomic cadence.
- Drop stock AI transitions: furthermore, moreover, in conclusion, it is important
  to note, in today's fast-paced world.
- Prefer concrete verbs over hedging templates.
- Do not add em-dash spam or "not only X but also Y" constructions.
- Do not add new claims, examples, or opinions.
- Output only the rewritten sentence. No quotes, no preface, no bullet list.
- Return exactly one sentence unless the input already contains more than one.
"""


@dataclass(frozen=True)
class RewriteCandidate:
    text: str
    mean_logprob: float | None = None


@runtime_checkable
class Rewriter(Protocol):
    name: str

    def rewrite(self, sentence: str, *, n: int = 1) -> list[str]:
        """Return ``n`` candidate rewrites of ``sentence``."""

    def rewrite_candidates(self, sentence: str, *, n: int = 1) -> list[RewriteCandidate]:
        """Return candidates with optional mean token logprob when available."""


class IdentityRewriter:
    """Test helper that returns the input unchanged."""

    name = "identity"

    def rewrite(self, sentence: str, *, n: int = 1) -> list[str]:
        return [candidate.text for candidate in self.rewrite_candidates(sentence, n=n)]

    def rewrite_candidates(self, sentence: str, *, n: int = 1) -> list[RewriteCandidate]:
        if not sentence.strip():
            raise InputError("cannot rewrite empty sentence")
        if n < 1:
            raise InputError("n must be at least 1")
        return [RewriteCandidate(text=sentence, mean_logprob=-0.05)] * n


class ScriptedRewriter:
    """Map original sentences to scripted candidates for unit tests."""

    name = "scripted"

    def __init__(
        self,
        mapping: dict[str, list[str]],
        *,
        logprobs: dict[str, list[float | None]] | None = None,
    ) -> None:
        self.mapping = {key.strip(): list(values) for key, values in mapping.items()}
        self.logprobs = logprobs or {}

    def rewrite(self, sentence: str, *, n: int = 1) -> list[str]:
        return [candidate.text for candidate in self.rewrite_candidates(sentence, n=n)]

    def rewrite_candidates(self, sentence: str, *, n: int = 1) -> list[RewriteCandidate]:
        if n < 1:
            raise InputError("n must be at least 1")
        key = sentence.strip()
        if key not in self.mapping:
            raise RewriterError(f"no scripted rewrite for: {sentence!r}")
        candidates = self.mapping[key]
        if not candidates:
            raise RewriterError("scripted rewriter returned no candidates")
        probs = self.logprobs.get(key, [None] * len(candidates))
        selected = (candidates * n)[:n]
        selected_probs = (probs * n)[:n]
        return [
            RewriteCandidate(text=text, mean_logprob=prob)
            for text, prob in zip(selected, selected_probs, strict=False)
        ]


class OpenAICompatibleRewriter:
    """Chat-completions client compatible with OpenAI, Groq, OpenRouter, Ollama."""

    name = "openai-compatible"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        request_logprobs: bool = True,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("ADH_REWRITER_MODEL") or "gpt-4o-mini"
        self.timeout = timeout
        self.request_logprobs = request_logprobs
        if not self.api_key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            raise RewriterError(
                "OPENAI_API_KEY is not set. The engine will not fall back to "
                "regex phrase-swapping. Point OPENAI_BASE_URL at a local "
                "OpenAI-compatible server if you do not want a cloud key."
            )

    def rewrite(self, sentence: str, *, n: int = 1) -> list[str]:
        return [candidate.text for candidate in self.rewrite_candidates(sentence, n=n)]

    def rewrite_candidates(self, sentence: str, *, n: int = 1) -> list[RewriteCandidate]:
        if not sentence.strip():
            raise InputError("cannot rewrite empty sentence")
        if n < 1:
            raise InputError("n must be at least 1")

        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "temperature": 0.8,
            "n": n,
            "messages": [
                {"role": "system", "content": REGISTER_SHIFT_SYSTEM_PROMPT},
                {"role": "user", "content": sentence},
            ],
        }
        if self.request_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 0
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=self.timeout)
        except httpx.HTTPError as error:
            raise RewriterError(f"rewriter request failed: {error}") from error
        if response.status_code >= 400:
            raise RewriterError(
                f"rewriter HTTP {response.status_code}: {response.text[:400]}"
            )
        try:
            body = response.json()
        except json.JSONDecodeError as error:
            raise RewriterError("rewriter returned non-JSON") from error
        choices = body.get("choices") or []
        candidates: list[RewriteCandidate] = []
        for choice in choices:
            message = choice.get("message") or {}
            content = (message.get("content") or "").strip()
            if content:
                candidates.append(
                    RewriteCandidate(
                        text=_strip_wrapping_quotes(content),
                        mean_logprob=mean_token_logprob(choice),
                    )
                )
        if not candidates:
            raise RewriterError("rewriter returned no usable candidates")
        return candidates[:n]


def _strip_wrapping_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text
