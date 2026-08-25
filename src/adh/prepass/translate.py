"""Pluggable translation backends for structural pre-pass."""

from __future__ import annotations

import json
import os
from typing import Protocol, runtime_checkable

import httpx

from adh.exceptions import RewriterError

LANG_NAMES = {
    "fi": "Finnish",
    "zh": "Chinese",
    "ja": "Japanese",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
}


@runtime_checkable
class Translator(Protocol):
    name: str

    def translate(self, text: str, *, source: str, target: str) -> str:
        """Translate ``text`` from ``source`` ISO code to ``target`` ISO code."""


class IdentityTranslator:
    """Test helper that returns input unchanged."""

    name = "identity"

    def translate(self, text: str, *, source: str, target: str) -> str:
        return text


class GoogleTranslator:
    """Free Google Translate via deep-translator (optional extra)."""

    name = "google"

    def translate(self, text: str, *, source: str, target: str) -> str:
        try:
            from deep_translator import GoogleTranslator as _GoogleTranslator
        except ImportError as error:
            raise RewriterError(
                "Google translation requires deep-translator. "
                "Install: pip install 'adversarial-detector-humanizer[prepass]'"
            ) from error
        try:
            return _GoogleTranslator(source=source, target=target).translate(text)
        except Exception as error:
            raise RewriterError(f"google translation failed: {error}") from error


class LLMTranslator:
    """OpenAI-compatible translation hop preserving __LOCK_*__ sentinels."""

    name = "llm"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("ADH_REWRITER_MODEL") or "gpt-4o-mini"
        self.timeout = timeout

    def translate(self, text: str, *, source: str, target: str) -> str:
        target_name = LANG_NAMES.get(target, target)
        source_name = LANG_NAMES.get(source, source)
        prompt = (
            f"Translate the following text from {source_name} to {target_name}. "
            "Preserve every __LOCK_...__ token exactly. Output only the translation.\n\n"
            f"{text}"
        )
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "temperature": 0.3,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise translator. Never alter lock sentinels.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=self.timeout)
        except httpx.HTTPError as error:
            raise RewriterError(f"llm translation request failed: {error}") from error
        if response.status_code >= 400:
            raise RewriterError(
                f"llm translation HTTP {response.status_code}: {response.text[:400]}"
            )
        try:
            body = response.json()
        except json.JSONDecodeError as error:
            raise RewriterError("llm translation returned non-JSON") from error
        choices = body.get("choices") or []
        if not choices:
            raise RewriterError("llm translation returned no choices")
        content = (choices[0].get("message") or {}).get("content") or ""
        translated = content.strip()
        if not translated:
            raise RewriterError("llm translation returned empty text")
        return translated


def round_trip_translate(
    text: str,
    *,
    lang: str,
    translator: Translator,
    source: str = "en",
) -> str:
    outbound = translator.translate(text, source=source, target=lang)
    return translator.translate(outbound, source=lang, target=source)


def load_translator(name: str = "llm") -> Translator:
    if name == "google":
        return GoogleTranslator()
    if name == "llm":
        return LLMTranslator()
    if name == "identity":
        return IdentityTranslator()
    raise RewriterError(f"unknown translation backend {name!r}")
