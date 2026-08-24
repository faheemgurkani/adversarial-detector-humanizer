"""Commercial detector adapters (Pangram, GPTZero).

These are intended for ``adh score`` and post-loop verification. They must not
be used as the inner-loop detector in ``humanize()`` — each API call is billed
and latency is too high for per-sentence rewrites.
"""

from __future__ import annotations

import os
import time
from typing import Any, Literal

import httpx

from adh.detectors.base import ScoreResult, Window, probability_to_result, require_text
from adh.exceptions import InputError, RemoteDetectorError, RemoteDetectorUnavailableError
from adh.report import score_to_label

INNER_LOOP_BLOCKED = frozenset({"pangram", "gptzero"})

PANGRAM_API_BASE = "https://text.external-api.pangram.com"
GPTZERO_API_BASE = "https://api.gptzero.me/v2"


def assert_inner_loop_detector(name: str) -> None:
    if name in INNER_LOOP_BLOCKED:
        raise RemoteDetectorUnavailableError(
            f"{name} cannot drive the humanize inner loop. Run humanize with a "
            "local detector, then verify with "
            f"`adh score --detector {name}`."
        )


def _require_api_key(name: str, value: str | None) -> str:
    key = (value or os.environ.get(f"{name.upper()}_API_KEY") or "").strip()
    if not key:
        raise InputError(
            f"{name.upper()}_API_KEY is not set. Add it to `.env` or export it "
            f"before using --detector {name.lower()}."
        )
    return key


def _http_error(provider: str, response: httpx.Response) -> RemoteDetectorError:
    detail = response.text[:400].strip() or response.reason_phrase
    return RemoteDetectorError(f"{provider} HTTP {response.status_code}: {detail}")


def _pangram_document_probability(body: dict[str, Any]) -> float:
    fraction_ai = float(body.get("fraction_ai") or 0.0)
    fraction_ai_assisted = float(body.get("fraction_ai_assisted") or 0.0)
    fraction_human = float(body.get("fraction_human") or 0.0)
    combined = fraction_ai + fraction_ai_assisted
    if combined > 0.0:
        return min(1.0, max(0.0, combined))
    if fraction_human > 0.0:
        return min(1.0, max(0.0, 1.0 - fraction_human))
    windows = body.get("windows") or []
    if windows:
        scores = [float(window.get("ai_assistance_score") or 0.0) for window in windows]
        return min(1.0, max(0.0, sum(scores) / len(scores)))
    return 0.0


def _pangram_windows(body: dict[str, Any]) -> list[Window]:
    windows: list[Window] = []
    for window in body.get("windows") or []:
        probability = float(window.get("ai_assistance_score") or 0.0)
        score = round(probability * 100.0, 4)
        label = score_to_label(score)
        if window.get("is_humanized"):
            label = f"{label};humanized"
        windows.append(
            Window(
                text=str(window.get("text") or ""),
                score=score,
                label=label,
                start=window.get("start_index"),
                end=window.get("end_index"),
            )
        )
    return windows


def _gptzero_document(body: dict[str, Any]) -> tuple[float, list[Window]]:
    documents = body.get("documents") or []
    if not documents:
        raise RemoteDetectorError("gptzero returned no documents")
    document = documents[0]
    probability = document.get("completely_generated_prob")
    if probability is None:
        probability = document.get("average_generated_prob")
    if probability is None:
        raise RemoteDetectorError("gptzero response missing generated probability")
    probability = float(probability)
    windows: list[Window] = []
    for sentence in document.get("sentences") or []:
        sentence_prob = float(sentence.get("generated_prob") or 0.0)
        score = round(sentence_prob * 100.0, 4)
        windows.append(
            Window(
                text=str(sentence.get("sentence") or ""),
                score=score,
                label=score_to_label(score),
            )
        )
    return probability, windows


class PangramDetector:
    """Pangram async inference API (pangram-4 by default)."""

    name = "pangram"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "pangram-4",
        timeout: float = 300.0,
        poll_interval: float = 0.5,
        base_url: str = PANGRAM_API_BASE,
    ) -> None:
        self.api_key = _require_api_key("PANGRAM", api_key)
        self.model = model
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.base_url = base_url.rstrip("/")

    def _predict(self, text: str) -> dict[str, Any]:
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "text": text,
            "model": self.model,
            "public_dashboard_link": False,
        }
        try:
            create = httpx.post(
                f"{self.base_url}/task",
                headers=headers,
                json=payload,
                timeout=min(60.0, self.timeout),
            )
        except httpx.HTTPError as error:
            raise RemoteDetectorError(f"pangram request failed: {error}") from error
        if create.status_code >= 400:
            raise _http_error("pangram", create)
        try:
            task_id = create.json()["task_id"]
        except (KeyError, TypeError, ValueError) as error:
            raise RemoteDetectorError("pangram create-task returned no task_id") from error

        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                poll = httpx.get(
                    f"{self.base_url}/task/{task_id}",
                    headers={"x-api-key": self.api_key},
                    timeout=min(60.0, self.timeout),
                )
            except httpx.HTTPError as error:
                raise RemoteDetectorError(f"pangram poll failed: {error}") from error
            if poll.status_code >= 400:
                raise _http_error("pangram", poll)
            body = poll.json()
            stage = body.get("stage")
            if stage == "STAGE_SUCCESS":
                return body
            if stage == "STAGE_FAILED":
                headline = body.get("headline") or "task failed"
                raise RemoteDetectorError(f"pangram task failed: {headline}")
            time.sleep(self.poll_interval)
        raise RemoteDetectorError("pangram task timed out")

    def score(self, text: str) -> ScoreResult:
        require_text(text)
        body = self._predict(text)
        probability = _pangram_document_probability(body)
        return ScoreResult(
            score=round(probability * 100.0, 4),
            label=score_to_label(probability * 100.0),
            windows=_pangram_windows(body),
        )

    def score_spans(self, texts: list[str]) -> list[ScoreResult]:
        if not texts:
            return []
        return [self.score(text) for text in texts]


class GPTZeroDetector:
    """GPTZero v2 text prediction API."""

    name = "gptzero"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 60.0,
        base_url: str = GPTZERO_API_BASE,
    ) -> None:
        self.api_key = _require_api_key("GPTZERO", api_key)
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")

    def _predict(self, text: str) -> dict[str, Any]:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            response = httpx.post(
                f"{self.base_url}/predict/text",
                headers=headers,
                json={"document": text},
                timeout=self.timeout,
            )
        except httpx.HTTPError as error:
            raise RemoteDetectorError(f"gptzero request failed: {error}") from error
        if response.status_code >= 400:
            raise _http_error("gptzero", response)
        try:
            return response.json()
        except ValueError as error:
            raise RemoteDetectorError("gptzero returned non-JSON") from error

    def score(self, text: str) -> ScoreResult:
        require_text(text)
        probability, windows = _gptzero_document(self._predict(text))
        return probability_to_result(probability, windows=windows)

    def score_spans(self, texts: list[str]) -> list[ScoreResult]:
        if not texts:
            return []
        return [self.score(text) for text in texts]


class EnsembleDetector:
    """Combine scores from multiple detectors."""

    name = "ensemble"

    def __init__(
        self,
        detectors: list[Any],
        *,
        weights: list[float] | None = None,
        aggregate: Literal["mean", "max"] = "max",
    ) -> None:
        if not detectors:
            raise ValueError("ensemble requires at least one detector")
        if weights is not None and len(weights) != len(detectors):
            raise ValueError("weights must match the number of detectors")
        if weights is not None and any(weight < 0 for weight in weights):
            raise ValueError("weights must be non-negative")
        self.detectors = list(detectors)
        self.aggregate = aggregate
        if weights is None:
            self.weights = [1.0] * len(detectors)
        else:
            total = sum(weights)
            if total <= 0:
                raise ValueError("weights must sum to a positive value")
            self.weights = [weight / total for weight in weights]

    def _aggregate(self, scores: list[float]) -> float:
        if self.aggregate == "max":
            return round(max(scores), 4)
        blended = sum(score * weight for score, weight in zip(scores, self.weights, strict=True))
        return round(blended, 4)

    def score(self, text: str) -> ScoreResult:
        require_text(text)
        scores = [detector.score(text).score for detector in self.detectors]
        score = self._aggregate(scores)
        return ScoreResult(score=score, label=score_to_label(score))

    def score_spans(self, texts: list[str]) -> list[ScoreResult]:
        if not texts:
            return []
        matrices = [detector.score_spans(texts) for detector in self.detectors]
        results: list[ScoreResult] = []
        for index in range(len(texts)):
            scores = [row[index].score for row in matrices]
            score = self._aggregate(scores)
            results.append(ScoreResult(score=score, label=score_to_label(score)))
        return results
