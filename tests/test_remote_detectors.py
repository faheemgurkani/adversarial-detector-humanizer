from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from adh.detectors.remote import (
    GPTZeroDetector,
    PangramDetector,
    assert_inner_loop_detector,
)
from adh.exceptions import InputError, RemoteDetectorError, RemoteDetectorUnavailableError
from adh.factory import load_detector


def test_pangram_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("PANGRAM_API_KEY", raising=False)
    with pytest.raises(InputError, match="PANGRAM_API_KEY"):
        PangramDetector()


def test_gptzero_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GPTZERO_API_KEY", raising=False)
    with pytest.raises(InputError, match="GPTZERO_API_KEY"):
        GPTZeroDetector()


def test_inner_loop_blocked() -> None:
    with pytest.raises(RemoteDetectorUnavailableError):
        assert_inner_loop_detector("pangram")
    with pytest.raises(RemoteDetectorUnavailableError):
        assert_inner_loop_detector("gptzero")


def test_pangram_score(monkeypatch) -> None:
    detector = PangramDetector(api_key="test-key", poll_interval=0.0, timeout=5.0)

    create = MagicMock()
    create.status_code = 200
    create.json.return_value = {"task_id": "task-1"}

    success = MagicMock()
    success.status_code = 200
    success.json.return_value = {
        "stage": "STAGE_SUCCESS",
        "fraction_ai": 0.2,
        "fraction_ai_assisted": 0.3,
        "fraction_human": 0.5,
        "windows": [
            {
                "text": "Flagged passage.",
                "ai_assistance_score": 0.85,
                "start_index": 0,
                "end_index": 16,
                "is_humanized": True,
            }
        ],
    }

    def fake_post(url: str, **kwargs) -> MagicMock:
        assert url.endswith("/task")
        return create

    def fake_get(url: str, **kwargs) -> MagicMock:
        assert url.endswith("/task/task-1")
        return success

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)

    result = detector.score("Flagged passage.")
    assert result.score == pytest.approx(50.0)
    assert len(result.windows) == 1
    assert result.windows[0].score == pytest.approx(85.0)
    assert "humanized" in result.windows[0].label


def test_gptzero_score(monkeypatch) -> None:
    detector = GPTZeroDetector(api_key="test-key")

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "documents": [
            {
                "completely_generated_prob": 0.91,
                "average_generated_prob": 0.75,
                "sentences": [
                    {"sentence": "First sentence.", "generated_prob": 0.95},
                    {"sentence": "Second sentence.", "generated_prob": 0.12},
                ],
            }
        ]
    }

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response)

    result = detector.score("First sentence. Second sentence.")
    assert result.score == pytest.approx(91.0)
    assert len(result.windows) == 2
    assert result.windows[0].score == pytest.approx(95.0)


def test_pangram_http_error(monkeypatch) -> None:
    detector = PangramDetector(api_key="test-key")

    response = MagicMock()
    response.status_code = 401
    response.text = "unauthorized"
    response.reason_phrase = "Unauthorized"
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response)

    with pytest.raises(RemoteDetectorError, match="401"):
        detector.score("Some text long enough.")


def test_load_remote_detectors_use_env(monkeypatch) -> None:
    monkeypatch.setenv("PANGRAM_API_KEY", "p-key")
    monkeypatch.setenv("GPTZERO_API_KEY", "g-key")
    assert load_detector("pangram").name == "pangram"
    assert load_detector("gptzero").name == "gptzero"
