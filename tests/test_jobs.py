from __future__ import annotations

import json
import re
from typing import Any

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from adh.api import create_app
from adh.detectors.fake import FakeDetector
from adh.jobs.store import JobStore
from adh.rewriter import IdentityRewriter
from adh.semantic import LexicalSemanticGate
from tests.conftest import CueDetector

runner = CliRunner()
JOB_ID = re.compile(r"^job_[0-9a-f]{16}$")
REPORT_ID = re.compile(r"^report_[0-9a-f]{16}$")


def _job_client(**kwargs: Any) -> tuple[TestClient, Any]:
    app = create_app(
        detector=kwargs.get("detector", CueDetector()),
        rewriter=kwargs.get("rewriter", IdentityRewriter()),
        semantic_gate=kwargs.get("gate", LexicalSemanticGate()),
        default_detector=kwargs.get("default_detector", "cue"),
        job_store=kwargs.get("job_store"),
        start_job_worker=False,
    )
    return TestClient(app), app


def _payload(**overrides: object) -> dict:
    body = {
        "text": "This is a human sentence.",
        "detector": "cue",
        "allow_lexical_gate": True,
        "semantic": "lexical",
        "min_semantic_similarity": 0.2,
        "metadata": {"ticket": "JIRA-123"},
    }
    body.update(overrides)
    return body


def test_create_job_returns_202() -> None:
    client, app = _job_client()
    response = client.post("/v1/jobs/humanize", json=_payload())
    assert response.status_code == 202
    body = response.json()
    assert JOB_ID.match(body["job_id"])
    assert body["status"] == "pending"
    assert response.headers["Location"] == f"/v1/jobs/{body['job_id']}"
    app.state.job_worker.drain()


def test_poll_until_done() -> None:
    client, app = _job_client()
    created = client.post("/v1/jobs/humanize", json=_payload())
    job_id = created.json()["job_id"]
    app.state.job_worker.drain()
    polled = client.get(f"/v1/jobs/{job_id}")
    assert polled.status_code == 200
    body = polled.json()
    assert body["status"] == "done"
    assert REPORT_ID.match(body["report_id"])
    assert body["report"] is not None
    assert "output_text" in body["report"]


def test_get_job_always_200_while_polling() -> None:
    client, app = _job_client()
    created = client.post("/v1/jobs/humanize", json=_payload())
    job_id = created.json()["job_id"]
    pending = client.get(f"/v1/jobs/{job_id}")
    assert pending.status_code == 200
    assert pending.json()["status"] in {"pending", "processing"}
    app.state.job_worker.drain()
    done = client.get(f"/v1/jobs/{job_id}")
    assert done.status_code == 200
    assert done.json()["status"] == "done"


def test_failed_job_structured_error(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client, app = _job_client(
        detector=FakeDetector(document_score=80.0),
        rewriter=None,
        default_detector="fake",
    )
    created = client.post(
        "/v1/jobs/humanize",
        json={"text": "Furthermore, note this.", "detector": "fake"},
    )
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    app.state.job_worker.drain()
    polled = client.get(f"/v1/jobs/{job_id}")
    assert polled.status_code == 200
    body = polled.json()
    assert body["status"] == "failed"
    assert body["error"]["code"] == "rewriter_unavailable"


def test_job_idempotency() -> None:
    store = JobStore()
    client, app = _job_client(job_store=store)
    headers = {"Idempotency-Key": "job-idem-1"}
    first = client.post("/v1/jobs/humanize", json=_payload(), headers=headers)
    second = client.post("/v1/jobs/humanize", json=_payload(), headers=headers)
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["job_id"] == second.json()["job_id"]
    app.state.job_worker.drain()


def test_job_idempotency_conflict() -> None:
    client, _app = _job_client()
    headers = {"Idempotency-Key": "job-idem-2"}
    first = client.post("/v1/jobs/humanize", json=_payload(), headers=headers)
    assert first.status_code == 202
    second = client.post(
        "/v1/jobs/humanize",
        json=_payload(text="Different sentence entirely here."),
        headers=headers,
    )
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "idempotency_key_reused"


def test_job_metadata_persisted() -> None:
    client, app = _job_client()
    created = client.post("/v1/jobs/humanize", json=_payload())
    job_id = created.json()["job_id"]
    assert created.json()["metadata"] == {"ticket": "JIRA-123"}
    app.state.job_worker.drain()
    polled = client.get(f"/v1/jobs/{job_id}")
    assert polled.json()["metadata"] == {"ticket": "JIRA-123"}


def test_cli_async_humanize(monkeypatch) -> None:
    from adh.cli import app as cli_app

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    result = runner.invoke(
        cli_app,
        [
            "humanize",
            "--async",
            "--profile",
            "fast",
            "--text",
            "Furthermore, note this.",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert JOB_ID.match(body["job_id"])
    assert body["status"] == "done"
    assert "output_text" in body


def test_openapi_lists_job_routes() -> None:
    client, _app = _job_client()
    response = client.get("/openapi.json")
    paths = response.json()["paths"]
    assert "/v1/jobs/humanize" in paths
    assert "/v1/jobs/{job_id}" in paths
