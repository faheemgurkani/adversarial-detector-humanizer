"""In-memory job store for async humanize requests."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from adh.exceptions import IdempotencyConflictError, InputError
from adh.ids import new_job_id
from adh.schemas import HumanizeRequest

JobStatus = Literal["pending", "processing", "done", "failed"]


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    body_hash: str
    request: dict[str, Any]
    metadata: dict[str, str] = field(default_factory=dict)
    report_id: str | None = None
    report: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class JobStore:
    """Thread-safe in-memory job registry with create idempotency."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._idempotency: dict[str, tuple[str, str]] = {}
        self._lock = threading.Lock()

    def create(
        self,
        payload: HumanizeRequest,
        *,
        body_hash: str,
        idempotency_key: str | None = None,
    ) -> tuple[JobRecord, bool]:
        """Return ``(record, created)``. Replays return the existing record."""
        with self._lock:
            if idempotency_key:
                existing = self._idempotency.get(idempotency_key)
                if existing is not None:
                    prior_hash, job_id = existing
                    if prior_hash != body_hash:
                        raise IdempotencyConflictError(
                            "Idempotency-Key was already used with a different request body."
                        )
                    record = self._jobs.get(job_id)
                    if record is not None:
                        return record, False

            job_id = new_job_id()
            record = JobRecord(
                job_id=job_id,
                status="pending",
                body_hash=body_hash,
                request=payload.model_dump(mode="json"),
                metadata=dict(payload.metadata),
            )
            self._jobs[job_id] = record
            if idempotency_key:
                self._idempotency[idempotency_key] = (body_hash, job_id)
            return record, True

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def require(self, job_id: str) -> JobRecord:
        record = self.get(job_id)
        if record is None:
            raise InputError(f"unknown job {job_id!r}")
        return record

    def claim_pending(self) -> JobRecord | None:
        with self._lock:
            for record in self._jobs.values():
                if record.status == "pending":
                    record.status = "processing"
                    record.updated_at = time.time()
                    return record
        return None

    def mark_done(
        self,
        job_id: str,
        *,
        report_id: str,
        report: dict[str, Any],
    ) -> JobRecord:
        with self._lock:
            record = self._jobs[job_id]
            record.status = "done"
            record.report_id = report_id
            record.report = dict(report)
            record.error = None
            record.updated_at = time.time()
            return record

    def mark_failed(self, job_id: str, *, error: dict[str, Any]) -> JobRecord:
        with self._lock:
            record = self._jobs[job_id]
            record.status = "failed"
            record.error = dict(error)
            record.updated_at = time.time()
            return record

    def to_response(self, record: JobRecord) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": record.job_id,
            "status": record.status,
            "metadata": dict(record.metadata),
        }
        if record.report_id is not None:
            payload["report_id"] = record.report_id
        if record.report is not None:
            payload["report"] = dict(record.report)
        if record.error is not None:
            payload["error"] = dict(record.error)
        return payload
