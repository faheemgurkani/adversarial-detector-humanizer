"""Run a single humanize job synchronously (CLI and tests)."""

from __future__ import annotations

from typing import Any

from adh.idempotency import IdempotencyStore
from adh.jobs.store import JobRecord, JobStore
from adh.jobs.worker import JobWorker, build_humanize_handler
from adh.schemas import HumanizeRequest


def execute_humanize_job(
    payload: HumanizeRequest,
    *,
    context: dict[str, Any],
    idempotency_key: str | None = None,
    store: JobStore | None = None,
) -> JobRecord:
    """Create a job, process it inline, and return the final record."""
    job_store = store or JobStore()
    body_hash = IdempotencyStore.hash_body(payload.model_dump(mode="json"))
    record, _created = job_store.create(
        payload,
        body_hash=body_hash,
        idempotency_key=idempotency_key,
    )
    runtime = dict(context)
    runtime["store"] = job_store
    worker = JobWorker(job_store, build_humanize_handler(lambda: runtime))
    worker.drain()
    return job_store.require(record.job_id)
