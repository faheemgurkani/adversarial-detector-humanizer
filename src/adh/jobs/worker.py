"""Background worker that runs pending humanize jobs."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from adh.jobs.store import JobRecord, JobStore


class JobWorker:
    """Poll ``JobStore`` and invoke a handler for each pending job."""

    def __init__(
        self,
        store: JobStore,
        handler: Callable[[JobRecord], None],
        *,
        poll_interval: float = 0.05,
    ) -> None:
        self._store = store
        self._handler = handler
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="adh-job-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def process_once(self) -> bool:
        """Process one pending job synchronously. Returns True if work ran."""
        job = self._store.claim_pending()
        if job is None:
            return False
        self._handler(job)
        return True

    def drain(self) -> None:
        """Process all currently pending jobs (used by tests and CLI --async)."""
        while self.process_once():
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self.process_once():
                time.sleep(self._poll_interval)


def build_humanize_handler(
    get_context: Callable[[], dict[str, Any]],
) -> Callable[[JobRecord], None]:
    """Build a job handler that calls ``service.run_humanize()``."""
    from adh.errors import error_response
    from adh.exceptions import AdhError
    from adh.ids import new_request_id
    from adh.schemas import HumanizeRequest
    from adh.service import run_humanize

    def _serialize(report: Any, payload: HumanizeRequest) -> dict[str, Any]:
        from adh.schemas import compact_from_report

        if payload.compact:
            return compact_from_report(report, metadata=payload.metadata).model_dump()
        body = report.model_dump()
        if payload.metadata:
            body["metadata"] = dict(payload.metadata)
        return body

    def _handler(job: JobRecord) -> None:
        context = get_context()
        store: JobStore = context["store"]
        payload = HumanizeRequest.model_validate(job.request)
        try:
            report = run_humanize(
                payload.text,
                config=payload,
                file=context.get("file_config"),
                detector=context.get("detector"),
                rewriter=context.get("rewriter"),
                semantic_gate=context.get("semantic_gate"),
                default_detector=context.get("default_detector"),
                device=context.get("device"),
                models_dir=context.get("models_dir"),
            )
            response = _serialize(report, payload)
            store.mark_done(
                job.job_id,
                report_id=str(report.report_id),
                report=response,
            )
        except AdhError as error:
            store.mark_failed(
                job.job_id,
                error=error_response(error, new_request_id()),
            )

    return _handler
