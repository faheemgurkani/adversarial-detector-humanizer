"""In-memory idempotency cache for expensive POST routes."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from adh.exceptions import IdempotencyConflictError

DEFAULT_TTL_SECONDS = 24 * 60 * 60


@dataclass
class _Entry:
    body_hash: str
    response: dict[str, Any]
    report_id: str
    created_at: float


class IdempotencyStore:
    """Cache humanize responses keyed by idempotency key + request body hash."""

    def __init__(self, *, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._entries: dict[str, _Entry] = {}

    @staticmethod
    def hash_body(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _purge_expired(self, now: float | None = None) -> None:
        current = now if now is not None else time.time()
        expired = [
            key
            for key, entry in self._entries.items()
            if current - entry.created_at > self._ttl
        ]
        for key in expired:
            del self._entries[key]

    def lookup(self, key: str, body_hash: str) -> dict[str, Any] | None:
        """Return a cached response or raise when the key was reused with another body."""
        self._purge_expired()
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.body_hash != body_hash:
            raise IdempotencyConflictError(
                "Idempotency-Key was already used with a different request body."
            )
        return dict(entry.response)

    def store(
        self,
        key: str,
        *,
        body_hash: str,
        response: dict[str, Any],
        report_id: str,
    ) -> None:
        self._purge_expired()
        self._entries[key] = _Entry(
            body_hash=body_hash,
            response=dict(response),
            report_id=report_id,
            created_at=time.time(),
        )
