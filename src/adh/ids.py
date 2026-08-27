"""Prefixed identifiers for reports, jobs, and HTTP requests."""

from __future__ import annotations

import secrets


def new_report_id() -> str:
    return f"report_{secrets.token_hex(8)}"


def new_job_id() -> str:
    return f"job_{secrets.token_hex(8)}"


def new_request_id() -> str:
    return f"req_{secrets.token_hex(8)}"
