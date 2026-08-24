from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_humanize_smoke() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "benchmark.py"), "humanize", "--detector", "fake"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    payload = json.loads(result.stdout)
    assert len(payload) >= 3
    assert all("score_before" in row and "score_after" in row for row in payload)
