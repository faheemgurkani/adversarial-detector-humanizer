#!/usr/bin/env python3
"""Minimal benchmark harness for ADH."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "benchmarks" / "samples.jsonl"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adh.engine import EngineConfig, humanize
from adh.factory import load_detector, load_gate
from adh.rewriter import IdentityRewriter
from adh.semantic import LexicalSemanticGate


def _load_corpus(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def cmd_score(args: argparse.Namespace) -> int:
    detector = load_detector(args.detector)
    rows = _load_corpus(args.corpus)
    output = []
    for row in rows:
        started = time.perf_counter()
        score = detector.score(row["text"]).score
        output.append(
            {
                "id": row["id"],
                "score": score,
                "seconds": round(time.perf_counter() - started, 4),
            }
        )
    print(json.dumps(output, indent=2))
    return 0


def cmd_humanize(args: argparse.Namespace) -> int:
    detector = load_detector(args.detector)
    gate = LexicalSemanticGate() if args.detector == "fake" else load_gate(prefer="lexical", allow_lexical=True)
    rewriter = IdentityRewriter()
    rows = _load_corpus(args.corpus)
    output = []
    for row in rows:
        started = time.perf_counter()
        report = humanize(
            row["text"],
            detector=detector,
            rewriter=rewriter,
            semantic_gate=gate,
            config=EngineConfig(detector=args.detector),
        )
        output.append(
            {
                "id": row["id"],
                "score_before": report.score_before,
                "score_after": report.score_after,
                "semantic_similarity": report.semantic_similarity,
                "stop_reason": report.stop_reason,
                "seconds": round(time.perf_counter() - started, 4),
            }
        )
    print(json.dumps(output, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ADH benchmark harness")
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("--detector", default="fake")
    score.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    score.set_defaults(func=cmd_score)
    humanize_cmd = sub.add_parser("humanize")
    humanize_cmd.add_argument("--detector", default="fake")
    humanize_cmd.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    humanize_cmd.set_defaults(func=cmd_humanize)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
