"""Construct detectors, rewriters, and gates from the plugin registry."""

from __future__ import annotations

from pathlib import Path

from adh.detectors.base import Detector
from adh.detectors.remote import assert_inner_loop_detector
from adh.gates.stack import MeaningGateStack
from adh.registry import (
    GROUP_DETECTORS,
    GROUP_GATES,
    GROUP_REWRITERS,
    load_plugin,
)
from adh.rewriter import Rewriter
from adh.semantic import SemanticGate

__all__ = [
    "assert_inner_loop_detector",
    "load_detector",
    "load_gate",
    "load_meaning_gate_stack",
    "load_rewriter",
]


def load_detector(
    name: str,
    *,
    models_dir: Path | str | None = None,
    device: str = "auto",
) -> Detector:
    return load_plugin(
        GROUP_DETECTORS,
        name,
        models_dir=models_dir,
        device=device,
    )


def load_rewriter(*, name: str | None = None, model: str | None = None) -> Rewriter:
    backend = (name or "openai").strip().lower() or "openai"
    return load_plugin(GROUP_REWRITERS, backend, model=model)


def load_gate(*, prefer: str = "auto", allow_lexical: bool = False) -> SemanticGate:
    return load_plugin(
        GROUP_GATES,
        prefer,
        prefer=prefer,
        allow_lexical=allow_lexical,
    )


def load_meaning_gate_stack(**kwargs) -> MeaningGateStack:
    return load_plugin(GROUP_GATES, "meaning", **kwargs)
