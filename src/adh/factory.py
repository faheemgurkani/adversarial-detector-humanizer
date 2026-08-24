"""Construct detectors, rewriters, and gates from CLI/API options."""

from __future__ import annotations

from pathlib import Path

from adh.detectors.base import Detector
from adh.detectors.fake import FakeDetector
from adh.detectors.local_raschka import LocalRaschkaDetector
from adh.detectors.remote import (
    EnsembleDetector,
    GPTZeroDetector,
    PangramDetector,
    assert_inner_loop_detector,
)
from adh.exceptions import InputError
from adh.models import DEFAULT_MODEL
from adh.gates import build_meaning_gate_stack
from adh.gates.stack import MeaningGateStack
from adh.rewriter import OpenAICompatibleRewriter, Rewriter
from adh.semantic import SemanticGate, build_semantic_gate


def load_detector(
    name: str,
    *,
    models_dir: Path | str | None = None,
    device: str = "auto",
) -> Detector:
    if name == "fake":
        return FakeDetector()
    if name == "pangram":
        return PangramDetector()
    if name == "gptzero":
        return GPTZeroDetector()
    if name == "ensemble":
        return EnsembleDetector([FakeDetector()], aggregate="max")
    if name == "ensemble-max":
        return EnsembleDetector([FakeDetector()], aggregate="max")
    if name == DEFAULT_MODEL or name in {
        "logreg",
        "distilbert",
        "distilbert-lora",
        "distilbert-mica",
        "modernbert",
        "gpt2-variable",
        "gpt2-fixed",
        "qwen3-variable",
        "qwen3-fixed",
    }:
        return LocalRaschkaDetector(name, models_dir=models_dir, device=device)
    raise InputError(f"unknown detector {name!r}")


def load_rewriter(*, model: str | None = None) -> Rewriter:
    return OpenAICompatibleRewriter(model=model)


def load_gate(*, prefer: str = "auto", allow_lexical: bool = False) -> SemanticGate:
    return build_semantic_gate(prefer=prefer, allow_lexical=allow_lexical)


def load_meaning_gate_stack(**kwargs) -> MeaningGateStack:
    return build_meaning_gate_stack(**kwargs)
