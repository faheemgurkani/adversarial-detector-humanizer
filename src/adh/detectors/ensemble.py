"""Composite detector builders registered as ``adh.detectors`` entry points."""

from __future__ import annotations

from pathlib import Path

from adh.detectors.fake import FakeDetector
from adh.detectors.local_raschka import LocalRaschkaDetector
from adh.detectors.remote import EnsembleDetector
from adh.detectors.statistical import StatisticalDetector
from adh.models import DEFAULT_MODEL


def build_ensemble() -> EnsembleDetector:
    """Fake-only ensemble kept for the historical ``ensemble`` name."""
    return EnsembleDetector([FakeDetector()], aggregate="max")


def build_ensemble_max() -> EnsembleDetector:
    """Same as ``build_ensemble``; historical ``ensemble-max`` alias."""
    return EnsembleDetector([FakeDetector()], aggregate="max")


def build_ensemble_local(
    *,
    models_dir: Path | str | None = None,
    device: str = "auto",
) -> EnsembleDetector:
    """Max ensemble of the default Raschka export plus the statistical detector."""
    return EnsembleDetector(
        [
            LocalRaschkaDetector(
                DEFAULT_MODEL,
                models_dir=models_dir,
                device=device,
            ),
            StatisticalDetector(),
        ],
        aggregate="max",
    )
