from __future__ import annotations

import ast
import time
from importlib.metadata import EntryPoint, EntryPoints
from importlib.metadata import entry_points as metadata_entry_points
from pathlib import Path

import pytest

from adh.detectors.fake import FakeDetector
from adh.detectors.local_raschka import LocalRaschkaDetector
from adh.detectors.statistical import StatisticalDetector
from adh.exceptions import InputError
from adh.factory import load_detector, load_gate, load_meaning_gate_stack, load_rewriter
from adh.registry import (
    GROUP_DETECTORS,
    GROUP_GATES,
    GROUP_REWRITERS,
    list_plugins,
    load_plugin,
)
from adh.rewriter import IdentityRewriter
from adh.semantic import LexicalSemanticGate
from adh.service import run_score

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures_plugin"
FACTORY_PATH = ROOT / "src" / "adh" / "factory.py"
REGISTRY_PATH = ROOT / "src" / "adh" / "registry.py"


def test_load_builtin_fake_detector() -> None:
    detector = load_detector("fake")
    assert isinstance(detector, FakeDetector)
    assert detector.name == "fake"


def test_load_all_entry_point_detectors(monkeypatch) -> None:
    monkeypatch.setenv("PANGRAM_API_KEY", "test-pangram")
    monkeypatch.setenv("GPTZERO_API_KEY", "test-gptzero")
    names = list_plugins(GROUP_DETECTORS)
    assert "fake" in names
    assert "statistical" in names
    assert "ensemble-local" in names
    for name in names:
        loaded = load_detector(name)
        assert loaded is not None
        assert getattr(loaded, "name", None)


def test_unknown_detector_lists_available() -> None:
    with pytest.raises(InputError, match="fake") as caught:
        load_detector("nope")
    message = str(caught.value)
    assert "adh.detectors" in message
    assert "statistical" in message


def test_third_party_entry_point(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(FIXTURE_DIR))
    extra = EntryPoint(
        name="acme-nlp",
        value="acme_plugin:AcmeDetector",
        group=GROUP_DETECTORS,
    )

    def combined(*, group: str | None = None, name: str | None = None):
        kwargs: dict[str, str] = {}
        if group is not None:
            kwargs["group"] = group
        if name is not None:
            kwargs["name"] = name
        discovered = list(metadata_entry_points(**kwargs))
        if group in {None, GROUP_DETECTORS} and name in {None, "acme-nlp"}:
            discovered.append(extra)
        return EntryPoints(discovered)

    monkeypatch.setattr("adh.registry.entry_points", combined)
    loaded = load_plugin(GROUP_DETECTORS, "acme-nlp")
    assert loaded.name == "acme-nlp"
    assert loaded.score("A complete sentence for scoring.").score == 10.0


def test_registry_cold_start_under_200ms() -> None:
    started = time.perf_counter()
    detector = load_detector("fake")
    elapsed = time.perf_counter() - started
    assert detector.name == "fake"
    assert elapsed < 0.2


def test_service_uses_registry() -> None:
    loaded, result = run_score(
        "The result is clear. The result is clear. The result is clear.",
        detector_name="statistical",
    )
    assert loaded.name == "statistical"
    assert 0.0 <= result.score <= 100.0


def test_raschka_entry_point_injects_model_name() -> None:
    detector = load_detector("distilbert")
    assert isinstance(detector, LocalRaschkaDetector)
    assert detector.name == "distilbert"


def test_identity_rewriter_via_registry() -> None:
    rewriter = load_rewriter(name="identity")
    assert isinstance(rewriter, IdentityRewriter)
    assert rewriter.name == "identity"


def test_unknown_rewriter_lists_available() -> None:
    with pytest.raises(InputError, match="identity"):
        load_rewriter(name="nope")
    assert "openai" in ", ".join(list_plugins(GROUP_REWRITERS))


def test_lexical_gate_via_registry() -> None:
    gate = load_gate(prefer="lexical", allow_lexical=True)
    assert isinstance(gate, LexicalSemanticGate)
    assert gate.name == "lexical"


def test_meaning_gate_stack_via_registry() -> None:
    stack = load_meaning_gate_stack(prefer="lexical", allow_lexical=True)
    assert "lexical" in stack.name


def test_factory_has_no_name_if_chain() -> None:
    source = FACTORY_PATH.read_text(encoding="utf-8")
    assert 'if name == "fake"' not in source
    assert "if name ==" not in source
    assert "pkg_resources" not in source


def test_registry_uses_importlib_metadata_not_pkg_resources() -> None:
    tree = ast.parse(REGISTRY_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "pkg_resources" not in imported
    source = REGISTRY_PATH.read_text(encoding="utf-8")
    assert "importlib.metadata" in source
    assert "entry_points" in source


def test_statistical_still_loads() -> None:
    detector = load_detector("statistical")
    assert isinstance(detector, StatisticalDetector)
    assert detector.name == "statistical"


def test_unknown_gate_lists_available() -> None:
    with pytest.raises(InputError) as caught:
        load_gate(prefer="nope")
    message = str(caught.value)
    assert "lexical" in message
    assert GROUP_GATES in message
