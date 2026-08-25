from __future__ import annotations

from adh.engine import EngineConfig, humanize
from adh.hard.guided_decode import pick_token_by_detector
from adh.rewriter import ScriptedRewriter


def test_pick_token_by_detector_chooses_lowest_score() -> None:
    chosen = pick_token_by_detector(
        token_ids=[1, 2, 3],
        token_probs=[0.5, 0.3, 0.2],
        prefix_text="Hello ",
        decode_token=lambda token_id: {1: "world", 2: "there", 3: "friend"}[token_id],
        detector_scores=lambda texts: [90.0 if "world" in text else 10.0 for text in texts],
    )
    assert chosen in {2, 3}


def test_humanize_attaches_detector_breakdown(lexical_gate) -> None:
    guidance = CueDetector()

    class DeployDetector:
        name = "deploy"

        def score(self, text: str):
            from adh.detectors.base import probability_to_result, require_text

            require_text(text)
            return probability_to_result(0.6 if "Furthermore" in text else 0.3)

        def score_spans(self, texts: list[str]):
            return [self.score(text) for text in texts]

    from unittest.mock import patch

    with patch("adh.engine.load_detector", return_value=DeployDetector()):
        report = humanize(
            "Furthermore, the method is important to note in 2024.",
            detector=guidance,
            rewriter=ScriptedRewriter(
                {"Furthermore, the method is important to note in 2024.": ["The method mattered in 2024."]}
            ),
            semantic_gate=lexical_gate,
            config=EngineConfig(
                target_score=30,
                max_rounds=2,
                sentence_threshold=50,
                min_semantic_similarity=0.2,
                deploy_detectors=["deploy"],
            ),
        )
    assert report.detector_breakdown is not None
    assert report.detector_breakdown.guidance == "cue"
    assert len(report.detector_breakdown.entries) == 2
