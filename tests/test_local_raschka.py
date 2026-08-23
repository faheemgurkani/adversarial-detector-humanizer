from __future__ import annotations

import pytest

from adh.detectors.local_raschka import _build_causal_batch
from adh.exceptions import DetectorNotReadyError
from adh.factory import load_detector


def test_causal_batch_variable() -> None:
    rows, masks = _build_causal_batch(
        [[1, 2], [3]],
        pad_token_id=0,
        eos_token_id=9,
        context_length=8,
        readout_position="variable",
    )
    assert rows[0] == [1, 2, 9]
    assert rows[1] == [3, 9, 0]
    assert masks[0] == [1, 1, 1]
    assert masks[1] == [1, 1, 0]


def test_causal_batch_fixed() -> None:
    rows, masks = _build_causal_batch(
        [[1, 2]],
        pad_token_id=0,
        eos_token_id=9,
        context_length=5,
        readout_position="fixed",
    )
    assert rows[0] == [1, 2, 0, 0, 9]
    assert masks[0] == [1, 1, 0, 0, 1]


def test_causal_batch_empty_and_errors() -> None:
    assert _build_causal_batch(
        [],
        pad_token_id=0,
        eos_token_id=9,
        context_length=4,
        readout_position="variable",
    ) == ([], [])
    with pytest.raises(DetectorNotReadyError):
        _build_causal_batch(
            [[1, 2, 3, 4]],
            pad_token_id=0,
            eos_token_id=9,
            context_length=3,
            readout_position="variable",
        )
    with pytest.raises(DetectorNotReadyError):
        _build_causal_batch(
            [[1]],
            pad_token_id=0,
            eos_token_id=9,
            context_length=4,
            readout_position="sideways",
        )


def test_local_detector_not_ready(tmp_path) -> None:
    detector = load_detector("distilbert", models_dir=tmp_path)
    assert detector.ready() is False
    with pytest.raises(DetectorNotReadyError):
        detector.score("A sentence that would require weights.")
