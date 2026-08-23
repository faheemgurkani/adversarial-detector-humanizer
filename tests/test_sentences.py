from __future__ import annotations

import pytest

from adh.exceptions import InputError
from adh.sentences import reassemble, split_sentences


def test_empty_text_raises() -> None:
    with pytest.raises(InputError):
        split_sentences("   ")


def test_offsets_cover_original() -> None:
    text = "Dr. Smith arrived. The room was quiet."
    spans = split_sentences(text)
    assert len(spans) >= 1
    rebuilt = "".join(
        text[spans[0].start : spans[0].end]
        if index == 0
        else text[spans[index - 1].end : spans[index].end]
        for index in range(len(spans))
    )
    assert rebuilt.endswith(spans[-1].text)
    assert all(span.start < span.end for span in spans)


def test_single_sentence_without_terminator() -> None:
    text = "Just one clause without a stop"
    spans = split_sentences(text)
    assert len(spans) == 1
    assert spans[0].text.strip() == text


def test_reassemble_out_of_range() -> None:
    with pytest.raises(InputError):
        reassemble("Hello there.", {9: "Nope"})


def test_reassemble_replaces_one_sentence() -> None:
    text = "Alpha is first. Beta is second."
    spans = split_sentences(text)
    assert len(spans) >= 2
    rewritten = reassemble(text, {0: "Alpha changed."})
    assert "Alpha changed." in rewritten
    assert "Beta is second." in rewritten
