"""Unit tests for the pure turn-window chunker."""
from __future__ import annotations

from lib.conversations.chunking import TranscriptMessage, build_turn_windows


def _msgs(*pairs):
    return [
        TranscriptMessage(role=role, content=content, sequence_number=i)
        for i, (role, content) in enumerate(pairs)
    ]


def test_empty_transcript_yields_no_windows():
    assert build_turn_windows([], target_size=100, overlap=10) == []


def test_single_turn_packs_into_one_window():
    msgs = _msgs(("user", "hello there"), ("assistant", "general kenobi"))
    windows = build_turn_windows(msgs, target_size=1000, overlap=0)
    assert len(windows) == 1
    w = windows[0]
    assert "User: hello there" in w.content
    assert "Assistant: general kenobi" in w.content
    assert w.start_sequence == 0
    assert w.end_sequence == 1


def test_large_transcript_splits_into_multiple_windows():
    pairs = []
    for n in range(6):
        pairs.append(("user", f"question number {n} " + "x" * 40))
        pairs.append(("assistant", f"answer number {n} " + "y" * 40))
    msgs = _msgs(*pairs)
    windows = build_turn_windows(msgs, target_size=200, overlap=0)
    assert len(windows) > 1
    # Sequence ranges are ordered and non-overlapping with overlap=0.
    for earlier, later in zip(windows, windows[1:]):
        assert earlier.end_sequence < later.start_sequence
    # Every message sequence is covered.
    assert windows[0].start_sequence == 0
    assert windows[-1].end_sequence == len(msgs) - 1


def test_overlap_repeats_trailing_turn():
    # Each turn is ~85 chars; with overlap >= one turn, the trailing turn of a
    # window is carried into the next so their sequence ranges overlap.
    pairs = []
    for n in range(6):
        pairs.append(("user", f"q{n} " + "x" * 30))
        pairs.append(("assistant", f"a{n} " + "y" * 30))
    msgs = _msgs(*pairs)
    windows = build_turn_windows(msgs, target_size=300, overlap=150)
    assert len(windows) > 1
    # With overlap, a later window should start at or before the previous window's end.
    assert windows[1].start_sequence <= windows[0].end_sequence


def test_messages_sorted_by_sequence():
    msgs = [
        TranscriptMessage(role="assistant", content="second", sequence_number=1),
        TranscriptMessage(role="user", content="first", sequence_number=0),
    ]
    windows = build_turn_windows(msgs, target_size=1000, overlap=0)
    assert windows[0].content.index("first") < windows[0].content.index("second")
