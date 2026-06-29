"""Turn-based chunking of a chat transcript (pure, Django-free).

Groups an ordered message list into user->assistant *turns*, then packs
consecutive turns into windows up to a target character size, carrying a small
turn-level overlap between windows so context isn't lost at boundaries. Each
window keeps the inclusive ``sequence_number`` range it covers so callers can
link a chunk back to the messages it came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class TranscriptMessage:
    """A single message, decoupled from the Django ``Message`` model."""

    role: str
    content: str
    sequence_number: int


@dataclass
class _Turn:
    """One user->assistant exchange (tool messages folded into the assistant)."""

    start_sequence: int
    end_sequence: int
    text: str
    roles: list[str] = field(default_factory=list)


@dataclass
class TurnWindow:
    """A packed window of one or more turns, ready to embed and persist."""

    content: str
    start_sequence: int
    end_sequence: int
    metadata: dict


_ROLE_LABELS = {"user": "User", "assistant": "Assistant", "tool": "Tool result"}


def _format_message(role: str, content: str) -> str:
    label = _ROLE_LABELS.get(role, role.capitalize())
    return f"{label}: {content.strip()}"


def _group_into_turns(messages: list[TranscriptMessage]) -> list[_Turn]:
    """Split the ordered transcript into turns.

    A turn starts at a ``user`` message and absorbs the following assistant/tool
    messages until the next user message. Leading assistant/tool messages (e.g. a
    greeting before any user input) form their own turn so nothing is dropped.
    """
    turns: list[_Turn] = []
    current: list[TranscriptMessage] = []

    def flush() -> None:
        if not current:
            return
        text = "\n\n".join(
            _format_message(m.role, m.content) for m in current if (m.content or "").strip()
        )
        if text.strip():
            turns.append(
                _Turn(
                    start_sequence=current[0].sequence_number,
                    end_sequence=current[-1].sequence_number,
                    text=text,
                    roles=[m.role for m in current],
                )
            )
        current.clear()

    for msg in messages:
        if msg.role == "user" and current:
            flush()
        current.append(msg)
    flush()
    return turns


def build_turn_windows(
    messages: Iterable[TranscriptMessage],
    target_size: int,
    overlap: int,
) -> list[TurnWindow]:
    """Pack ordered turns into windows up to ``target_size`` chars.

    ``overlap`` is a character budget: after closing a window, trailing turns whose
    combined length fits in ``overlap`` are repeated at the start of the next window
    so cross-window context is preserved. A single turn longer than ``target_size``
    becomes its own (oversized) window rather than being split mid-turn.
    """
    ordered = sorted(messages, key=lambda m: m.sequence_number)
    turns = _group_into_turns(ordered)
    if not turns:
        return []

    target = max(1, int(target_size))
    overlap_budget = max(0, int(overlap))

    windows: list[TurnWindow] = []
    buffer: list[_Turn] = []
    buffer_len = 0

    def emit() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        content = "\n\n".join(t.text for t in buffer)
        windows.append(
            TurnWindow(
                content=content,
                start_sequence=buffer[0].start_sequence,
                end_sequence=buffer[-1].end_sequence,
                metadata={
                    "turn_count": len(buffer),
                    "roles": [r for t in buffer for r in t.roles],
                },
            )
        )
        # Seed the next buffer with trailing turns that fit the overlap budget.
        carry: list[_Turn] = []
        carry_len = 0
        for t in reversed(buffer):
            extra = len(t.text) + (2 if carry else 0)
            if carry_len + extra > overlap_budget:
                break
            carry.insert(0, t)
            carry_len += extra
        buffer = carry
        buffer_len = sum(len(t.text) for t in buffer) + max(0, len(buffer) - 1) * 2

    for turn in turns:
        addition = len(turn.text) + (2 if buffer else 0)
        if buffer and buffer_len + addition > target:
            emit()
            addition = len(turn.text) + (2 if buffer else 0)
        buffer.append(turn)
        buffer_len += addition

    emit()
    # Number windows and drop a possible all-overlap trailing duplicate.
    deduped: list[TurnWindow] = []
    for win in windows:
        if deduped and win.start_sequence >= deduped[-1].start_sequence and win.end_sequence <= deduped[-1].end_sequence:
            continue
        deduped.append(win)
    return deduped


__all__ = ["TranscriptMessage", "TurnWindow", "build_turn_windows"]
