"""Bounded whole-source partitioning; no query, score or answer-label selection."""

import re

_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n[ \t]*\n")
_UNSAFE = re.compile(r"[<>\[\]`]|!\(|[\x00-\x08\x0b-\x1f]")


def partition_complete_source(text, *, slots, codepoints):
    """Return contiguous complete spans, or reject the entire source atomically.

    Limits are the remaining answer-wide allowance. Whitespace belongs to the
    preceding span, and exact reassembly is checked before any rendering occurs.
    A long sentence is never clipped to manufacture a permitted fragment.
    """
    if (
        not isinstance(text, str)
        or not text
        or type(slots) is not int
        or not 0 < slots <= 5
        or type(codepoints) is not int
        or not 0 < codepoints <= 6000
        or len(text) > codepoints
        or _UNSAFE.search(text)
    ):
        return None
    ends = [match.end() for match in _BOUNDARY.finditer(text)]
    if not ends or ends[-1] != len(text):
        ends.append(len(text))
    spans, start, end = [], 0, 0
    for boundary in ends:
        if boundary - start > 1200:
            if end == start:
                return None
            spans.append(text[start:end])
            start = end
        if boundary - start > 1200 or len(spans) >= slots:
            return None
        end = boundary
    if end > start:
        spans.append(text[start:end])
    if not spans or len(spans) > slots or "".join(spans) != text:
        return None
    return tuple(spans)
