"""
VTT (WebVTT) caption parsing.

Note: The full VTT parsing is handled by aquillm.vtt module.
This provides the SRT fallback parser.
"""

import re
from typing import Iterable


def iter_srt_text(lines: list[str]) -> Iterable[str]:
    """Extract spoken content from SRT lines, stripping indices and timestamps."""
    timestamp_re = re.compile(r"^\d{2}:\d{2}:\d{2}[,\.]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[,\.]\d{3}")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.isdigit():
            continue
        if timestamp_re.match(stripped):
            continue
        yield stripped


def extract_srt_text(text: str) -> str:
    """Extract spoken content from SRT text."""
    lines = list(iter_srt_text(text.splitlines()))
    return "\n".join(lines)


__all__ = ['iter_srt_text', 'extract_srt_text']
