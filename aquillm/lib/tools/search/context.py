"""More-context tool formatting (adjacent chunks)."""
from __future__ import annotations

from typing import Any, Callable, Sequence


def format_adjacent_chunks_tool_result(
    window: Sequence[Any],
    *,
    truncate: Callable[[str], str],
) -> dict[str, str]:
    """Build tool result text from ordered chunks with chunk_number and content attributes."""
    text_blob = "".join(chunk.content for chunk in window)
    text_blob = truncate(text_blob)
    return {
        "result": (
            f"chunk_numbers:{window[0].chunk_number} -> {window[-1].chunk_number} \n\n {text_blob}"
        )
    }


__all__ = ["format_adjacent_chunks_tool_result"]
