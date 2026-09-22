"""Local, deterministic snippet similarity for soft evidence redundancy."""

from __future__ import annotations

import re
import unicodedata

from apps.chat.services.rag_selection_types import SelectionCandidate

_WORDS = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_WORDS.findall(unicodedata.normalize("NFKC", text).casefold()))


def snippet_redundancy(left: SelectionCandidate, right: SelectionCandidate) -> float:
    """Three-shingle Jaccard, halved for independent document sources."""
    if not isinstance(left.text, str) or not isinstance(right.text, str):
        return 0.0
    left_tokens, right_tokens = _tokens(left.text), _tokens(right.text)
    if not left_tokens or not right_tokens:
        return 0.0
    if len(left_tokens) < 3 or len(right_tokens) < 3:
        similarity = float(left_tokens == right_tokens)
    else:
        left_shingles = set(zip(left_tokens, left_tokens[1:], left_tokens[2:]))
        right_shingles = set(zip(right_tokens, right_tokens[1:], right_tokens[2:]))
        similarity = len(left_shingles & right_shingles) / len(
            left_shingles | right_shingles
        )
    return similarity * (0.5 if left.doc_id != right.doc_id else 1.0)
