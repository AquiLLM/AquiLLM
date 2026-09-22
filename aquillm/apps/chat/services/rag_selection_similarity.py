"""Local, deterministic snippet similarity for soft evidence redundancy."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from apps.chat.services.rag_selection_types import SelectionCandidate

_WORDS = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_WORDS.findall(unicodedata.normalize("NFKC", text).casefold()))


@dataclass(frozen=True, slots=True)
class SnippetFeatures:
    tokens: tuple[str, ...]
    shingles: frozenset[tuple[str, str, str]]


def prepare_snippet(text: str) -> SnippetFeatures:
    """Keep features within this selection call, never in a shared text cache."""
    tokens = _tokens(text) if isinstance(text, str) else ()
    return SnippetFeatures(tokens, frozenset(zip(tokens, tokens[1:], tokens[2:])))


def prepared_redundancy(
    left: SnippetFeatures,
    right: SnippetFeatures,
    *,
    same_document: bool,
) -> float:
    left_tokens, right_tokens = left.tokens, right.tokens
    if not left_tokens or not right_tokens:
        return 0.0
    if len(left_tokens) < 3 or len(right_tokens) < 3:
        similarity = float(left_tokens == right_tokens)
    else:
        left_shingles, right_shingles = left.shingles, right.shingles
        similarity = len(left_shingles & right_shingles) / len(
            left_shingles | right_shingles
        )
    return similarity * (1.0 if same_document else 0.5)


def snippet_redundancy(left: SelectionCandidate, right: SelectionCandidate) -> float:
    """Three-shingle Jaccard, halved for independent document sources."""
    return prepared_redundancy(
        prepare_snippet(left.text),
        prepare_snippet(right.text),
        same_document=left.doc_id == right.doc_id,
    )
