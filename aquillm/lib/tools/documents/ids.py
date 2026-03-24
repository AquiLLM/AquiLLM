"""Pure helpers for parsing document IDs from LLM tool arguments."""
from __future__ import annotations

import re
from collections.abc import Sequence
from uuid import UUID

_UUID_HEX_LEN = 32
# Shorter fragments are too easy to collide when many documents are visible.
_MIN_PREFIX_MATCH_HEX_LEN = 16


def clean_and_parse_doc_id(doc_id: str) -> tuple[UUID | None, str]:
    """
    Attempt to parse a document ID, handling common LLM errors like:
    - Truncated UUIDs
    - XML tags mixed in (e.g., "uuid</doc_id>")
    - Extra whitespace

    Returns (parsed_uuid, error_message). If parsing succeeds, error_message is empty.
    """
    original = doc_id
    cleaned = doc_id.strip()

    cleaned = re.sub(r"</?\w+>", "", cleaned)
    cleaned = cleaned.strip()

    cleaned = re.sub(r"[^0-9a-fA-F-]", "", cleaned)

    try:
        return UUID(cleaned), ""
    except (ValueError, AttributeError):
        pass

    hex_only = cleaned.replace("-", "")
    if len(hex_only) >= 28:
        try:
            if len(hex_only) < 32:
                hex_only = hex_only + "0" * (32 - len(hex_only))
            formatted = (
                f"{hex_only[:8]}-{hex_only[8:12]}-{hex_only[12:16]}-"
                f"{hex_only[16:20]}-{hex_only[20:32]}"
            )
            return UUID(formatted), ""
        except (ValueError, AttributeError):
            pass

    return None, (
        f"Invalid document ID: '{original}'. Expected a valid UUID (36 characters with hyphens, "
        "e.g., '12345678-1234-5678-1234-567812345678')."
    )


def _hex_fragment_from_doc_id_arg(doc_id: str) -> str:
    cleaned = doc_id.strip()
    cleaned = re.sub(r"</?\w+>", "", cleaned)
    return re.sub(r"[^0-9a-fA-F]", "", cleaned).lower()


def resolve_doc_id_with_candidates(doc_id: str, candidates: Sequence[UUID]) -> tuple[UUID | None, str]:
    """
    Parse doc_id; if that fails, try a unique prefix match against candidate document UUIDs.

    Used when models truncate UUIDs while copying from long document lists.
    """
    parsed, err = clean_and_parse_doc_id(doc_id)
    if parsed is not None:
        return parsed, ""

    fragment = _hex_fragment_from_doc_id_arg(doc_id)
    if len(fragment) < _MIN_PREFIX_MATCH_HEX_LEN:
        return None, err
    if len(fragment) > _UUID_HEX_LEN:
        fragment = fragment[:_UUID_HEX_LEN]

    matches: list[UUID] = []
    for u in candidates:
        uh = str(u).replace("-", "").lower()
        if uh.startswith(fragment):
            matches.append(u)

    if len(matches) == 1:
        return matches[0], ""
    if not matches:
        return None, err
    return None, (
        f"Ambiguous document ID (matches {len(matches)} documents in the selected collections). "
        "Copy the full UUID from document_ids, or use vector_search to search all documents at once."
    )


__all__ = ["clean_and_parse_doc_id", "resolve_doc_id_with_candidates"]
