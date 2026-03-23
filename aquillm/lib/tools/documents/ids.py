"""Pure helpers for parsing document IDs from LLM tool arguments."""
from __future__ import annotations

import re
from uuid import UUID


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


__all__ = ["clean_and_parse_doc_id"]
