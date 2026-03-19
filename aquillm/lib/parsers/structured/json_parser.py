"""
JSON/JSONL parsing.
"""

import json

from ..text_utils import read_text_bytes


def extract_json_text(data: bytes) -> str:
    """Extract text content from JSON bytes."""
    raw = read_text_bytes(data)
    parsed = json.loads(raw)
    return json.dumps(parsed, indent=2, ensure_ascii=False)


def extract_jsonl_text(data: bytes) -> str:
    """Extract text content from JSONL bytes."""
    lines = [line.strip() for line in read_text_bytes(data).splitlines() if line.strip()]
    parts: list[str] = []
    for line in lines:
        try:
            parts.append(json.dumps(json.loads(line), ensure_ascii=False))
        except Exception:
            parts.append(line)
    return "\n".join(parts)


__all__ = ['extract_json_text', 'extract_jsonl_text']
