"""
YAML parsing.
"""

import json

from ..text_utils import read_text_bytes


def extract_yaml_text(data: bytes) -> str:
    """Extract text content from YAML bytes."""
    raw = read_text_bytes(data)
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(raw)
        if parsed is None:
            return ""
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception:
        return raw


__all__ = ['extract_yaml_text']
