"""
Text processing utilities for parsers.
"""


def read_text_bytes(data: bytes) -> str:
    """Decode bytes to text with encoding detection."""
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")


__all__ = ['read_text_bytes']
