"""
EPUB document parsing.
"""

import io

from bs4 import BeautifulSoup


def extract_epub_text(data: bytes) -> str:
    """Extract text content from EPUB bytes."""
    try:
        from ebooklib import epub  # type: ignore
    except Exception as exc:
        raise ValueError("ebooklib is required for .epub extraction.") from exc
    book = epub.read_epub(io.BytesIO(data))
    text_parts: list[str] = []
    for item in book.get_items():
        content = getattr(item, "get_content", None)
        if not callable(content):
            continue
        raw = item.get_content()
        if not isinstance(raw, (bytes, bytearray)):
            continue
        text_parts.append(BeautifulSoup(raw.decode("utf-8", errors="ignore"), "html.parser").get_text("\n", strip=True))
    return "\n\n".join(part for part in text_parts if part)


__all__ = ['extract_epub_text']
