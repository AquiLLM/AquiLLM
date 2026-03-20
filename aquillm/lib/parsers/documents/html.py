"""
HTML document parsing.
"""

from bs4 import BeautifulSoup

from ..text_utils import read_text_bytes


def extract_html_text(data: bytes) -> str:
    """Extract text content from HTML bytes."""
    raw = read_text_bytes(data)
    return BeautifulSoup(raw, "html.parser").get_text("\n", strip=True)


__all__ = ['extract_html_text']
