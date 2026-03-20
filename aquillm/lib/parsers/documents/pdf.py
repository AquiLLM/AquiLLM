"""
PDF document parsing.
"""

import io

from pypdf import PdfReader


def extract_pdf_text(data: bytes) -> str:
    """Extract text content from PDF bytes."""
    reader = PdfReader(io.BytesIO(data))
    text_parts = []
    for page in reader.pages:
        text_parts.append((page.extract_text() or "").strip())
    return "\n\n".join(part for part in text_parts if part)


__all__ = ['extract_pdf_text']
