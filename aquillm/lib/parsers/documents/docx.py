"""
DOCX document parsing.
"""

import io


def extract_docx_text(data: bytes) -> str:
    """Extract text content from DOCX bytes."""
    try:
        from docx import Document  # type: ignore
    except Exception as exc:
        raise ValueError("python-docx is required for .docx extraction.") from exc
    doc = Document(io.BytesIO(data))
    return "\n".join(paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip())


__all__ = ['extract_docx_text']
