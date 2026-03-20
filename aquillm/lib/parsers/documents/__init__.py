"""
Document parsers.
"""

from .pdf import extract_pdf_text
from .html import extract_html_text
from .docx import extract_docx_text
from .epub import extract_epub_text

__all__ = [
    'extract_pdf_text',
    'extract_html_text',
    'extract_docx_text',
    'extract_epub_text',
]
