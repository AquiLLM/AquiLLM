"""
Presentation parsers.
"""

from .pptx import extract_pptx_text
from .odp import extract_odp_text

__all__ = [
    'extract_pptx_text',
    'extract_odp_text',
]
