"""
Spreadsheet parsers.
"""

from .xlsx import extract_xlsx_text
from .xls import extract_xls_text
from .ods import extract_ods_text
from .csv_parser import extract_csv_text

__all__ = [
    'extract_xlsx_text',
    'extract_xls_text',
    'extract_ods_text',
    'extract_csv_text',
]
