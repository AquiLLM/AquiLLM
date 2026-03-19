"""
CSV/TSV parsing.
"""

import csv
import io

from ..text_utils import read_text_bytes


def extract_csv_text(data: bytes, delimiter: str = ",") -> str:
    """Extract text content from CSV/TSV bytes."""
    text = read_text_bytes(data)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [", ".join(cell.strip() for cell in row) for row in reader]
    return "\n".join(rows)


__all__ = ['extract_csv_text']
