"""
Excel XLS spreadsheet parsing.
"""


def extract_xls_text(data: bytes) -> str:
    """Extract text content from XLS bytes."""
    try:
        import xlrd  # type: ignore
    except Exception as exc:
        raise ValueError("xlrd is required for .xls extraction.") from exc
    book = xlrd.open_workbook(file_contents=data)
    lines: list[str] = []
    for sheet in book.sheets():
        lines.append(f"# Sheet: {sheet.name}")
        for row_index in range(sheet.nrows):
            values = [str(value).strip() for value in sheet.row_values(row_index) if str(value).strip()]
            if values:
                lines.append(", ".join(values))
    return "\n".join(lines)


__all__ = ['extract_xls_text']
