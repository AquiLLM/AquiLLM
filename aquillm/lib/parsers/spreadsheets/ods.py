"""
ODS (OpenDocument Spreadsheet) parsing.
"""

import io


def extract_ods_text(data: bytes) -> str:
    """Extract text content from ODS bytes."""
    try:
        from odf.opendocument import load  # type: ignore
        from odf.table import Table, TableCell, TableRow  # type: ignore
        from odf.text import P  # type: ignore
    except Exception as exc:
        raise ValueError("odfpy is required for .ods extraction.") from exc
    doc = load(io.BytesIO(data))
    lines: list[str] = []
    for table in doc.getElementsByType(Table):
        name = table.getAttribute("name") or "Sheet"
        lines.append(f"# Sheet: {name}")
        for row in table.getElementsByType(TableRow):
            row_values: list[str] = []
            for cell in row.getElementsByType(TableCell):
                text_nodes = cell.getElementsByType(P)
                value = " ".join(node.firstChild.data for node in text_nodes if getattr(node, "firstChild", None))
                value = value.strip()
                if value:
                    row_values.append(value)
            if row_values:
                lines.append(", ".join(row_values))
    return "\n".join(lines)


__all__ = ['extract_ods_text']
