"""
ODP (OpenDocument Presentation) parsing.
"""

import io


def extract_odp_text(data: bytes) -> str:
    """Extract text content from ODP bytes."""
    try:
        from odf.opendocument import load  # type: ignore
        from odf.draw import Page  # type: ignore
        from odf.text import P  # type: ignore
    except Exception as exc:
        raise ValueError("odfpy is required for .odp extraction.") from exc
    doc = load(io.BytesIO(data))
    lines: list[str] = []
    for page in doc.getElementsByType(Page):
        lines.append(f"# Slide: {page.getAttribute('name') or 'unnamed'}")
        for paragraph in page.getElementsByType(P):
            text = ""
            if getattr(paragraph, "firstChild", None):
                text = str(paragraph.firstChild.data).strip()
            if text:
                lines.append(text)
    return "\n".join(lines)


__all__ = ['extract_odp_text']
