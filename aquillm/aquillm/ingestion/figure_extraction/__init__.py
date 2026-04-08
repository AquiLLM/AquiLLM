"""
Document figure extraction for all supported formats.

Usage:
    from aquillm.ingestion.figure_extraction import extract_figures_from_document

    for figure in extract_figures_from_document(data, "pdf", "document.pdf"):
        print(figure.width, figure.height, figure.location_metadata)
"""

import structlog
from typing import Iterator

from .types import ExtractedFigure

logger = structlog.stdlib.get_logger(__name__)

__all__ = ["extract_figures_from_document", "ExtractedFigure"]


def extract_figures_from_document(
    data: bytes,
    source_format: str,
    filename: str = "",
) -> Iterator[ExtractedFigure]:
    """
    Extract figures from a document based on its format.

    Args:
        data: Raw document bytes
        source_format: Format identifier ('pdf', 'docx', 'pptx', 'xlsx', 'ods', 'epub')
        filename: Optional filename for logging

    Yields:
        ExtractedFigure for each valid image found
    """
    source_format = source_format.lower().strip()

    if source_format == "pdf":
        from .pdf import extract_figures
        yield from extract_figures(data, filename)

    elif source_format == "docx":
        from .office import extract_figures_docx
        yield from extract_figures_docx(data, filename)

    elif source_format == "pptx":
        from .office import extract_figures_pptx
        yield from extract_figures_pptx(data, filename)

    elif source_format == "xlsx":
        from .spreadsheet import extract_figures_xlsx
        yield from extract_figures_xlsx(data, filename)

    elif source_format == "ods":
        from .spreadsheet import extract_figures_ods
        yield from extract_figures_ods(data, filename)

    elif source_format == "epub":
        from .ebook import extract_figures_epub
        yield from extract_figures_epub(data, filename)

    else:
        logger.debug("obs.ingest.figures_format_unsupported", format=source_format)


def generate_figure_caption(figure: ExtractedFigure, doc_title: str, source_format: str) -> str:
    """
    Generate a caption/description for a figure.

    Combines extracted nearby text with location context.
    """
    parts = []

    if figure.nearby_text:
        parts.append(figure.nearby_text)

    location = figure.location_metadata
    if "page_number" in location:
        parts.append(f"(Page {location['page_number']})")
    elif "slide_number" in location:
        slide_info = f"Slide {location['slide_number']}"
        if location.get('slide_title'):
            slide_info += f": {location['slide_title']}"
        parts.append(f"({slide_info})")
    elif "sheet_name" in location:
        parts.append(f"(Sheet: {location['sheet_name']})")
    elif "chapter" in location:
        parts.append(f"(Chapter: {location['chapter']})")

    if not parts:
        parts.append(f"Figure from {source_format.upper()}")

    parts.append(f"[Source: {doc_title}]")

    return " ".join(parts)


def enhance_figure_with_ocr(figure: ExtractedFigure) -> tuple[str, str, str]:
    """
    Run OCR on a figure to extract embedded text.

    Returns:
        Tuple of (ocr_text, provider, model)
    """
    try:
        import io as _io
        from aquillm.ocr_utils import extract_text_from_image

        result = extract_text_from_image(_io.BytesIO(figure.image_bytes))

        ocr_text = result.get("extracted_text", "")
        provider = result.get("provider", "")
        model = result.get("model", "")

        return ocr_text, provider, model
    except Exception as exc:
        logger.debug("obs.ingest.figures_ocr_error", error_type=type(exc).__name__, error=str(exc))
        return "", "", ""
