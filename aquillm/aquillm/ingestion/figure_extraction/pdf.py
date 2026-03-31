"""PDF figure extraction using PyMuPDF (fitz).

Supports two extraction methods:
1. Raster image extraction - extracts embedded bitmap images
2. Caption-based rendering - detects figure captions and renders the figure region
"""

import structlog
from typing import Iterator

from .pdf_page_extractors import (
    MAX_IMAGES_PER_DOCUMENT,
    extract_raster_images,
    extract_rendered_figures,
)
from .types import ExtractedFigure

logger = structlog.stdlib.get_logger(__name__)


def extract_figures(data: bytes, filename: str = "") -> Iterator[ExtractedFigure]:
    """
    Extract figures from a PDF using multiple methods.

    1. First extracts embedded raster images (photos, scans, etc.)
    2. Then detects figure captions and renders vector figure regions
    """
    try:
        import fitz
    except ImportError:
        logger.warning("obs.ingest.pdf_dependency_missing", dependency="PyMuPDF")
        return

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        logger.warning("obs.ingest.pdf_open_failed", error_type=type(exc).__name__, error=str(exc))
        return

    logger.info("obs.ingest.figures_pdf_start", filename=filename, page_count=len(doc))
    total_extracted = 0
    extracted_xrefs: set[int] = set()

    try:
        for page_num, page in enumerate(doc, start=1):
            if total_extracted >= MAX_IMAGES_PER_DOCUMENT:
                logger.info("obs.ingest.figures_pdf_limit", filename=filename, max_limit=MAX_IMAGES_PER_DOCUMENT)
                break

            raster_regions: list[tuple] = []

            for figure in extract_raster_images(doc, page, page_num, extracted_xrefs):
                if total_extracted >= MAX_IMAGES_PER_DOCUMENT:
                    break
                try:
                    for xref_info in page.get_images(full=True):
                        xref = xref_info[0]
                        for img_rect in page.get_image_rects(xref):
                            raster_regions.append(tuple(img_rect))
                            break
                except Exception:
                    pass

                figure.figure_index = total_extracted
                yield figure
                total_extracted += 1

            for figure in extract_rendered_figures(page, page_num, raster_regions):
                if total_extracted >= MAX_IMAGES_PER_DOCUMENT:
                    break
                figure.figure_index = total_extracted
                yield figure
                total_extracted += 1

    finally:
        doc.close()

    if total_extracted > 0:
        logger.info("obs.ingest.figures_pdf_done", filename=filename, figure_count=total_extracted)
    else:
        logger.info("obs.ingest.figures_pdf_none", filename=filename)


__all__ = ["extract_figures"]
