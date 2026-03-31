"""Figure extraction hooks for ingestion parsers (lazy-imports figure pipeline)."""
from __future__ import annotations

import structlog

from lib.parsers import get_stem as _stem

from .types import ExtractedTextPayload

logger = structlog.stdlib.get_logger(__name__)


def extract_figure_payloads_for_format(
    filename: str,
    data: bytes,
    source_format: str,
    payloads: list[ExtractedTextPayload],
) -> None:
    """Extract figures from a document and append to payloads list."""
    try:
        from aquillm.ingestion.figure_extraction import (
            extract_figures_from_document,
            generate_figure_caption,
            enhance_figure_with_ocr,
        )

        doc_title = _stem(filename)
        figure_count = 0

        for figure in extract_figures_from_document(data, source_format, filename):
            caption = generate_figure_caption(figure, doc_title, source_format)

            ocr_text, ocr_provider, ocr_model = "", "", ""
            try:
                ocr_text, ocr_provider, ocr_model = enhance_figure_with_ocr(figure)
            except Exception:
                pass

            combined_text = caption
            if ocr_text and ocr_text.strip():
                combined_text = f"{caption}\n\nText in figure: {ocr_text.strip()}"

            fig_filename = f"{doc_title}_fig{figure.figure_index}.{figure.image_format}"

            payloads.append(
                ExtractedTextPayload(
                    title=f"{doc_title} - Figure {figure.figure_index + 1}",
                    normalized_type="document_figure",
                    full_text=combined_text,
                    modality="image",
                    media_bytes=figure.image_bytes,
                    media_filename=fig_filename,
                    media_content_type=f"image/{figure.image_format}",
                    provider=ocr_provider or None,
                    model=ocr_model or None,
                    metadata={
                        "source_format": source_format,
                        "source_document_title": doc_title,
                        "figure_index": figure.figure_index,
                        "extracted_caption": figure.nearby_text,
                        "location_metadata": figure.location_metadata,
                        "width": figure.width,
                        "height": figure.height,
                    },
                )
            )
            figure_count += 1

        if figure_count > 0:
            logger.info("obs.ingest.figures_extracted", figure_count=figure_count, format=source_format.upper(), filename=filename)

    except ImportError:
        logger.debug("obs.ingest.figures_unavailable", filename=filename)
    except Exception as exc:
        logger.warning("obs.ingest.figures_error", filename=filename, error_type=type(exc).__name__, error=str(exc))


__all__ = ["extract_figure_payloads_for_format"]
