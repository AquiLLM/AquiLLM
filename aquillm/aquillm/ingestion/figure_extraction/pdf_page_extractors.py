"""Per-page PDF figure extractors (raster + caption-rendered)."""
from __future__ import annotations

import io
import structlog
from typing import Iterator

from .pdf_geometry import (
    MAX_IMAGES_PER_DOCUMENT,
    MIN_IMAGE_BYTES,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
    MIN_RENDERED_HEIGHT,
    MIN_RENDERED_WIDTH,
    extract_nearby_text,
    find_figure_captions,
    refine_figure_region,
    render_region,
)
from .types import ExtractedFigure

logger = structlog.stdlib.get_logger(__name__)


def extract_raster_images(doc, page, page_num: int, extracted_xrefs: set) -> Iterator[ExtractedFigure]:
    """Extract embedded raster images from a page."""
    image_list = page.get_images(full=True)

    for img_info in image_list:
        try:
            xref = img_info[0]
            if xref in extracted_xrefs:
                continue

            base_image = doc.extract_image(xref)
            if not base_image:
                continue

            image_bytes = base_image.get("image")
            if not image_bytes or len(image_bytes) < MIN_IMAGE_BYTES:
                continue

            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                continue

            img_ext = base_image.get("ext", "png")
            if img_ext not in ("png", "jpeg", "jpg", "webp"):
                try:
                    from PIL import Image

                    with Image.open(io.BytesIO(image_bytes)) as img:
                        if img.mode in ("RGBA", "P"):
                            img = img.convert("RGB")
                        output = io.BytesIO()
                        img.save(output, format="PNG")
                        image_bytes = output.getvalue()
                        img_ext = "png"
                except Exception:
                    continue

            bbox = None
            try:
                for img_rect in page.get_image_rects(xref):
                    bbox = tuple(img_rect)
                    break
            except Exception:
                pass

            nearby_text = extract_nearby_text(page, bbox) if bbox else ""
            extracted_xrefs.add(xref)

            yield ExtractedFigure(
                image_bytes=image_bytes,
                image_format=img_ext,
                figure_index=-1,
                nearby_text=nearby_text,
                width=width,
                height=height,
                location_metadata={"page_number": page_num, "extraction_method": "raster"},
            )

        except Exception as exc:
            logger.debug("Failed to extract raster image from page %d: %s", page_num, exc)
            continue


def extract_rendered_figures(page, page_num: int, raster_regions: list) -> Iterator[ExtractedFigure]:
    """Extract figures by detecting captions and rendering the figure regions."""
    import fitz

    captions = find_figure_captions(page, page_num)
    if captions:
        logger.info("Found %d figure captions on page %d", len(captions), page_num)

    for caption_info in captions:
        caption_text = caption_info["caption_text"]
        figure_label = caption_info["figure_label"]
        caption_rect = caption_info["caption_rect"]
        initial_figure_rect = caption_info["figure_rect"]

        overlaps_raster = False
        for raster_rect in raster_regions:
            if initial_figure_rect.intersects(fitz.Rect(raster_rect)):
                overlaps_raster = True
                break

        if overlaps_raster:
            logger.debug("Skipping rendered figure %s - overlaps with raster image", figure_label)
            continue

        figure_rect = refine_figure_region(page, initial_figure_rect, caption_rect)
        result = render_region(page, figure_rect)
        if result is None:
            logger.debug("Failed to render figure region for %s", figure_label)
            continue

        image_bytes, width, height = result
        if width < MIN_RENDERED_WIDTH or height < MIN_RENDERED_HEIGHT:
            continue

        yield ExtractedFigure(
            image_bytes=image_bytes,
            image_format="png",
            figure_index=-1,
            nearby_text=caption_text,
            width=width,
            height=height,
            location_metadata={
                "page_number": page_num,
                "extraction_method": "rendered",
                "figure_label": figure_label,
            },
        )


__all__ = ["MAX_IMAGES_PER_DOCUMENT", "extract_raster_images", "extract_rendered_figures"]
