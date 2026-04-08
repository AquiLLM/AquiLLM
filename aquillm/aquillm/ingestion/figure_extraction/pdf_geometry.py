"""PDF page geometry: captions, regions, and raster rendering helpers."""
from __future__ import annotations

import structlog
import re

logger = structlog.stdlib.get_logger(__name__)

MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100
MIN_IMAGE_BYTES = 5_000
MAX_IMAGES_PER_DOCUMENT = 50

FIGURE_CAPTION_PATTERNS = [
    re.compile(r"(Figure|Fig\.?)\s*(\d+[a-z]?)\s*[:\.\-–—]\s*(.{10,})", re.IGNORECASE),
    re.compile(r"(Scheme|Chart|Diagram|Plot)\s*(\d+[a-z]?)\s*[:\.\-–—]\s*(.{10,})", re.IGNORECASE),
]

MIN_RENDERED_WIDTH = 150
MIN_RENDERED_HEIGHT = 100
RENDER_DPI = 150


def extract_nearby_text(page, bbox: tuple, margin: float = 50) -> str:
    """Extract text near a figure's bounding box that might be a caption."""
    import fitz

    if not bbox:
        return ""

    x0, y0, x1, y1 = bbox
    page_height = page.rect.height

    search_regions = [
        (x0 - margin, y1, x1 + margin, min(y1 + 100, page_height)),
        (x0 - margin, max(0, y0 - 80), x1 + margin, y0),
    ]

    caption_candidates = []
    for region in search_regions:
        try:
            rect = fitz.Rect(region)
            text = page.get_text("text", clip=rect).strip()
            if text:
                caption_candidates.append(text)
        except Exception:
            continue

    for candidate in caption_candidates:
        lower = candidate.lower()
        if any(marker in lower for marker in ["figure", "fig.", "fig ", "table", "diagram", "chart"]):
            lines = candidate.split("\n")
            caption_lines = []
            for line in lines[:5]:
                line = line.strip()
                if line:
                    caption_lines.append(line)
                    if line.endswith("."):
                        break
            return " ".join(caption_lines)[:500]

    for candidate in caption_candidates:
        if len(candidate) > 20:
            return candidate[:300]

    return ""


def find_figure_captions(page, page_num: int = 0) -> list[dict]:
    """Find figure captions on a page and estimate figure regions."""
    import fitz

    results = []
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    page_width = page.rect.width
    page_height = page.rect.height

    for block in blocks:
        if block.get("type") != 0:
            continue

        block_text = ""
        block_rect = fitz.Rect(block["bbox"])

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                block_text += span.get("text", "")
            block_text += "\n"

        block_text = block_text.strip()
        if not block_text:
            continue

        for pattern in FIGURE_CAPTION_PATTERNS:
            match = pattern.search(block_text)
            if match:
                label_type = match.group(1)
                label_num = match.group(2)
                caption_rest = match.group(3) if match.lastindex >= 3 else ""

                figure_label = f"{label_type} {label_num}"
                caption_text = f"{figure_label}: {caption_rest}".strip(": ")

                caption_top = block_rect.y0
                caption_left = block_rect.x0
                caption_right = block_rect.x1

                figure_bottom = caption_top - 5
                figure_top = max(0, caption_top - 400)

                margin = 20
                figure_left = max(0, caption_left - margin)
                figure_right = min(page_width, caption_right + margin)

                if (figure_right - figure_left) < MIN_RENDERED_WIDTH:
                    center = (figure_left + figure_right) / 2
                    figure_left = max(0, center - MIN_RENDERED_WIDTH / 2)
                    figure_right = min(page_width, center + MIN_RENDERED_WIDTH / 2)

                if figure_bottom > figure_top + MIN_RENDERED_HEIGHT:
                    results.append(
                        {
                            "caption_text": caption_text,
                            "figure_label": figure_label,
                            "caption_rect": block_rect,
                            "figure_rect": fitz.Rect(figure_left, figure_top, figure_right, figure_bottom),
                        }
                    )
                break

    return results


def render_region(page, rect, dpi: int = RENDER_DPI) -> tuple[bytes, int, int] | None:
    """Render a region of a PDF page to PNG bytes."""
    import fitz

    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        clip = fitz.Rect(rect)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        png_bytes = pix.tobytes("png")
        if len(png_bytes) < MIN_IMAGE_BYTES:
            return None
        return png_bytes, pix.width, pix.height
    except Exception as exc:
        logger.debug("obs.ingest.pdf_render_failed", error_type=type(exc).__name__, error=str(exc))
        return None


def refine_figure_region(page, initial_rect, caption_rect):
    """Refine the figure region using vector drawings above the caption."""
    import fitz

    page_width = page.rect.width
    drawings = page.get_drawings()
    relevant_drawings = []

    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        if rect.y1 > caption_rect.y0:
            continue
        if rect.x1 < caption_rect.x0 - 50 or rect.x0 > caption_rect.x1 + 50:
            continue
        relevant_drawings.append(rect)

    if relevant_drawings:
        min_x = min(r.x0 for r in relevant_drawings)
        min_y = min(r.y0 for r in relevant_drawings)
        max_x = max(r.x1 for r in relevant_drawings)
        max_y = max(r.y1 for r in relevant_drawings)
        padding = 10
        refined = fitz.Rect(
            max(0, min_x - padding),
            max(0, min_y - padding),
            min(page_width, max_x + padding),
            min(caption_rect.y0 - 2, max_y + padding),
        )
        if refined.width >= MIN_RENDERED_WIDTH and refined.height >= MIN_RENDERED_HEIGHT:
            return refined

    return initial_rect


__all__ = [
    "MAX_IMAGES_PER_DOCUMENT",
    "MIN_IMAGE_BYTES",
    "MIN_IMAGE_HEIGHT",
    "MIN_IMAGE_WIDTH",
    "MIN_RENDERED_HEIGHT",
    "MIN_RENDERED_WIDTH",
    "RENDER_DPI",
    "extract_nearby_text",
    "find_figure_captions",
    "refine_figure_region",
    "render_region",
]
