"""PDF figure extraction using PyMuPDF (fitz).

Supports two extraction methods:
1. Raster image extraction - extracts embedded bitmap images
2. Caption-based rendering - detects figure captions and renders the figure region
"""

import io
import logging
import re
from typing import Iterator

from .types import ExtractedFigure

logger = logging.getLogger(__name__)

MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100
MIN_IMAGE_BYTES = 5_000
MAX_IMAGES_PER_DOCUMENT = 50

# Patterns for detecting figure captions - search anywhere in text
FIGURE_CAPTION_PATTERNS = [
    re.compile(r'(Figure|Fig\.?)\s*(\d+[a-z]?)\s*[:\.\-–—]\s*(.{10,})', re.IGNORECASE),
    re.compile(r'(Scheme|Chart|Diagram|Plot)\s*(\d+[a-z]?)\s*[:\.\-–—]\s*(.{10,})', re.IGNORECASE),
]

# Minimum rendered figure dimensions
MIN_RENDERED_WIDTH = 150
MIN_RENDERED_HEIGHT = 100
RENDER_DPI = 150  # Balance between quality and file size


def _extract_nearby_text(page, bbox: tuple, margin: float = 50) -> str:
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
        if any(marker in lower for marker in ['figure', 'fig.', 'fig ', 'table', 'diagram', 'chart']):
            lines = candidate.split('\n')
            caption_lines = []
            for line in lines[:5]:
                line = line.strip()
                if line:
                    caption_lines.append(line)
                    if line.endswith('.'):
                        break
            return ' '.join(caption_lines)[:500]
    
    for candidate in caption_candidates:
        if len(candidate) > 20:
            return candidate[:300]
    
    return ""


def _find_figure_captions(page, page_num: int = 0) -> list[dict]:
    """
    Find figure captions on a page and estimate figure regions.
    
    Returns list of dicts with:
        - caption_text: The full caption text
        - figure_label: e.g., "Figure 1", "Fig. 2"
        - caption_rect: Bounding box of caption text
        - figure_rect: Estimated bounding box of the figure (above caption)
    """
    import fitz
    
    results = []
    
    # Get text blocks with position info
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    
    page_width = page.rect.width
    page_height = page.rect.height
    
    for block in blocks:
        if block.get("type") != 0:  # Text block
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
        
        # Search for figure caption pattern anywhere in the block
        for pattern in FIGURE_CAPTION_PATTERNS:
            match = pattern.search(block_text)
            if match:
                label_type = match.group(1)
                label_num = match.group(2)
                caption_rest = match.group(3) if match.lastindex >= 3 else ""
                
                figure_label = f"{label_type} {label_num}"
                caption_text = f"{figure_label}: {caption_rest}".strip(": ")
                
                # Estimate figure region: area above the caption
                # Figures are typically above their captions
                caption_top = block_rect.y0
                caption_bottom = block_rect.y1
                caption_left = block_rect.x0
                caption_right = block_rect.x1
                
                # Look for figure above caption
                # Estimate figure height based on typical academic paper layout
                figure_bottom = caption_top - 5  # Small gap
                figure_top = max(0, caption_top - 400)  # Up to ~400 points above
                
                # Expand width slightly beyond caption
                margin = 20
                figure_left = max(0, caption_left - margin)
                figure_right = min(page_width, caption_right + margin)
                
                # Ensure minimum size
                if (figure_right - figure_left) < MIN_RENDERED_WIDTH:
                    center = (figure_left + figure_right) / 2
                    figure_left = max(0, center - MIN_RENDERED_WIDTH / 2)
                    figure_right = min(page_width, center + MIN_RENDERED_WIDTH / 2)
                
                if figure_bottom > figure_top + MIN_RENDERED_HEIGHT:
                    results.append({
                        "caption_text": caption_text,
                        "figure_label": figure_label,
                        "caption_rect": block_rect,
                        "figure_rect": fitz.Rect(figure_left, figure_top, figure_right, figure_bottom),
                    })
                break
    
    return results


def _render_region(page, rect, dpi: int = RENDER_DPI) -> tuple[bytes, int, int] | None:
    """
    Render a region of a PDF page to PNG bytes.
    
    Returns (image_bytes, width, height) or None if failed.
    """
    import fitz
    
    try:
        # Calculate zoom factor for desired DPI (72 is PDF default)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        # Render the clipped region
        clip = fitz.Rect(rect)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        
        # Convert to PNG
        png_bytes = pix.tobytes("png")
        
        if len(png_bytes) < MIN_IMAGE_BYTES:
            return None
        
        return png_bytes, pix.width, pix.height
        
    except Exception as exc:
        logger.debug("Failed to render PDF region: %s", exc)
        return None


def _refine_figure_region(page, initial_rect, caption_rect) -> "fitz.Rect":
    """
    Refine the figure region by analyzing the page content.
    
    Looks for drawing commands, images, and whitespace to better bound the figure.
    """
    import fitz
    
    page_width = page.rect.width
    page_height = page.rect.height
    
    # Get all drawings on the page
    drawings = page.get_drawings()
    
    # Find drawings that are above the caption and overlap horizontally
    relevant_drawings = []
    caption_center_x = (caption_rect.x0 + caption_rect.x1) / 2
    
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        
        # Drawing should be above caption
        if rect.y1 > caption_rect.y0:
            continue
        
        # Drawing should overlap horizontally with caption area
        if rect.x1 < caption_rect.x0 - 50 or rect.x0 > caption_rect.x1 + 50:
            continue
        
        relevant_drawings.append(rect)
    
    if relevant_drawings:
        # Find the bounding box of all relevant drawings
        min_x = min(r.x0 for r in relevant_drawings)
        min_y = min(r.y0 for r in relevant_drawings)
        max_x = max(r.x1 for r in relevant_drawings)
        max_y = max(r.y1 for r in relevant_drawings)
        
        # Add padding
        padding = 10
        refined = fitz.Rect(
            max(0, min_x - padding),
            max(0, min_y - padding),
            min(page_width, max_x + padding),
            min(caption_rect.y0 - 2, max_y + padding)
        )
        
        # Only use refined rect if it's reasonable
        if refined.width >= MIN_RENDERED_WIDTH and refined.height >= MIN_RENDERED_HEIGHT:
            return refined
    
    # Fall back to initial estimate
    return initial_rect


def _extract_raster_images(doc, page, page_num: int, extracted_xrefs: set) -> Iterator[ExtractedFigure]:
    """Extract embedded raster images from a page."""
    image_list = page.get_images(full=True)
    
    for img_info in image_list:
        try:
            xref = img_info[0]
            
            # Skip if already extracted
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
            
            nearby_text = _extract_nearby_text(page, bbox) if bbox else ""
            extracted_xrefs.add(xref)
            
            yield ExtractedFigure(
                image_bytes=image_bytes,
                image_format=img_ext,
                figure_index=-1,  # Will be assigned later
                nearby_text=nearby_text,
                width=width,
                height=height,
                location_metadata={"page_number": page_num, "extraction_method": "raster"},
            )
            
        except Exception as exc:
            logger.debug("Failed to extract raster image from page %d: %s", page_num, exc)
            continue


def _extract_rendered_figures(page, page_num: int, raster_regions: list) -> Iterator[ExtractedFigure]:
    """
    Extract figures by detecting captions and rendering the figure regions.
    
    Args:
        page: PyMuPDF page object
        page_num: Page number for metadata
        raster_regions: List of rects where raster images were found (to avoid duplicates)
    """
    import fitz
    
    captions = _find_figure_captions(page, page_num)
    
    if captions:
        logger.info("Found %d figure captions on page %d", len(captions), page_num)
    
    for caption_info in captions:
        caption_text = caption_info["caption_text"]
        figure_label = caption_info["figure_label"]
        caption_rect = caption_info["caption_rect"]
        initial_figure_rect = caption_info["figure_rect"]
        
        # Check if this region overlaps with an already-extracted raster image
        overlaps_raster = False
        for raster_rect in raster_regions:
            if initial_figure_rect.intersects(fitz.Rect(raster_rect)):
                overlaps_raster = True
                break
        
        if overlaps_raster:
            logger.debug("Skipping rendered figure %s - overlaps with raster image", figure_label)
            continue
        
        # Refine the figure region based on page content
        figure_rect = _refine_figure_region(page, initial_figure_rect, caption_rect)
        
        # Render the figure region
        result = _render_region(page, figure_rect)
        if result is None:
            logger.debug("Failed to render figure region for %s", figure_label)
            continue
        
        image_bytes, width, height = result
        
        # Skip if too small after rendering
        if width < MIN_RENDERED_WIDTH or height < MIN_RENDERED_HEIGHT:
            continue
        
        yield ExtractedFigure(
            image_bytes=image_bytes,
            image_format="png",
            figure_index=-1,  # Will be assigned later
            nearby_text=caption_text,
            width=width,
            height=height,
            location_metadata={
                "page_number": page_num,
                "extraction_method": "rendered",
                "figure_label": figure_label,
            },
        )


def extract_figures(data: bytes, filename: str = "") -> Iterator[ExtractedFigure]:
    """
    Extract figures from a PDF using multiple methods.
    
    1. First extracts embedded raster images (photos, scans, etc.)
    2. Then detects figure captions and renders vector figure regions
    
    Args:
        data: Raw PDF bytes
        filename: Optional filename for logging
        
    Yields:
        ExtractedFigure for each valid figure found
    """
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF not installed; skipping PDF figure extraction")
        return
    
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        logger.warning("Failed to open PDF for figure extraction: %s", exc)
        return
    
    logger.info("Starting figure extraction for PDF %s (%d pages)", filename, len(doc))
    total_extracted = 0
    extracted_xrefs: set[int] = set()  # Track extracted raster image xrefs
    
    try:
        for page_num, page in enumerate(doc, start=1):
            if total_extracted >= MAX_IMAGES_PER_DOCUMENT:
                logger.info("Reached max image limit (%d) for PDF", MAX_IMAGES_PER_DOCUMENT)
                break
            
            # Track raster image regions to avoid duplicate extraction
            raster_regions: list[tuple] = []
            
            # Phase 1: Extract embedded raster images
            for figure in _extract_raster_images(doc, page, page_num, extracted_xrefs):
                if total_extracted >= MAX_IMAGES_PER_DOCUMENT:
                    break
                
                # Track the region if we have bbox
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
            
            # Phase 2: Extract rendered figures from caption detection
            for figure in _extract_rendered_figures(page, page_num, raster_regions):
                if total_extracted >= MAX_IMAGES_PER_DOCUMENT:
                    break
                
                figure.figure_index = total_extracted
                yield figure
                total_extracted += 1
                
    finally:
        doc.close()
    
    if total_extracted > 0:
        logger.info("Extracted %d figures from PDF %s", total_extracted, filename)
    else:
        logger.info("No figures extracted from PDF %s", filename)
