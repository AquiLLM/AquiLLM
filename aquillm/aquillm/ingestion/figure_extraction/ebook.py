"""EPUB figure extraction."""

import io
import logging
from typing import Iterator

from .types import ExtractedFigure

logger = logging.getLogger(__name__)

MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100
MIN_IMAGE_BYTES = 5_000
MAX_IMAGES_PER_DOCUMENT = 50


def _get_image_dimensions(image_bytes: bytes) -> tuple[int, int]:
    """Get image dimensions using PIL."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as img:
            return img.width, img.height
    except Exception:
        return 0, 0


def _normalize_image(image_bytes: bytes, media_type: str) -> tuple[bytes, str]:
    """Normalize image to common format."""
    ext = "png"
    if "jpeg" in media_type or "jpg" in media_type:
        ext = "jpeg"
    elif "png" in media_type:
        ext = "png"
    elif "webp" in media_type:
        ext = "webp"
    elif "gif" in media_type:
        try:
            from PIL import Image
            with Image.open(io.BytesIO(image_bytes)) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                output = io.BytesIO()
                img.save(output, format="PNG")
                return output.getvalue(), "png"
        except Exception:
            pass
    return image_bytes, ext


def extract_figures_epub(data: bytes, filename: str = "") -> Iterator[ExtractedFigure]:
    """
    Extract figures from an EPUB file.
    
    Args:
        data: Raw EPUB bytes
        filename: Optional filename for logging
        
    Yields:
        ExtractedFigure for each valid image found
    """
    try:
        from ebooklib import epub, ITEM_IMAGE
    except ImportError:
        logger.warning("ebooklib not installed; skipping EPUB figure extraction")
        return
    
    try:
        book = epub.read_epub(io.BytesIO(data))
    except Exception as exc:
        logger.warning("Failed to open EPUB for figure extraction: %s", exc)
        return
    
    total_extracted = 0
    
    try:
        for item in book.get_items():
            if total_extracted >= MAX_IMAGES_PER_DOCUMENT:
                break
            
            if item.get_type() != ITEM_IMAGE:
                continue
            
            try:
                image_bytes = item.get_content()
                
                if not image_bytes or len(image_bytes) < MIN_IMAGE_BYTES:
                    continue
                
                width, height = _get_image_dimensions(image_bytes)
                if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                    continue
                
                media_type = item.media_type or 'image/png'
                image_bytes, img_format = _normalize_image(image_bytes, media_type)
                
                item_name = item.get_name() or ""
                
                yield ExtractedFigure(
                    image_bytes=image_bytes,
                    image_format=img_format,
                    figure_index=total_extracted,
                    nearby_text="",
                    width=width,
                    height=height,
                    location_metadata={"item_name": item_name},
                )
                
                total_extracted += 1
                
            except Exception as exc:
                logger.debug("Failed to extract EPUB image: %s", exc)
                continue
                
    except Exception as exc:
        logger.warning("EPUB figure extraction failed: %s", exc)
    
    if total_extracted > 0:
        logger.info("Extracted %d figures from EPUB %s", total_extracted, filename)
