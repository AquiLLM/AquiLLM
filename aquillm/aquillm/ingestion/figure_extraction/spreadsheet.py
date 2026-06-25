"""XLSX and ODS figure extraction.

Supports two extraction methods:
1. PDF conversion + extraction - converts to PDF using LibreOffice, then extracts figures
2. Direct raster extraction - extracts embedded images directly (fallback)
"""

import io
import structlog
import zipfile
from typing import Iterator

from .types import ExtractedFigure

logger = structlog.stdlib.get_logger(__name__)

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


def _normalize_image(image_bytes: bytes) -> tuple[bytes, str]:
    """Normalize image to PNG if needed."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as img:
            fmt = img.format.lower() if img.format else "png"
            if fmt in ("png", "jpeg", "jpg", "webp"):
                return image_bytes, fmt if fmt != "jpg" else "jpeg"
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            output = io.BytesIO()
            img.save(output, format="PNG")
            return output.getvalue(), "png"
    except Exception:
        return image_bytes, "png"


def _extract_via_pdf(data: bytes, source_format: str, filename: str) -> Iterator[ExtractedFigure] | None:
    """
    Try to extract figures by converting to PDF first.
    
    Returns None if conversion is not available or fails.
    """
    try:
        from .office_convert import convert_to_pdf, is_libreoffice_available
        from .pdf import extract_figures as extract_pdf_figures
    except ImportError:
        return None
    
    if not is_libreoffice_available():
        logger.debug("LibreOffice not available for %s conversion", source_format)
        return None
    
    pdf_bytes = convert_to_pdf(data, source_format, filename)
    if not pdf_bytes:
        logger.debug("PDF conversion failed for %s", filename or source_format)
        return None
    
    logger.info("Using PDF conversion for %s figure extraction: %s", source_format.upper(), filename)
    
    # Update location metadata to indicate conversion was used
    for figure in extract_pdf_figures(pdf_bytes, filename):
        figure.location_metadata["converted_from"] = source_format
        yield figure


def _extract_xlsx_direct(data: bytes, filename: str = "") -> Iterator[ExtractedFigure]:
    """Extract embedded raster images directly from XLSX."""
    try:
        from openpyxl import load_workbook
    except ImportError:
        logger.warning("openpyxl not installed; skipping XLSX figure extraction")
        return
    
    try:
        workbook = load_workbook(io.BytesIO(data), read_only=False)
    except Exception as exc:
        logger.warning("Failed to open XLSX for figure extraction: %s", exc)
        return
    
    total_extracted = 0
    
    try:
        for sheet in workbook.worksheets:
            if total_extracted >= MAX_IMAGES_PER_DOCUMENT:
                break
            
            sheet_name = sheet.title or "Sheet"
            
            if not hasattr(sheet, '_images'):
                continue
            
            for image in sheet._images:
                if total_extracted >= MAX_IMAGES_PER_DOCUMENT:
                    break
                
                try:
                    image_bytes = image._data()
                    
                    if not image_bytes or len(image_bytes) < MIN_IMAGE_BYTES:
                        continue
                    
                    width, height = _get_image_dimensions(image_bytes)
                    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                        continue
                    
                    image_bytes, img_format = _normalize_image(image_bytes)
                    
                    yield ExtractedFigure(
                        image_bytes=image_bytes,
                        image_format=img_format,
                        figure_index=total_extracted,
                        nearby_text=f"Image from sheet: {sheet_name}",
                        width=width,
                        height=height,
                        location_metadata={
                            "sheet_name": sheet_name,
                            "extraction_method": "direct",
                        },
                    )
                    
                    total_extracted += 1
                    
                except Exception as exc:
                    logger.debug("Failed to extract XLSX image: %s", exc)
                    continue
                    
    except Exception as exc:
        logger.warning("XLSX figure extraction failed: %s", exc)
    
    if total_extracted > 0:
        logger.info("Extracted %d embedded figures from XLSX %s", total_extracted, filename)


def _extract_ods_direct(data: bytes, filename: str = "") -> Iterator[ExtractedFigure]:
    """Extract embedded raster images directly from ODS."""
    total_extracted = 0
    
    try:
        with zipfile.ZipFile(io.BytesIO(data), 'r') as zf:
            for zip_info in zf.infolist():
                if total_extracted >= MAX_IMAGES_PER_DOCUMENT:
                    break
                
                if not zip_info.filename.startswith('Pictures/'):
                    continue
                
                if zip_info.is_dir():
                    continue
                
                try:
                    image_bytes = zf.read(zip_info.filename)
                    
                    if not image_bytes or len(image_bytes) < MIN_IMAGE_BYTES:
                        continue
                    
                    width, height = _get_image_dimensions(image_bytes)
                    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                        continue
                    
                    image_bytes, img_format = _normalize_image(image_bytes)
                    
                    yield ExtractedFigure(
                        image_bytes=image_bytes,
                        image_format=img_format,
                        figure_index=total_extracted,
                        nearby_text="",
                        width=width,
                        height=height,
                        location_metadata={
                            "source_path": zip_info.filename,
                            "extraction_method": "direct",
                        },
                    )
                    
                    total_extracted += 1
                    
                except Exception as exc:
                    logger.debug("Failed to extract ODS image %s: %s", zip_info.filename, exc)
                    continue
                    
    except Exception as exc:
        logger.warning("ODS figure extraction failed: %s", exc)
    
    if total_extracted > 0:
        logger.info("Extracted %d embedded figures from ODS %s", total_extracted, filename)


def extract_figures_xlsx(data: bytes, filename: str = "") -> Iterator[ExtractedFigure]:
    """
    Extract figures from an XLSX file.
    
    First tries PDF conversion (to capture charts and other vector graphics),
    falls back to direct extraction if LibreOffice is not available.
    
    Args:
        data: Raw XLSX bytes
        filename: Optional filename for logging
        
    Yields:
        ExtractedFigure for each valid image found
    """
    # Try PDF conversion first (captures charts)
    pdf_result = _extract_via_pdf(data, "xlsx", filename)
    if pdf_result is not None:
        yield from pdf_result
        return
    
    # Fall back to direct extraction
    logger.debug("Falling back to direct XLSX extraction for %s", filename)
    yield from _extract_xlsx_direct(data, filename)


def extract_figures_ods(data: bytes, filename: str = "") -> Iterator[ExtractedFigure]:
    """
    Extract figures from an ODS file.
    
    First tries PDF conversion (to capture charts and other vector graphics),
    falls back to direct extraction if LibreOffice is not available.
    
    Args:
        data: Raw ODS bytes
        filename: Optional filename for logging
        
    Yields:
        ExtractedFigure for each valid image found
    """
    # Try PDF conversion first (captures charts)
    pdf_result = _extract_via_pdf(data, "ods", filename)
    if pdf_result is not None:
        yield from pdf_result
        return
    
    # Fall back to direct extraction
    logger.debug("Falling back to direct ODS extraction for %s", filename)
    yield from _extract_ods_direct(data, filename)
