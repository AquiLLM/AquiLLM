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


def _extract_xlsx_direct(data: bytes, filename: str = "") -> Iterator[ExtractedFigure]:
    """Extract embedded raster images directly from XLSX."""
    try:
        from openpyxl import load_workbook
    except ImportError:
        logger.warning("obs.figures.dependency_missing", dependency="openpyxl", source_format="xlsx")
        return

    try:
        workbook = load_workbook(io.BytesIO(data), read_only=False)
    except Exception as exc:
        logger.warning("obs.figures.open_failed", source_format="xlsx", error=str(exc), error_type=type(exc).__name__)
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
                    logger.debug("obs.figures.image_extract_failed", source_format="xlsx", error=str(exc), error_type=type(exc).__name__)
                    continue

    except Exception as exc:
        logger.warning("obs.figures.extract_failed", source_format="xlsx", error=str(exc), error_type=type(exc).__name__)

    if total_extracted > 0:
        logger.info("obs.figures.extract_done", source_format="xlsx", figure_count=total_extracted, filename=filename)


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
                    logger.debug("obs.figures.image_extract_failed", source_format="ods", path=zip_info.filename, error=str(exc), error_type=type(exc).__name__)
                    continue

    except Exception as exc:
        logger.warning("obs.figures.extract_failed", source_format="ods", error=str(exc), error_type=type(exc).__name__)

    if total_extracted > 0:
        logger.info("obs.figures.extract_done", source_format="ods", figure_count=total_extracted, filename=filename)


def extract_figures_xlsx(data: bytes, filename: str = '') -> Iterator[ExtractedFigure]:
    from .office import _extract_office_figures
    yield from _extract_office_figures(data, 'xlsx', filename, _extract_xlsx_direct)


def extract_figures_ods(data: bytes, filename: str = '') -> Iterator[ExtractedFigure]:
    from .office import _extract_office_figures
    yield from _extract_office_figures(data, 'ods', filename, _extract_ods_direct)
