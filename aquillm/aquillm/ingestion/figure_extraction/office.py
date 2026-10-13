"""DOCX and PPTX figure extraction.

Supports two extraction methods:
1. PDF conversion + extraction - converts to PDF using LibreOffice, then extracts figures
2. Direct raster extraction - extracts embedded images directly (fallback)
"""

import io
import structlog
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


def _normalize_image(image_bytes: bytes, content_type: str) -> tuple[bytes, str]:
    """Normalize image to PNG if needed, return (bytes, format)."""
    ext = "png"
    if "jpeg" in content_type or "jpg" in content_type:
        ext = "jpeg"
    elif "png" in content_type:
        ext = "png"
    elif "webp" in content_type:
        ext = "webp"
    else:
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


def _extract_via_pdf(data: bytes, source_format: str, filename: str) -> tuple[ExtractedFigure, ...]:
    from .office_convert import convert_figures_pdf
    from .pdf import extract_figures as extract_pdf_figures
    outcome = convert_figures_pdf(data, source_format, filename)
    if outcome.status != 'converted':
        return ()
    figures = []
    try:
        for figure in extract_pdf_figures(outcome.pdf_bytes, filename):
            figure.location_metadata['converted_from'] = source_format
            figures.append(figure)
    except Exception:
        logger.debug('obs.figures.converted_extract_failed', source_format=source_format)
    return tuple(figures)


def _extract_office_figures(data, source_format, filename, direct):
    from hashlib import sha256
    seen = set()
    total = 0
    try:
        converted = _extract_via_pdf(data, source_format, filename)
    except Exception:
        converted = ()
    for factory in (lambda: iter(converted), lambda: direct(data, filename)):
        try:
            iterator = factory()
            for figure in iterator:
                digest = sha256(figure.image_bytes).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                figure.figure_index = total
                yield figure
                total += 1
                if total >= MAX_IMAGES_PER_DOCUMENT:
                    return
        except Exception:
            logger.debug('obs.figures.extract_failed', source_format=source_format)


def _extract_docx_direct(data: bytes, filename: str = "") -> Iterator[ExtractedFigure]:
    """Extract embedded raster images directly from DOCX."""
    try:
        from docx import Document
    except ImportError:
        logger.warning("obs.figures.dependency_missing", dependency="python-docx", source_format="docx")
        return

    try:
        doc = Document(io.BytesIO(data))
    except Exception as exc:
        logger.warning("obs.figures.open_failed", source_format="docx", error=str(exc), error_type=type(exc).__name__)
        return
    
    total_extracted = 0
    
    try:
        for rel in doc.part.rels.values():
            if total_extracted >= MAX_IMAGES_PER_DOCUMENT:
                break
            
            if "image" not in rel.reltype:
                continue
            
            try:
                image_part = rel.target_part
                image_bytes = image_part.blob
                
                if not image_bytes or len(image_bytes) < MIN_IMAGE_BYTES:
                    continue
                
                width, height = _get_image_dimensions(image_bytes)
                if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                    continue
                
                content_type = getattr(image_part, 'content_type', 'image/png')
                image_bytes, img_format = _normalize_image(image_bytes, content_type)
                
                yield ExtractedFigure(
                    image_bytes=image_bytes,
                    image_format=img_format,
                    figure_index=total_extracted,
                    nearby_text="",
                    width=width,
                    height=height,
                    location_metadata={"source": "docx_embedded", "extraction_method": "direct"},
                )
                
                total_extracted += 1
                
            except Exception as exc:
                logger.debug("obs.figures.image_extract_failed", source_format="docx", error=str(exc), error_type=type(exc).__name__)
                continue

    except Exception as exc:
        logger.warning("obs.figures.extract_failed", source_format="docx", error=str(exc), error_type=type(exc).__name__)

    if total_extracted > 0:
        logger.info("obs.figures.extract_done", source_format="docx", figure_count=total_extracted, filename=filename)


def _extract_pptx_direct(data: bytes, filename: str = "") -> Iterator[ExtractedFigure]:
    """Extract embedded raster images directly from PPTX."""
    def ordered_shapes(shapes):
        for shape in shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                yield from ordered_shapes(shape.shapes)
            else:
                yield shape

    try:
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE
    except ImportError:
        logger.warning("obs.figures.dependency_missing", dependency="python-pptx", source_format="pptx")
        return

    try:
        prs = Presentation(io.BytesIO(data))
    except Exception as exc:
        logger.warning("obs.figures.open_failed", source_format="pptx", error=str(exc), error_type=type(exc).__name__)
        return
    
    total_extracted = 0
    
    try:
        for slide_num, slide in enumerate(prs.slides, start=1):
            if total_extracted >= MAX_IMAGES_PER_DOCUMENT:
                break
            
            slide_title = ""
            for shape in ordered_shapes(slide.shapes):
                if shape.has_text_frame and shape.text.strip():
                    if hasattr(shape, 'is_placeholder') and shape.is_placeholder:
                        slide_title = shape.text.strip()[:200]
                        break
            
            for shape in ordered_shapes(slide.shapes):
                if total_extracted >= MAX_IMAGES_PER_DOCUMENT:
                    break
                
                if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
                    continue
                
                try:
                    image = shape.image
                    image_bytes = image.blob
                    
                    if not image_bytes or len(image_bytes) < MIN_IMAGE_BYTES:
                        continue
                    
                    width, height = _get_image_dimensions(image_bytes)
                    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                        continue
                    
                    content_type = image.content_type or 'image/png'
                    image_bytes, img_format = _normalize_image(image_bytes, content_type)
                    
                    alt_text = ""
                    try:
                        if hasattr(shape, '_element'):
                            desc_elem = shape._element.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}cNvPr')
                            if desc_elem is not None:
                                alt_text = desc_elem.get('descr', '')[:500]
                    except Exception:
                        pass
                    
                    nearby_text = alt_text or slide_title
                    
                    yield ExtractedFigure(
                        image_bytes=image_bytes,
                        image_format=img_format,
                        figure_index=total_extracted,
                        nearby_text=nearby_text,
                        width=width,
                        height=height,
                        location_metadata={
                            "slide_number": slide_num,
                            "slide_title": slide_title,
                            "extraction_method": "direct",
                        },
                    )
                    
                    total_extracted += 1
                    
                except Exception as exc:
                    logger.debug("obs.figures.image_extract_failed", source_format="pptx", slide_number=slide_num, error=str(exc), error_type=type(exc).__name__)
                    continue

    except Exception as exc:
        logger.warning("obs.figures.extract_failed", source_format="pptx", error=str(exc), error_type=type(exc).__name__)

    if total_extracted > 0:
        logger.info("obs.figures.extract_done", source_format="pptx", figure_count=total_extracted, filename=filename)


def extract_figures_docx(data: bytes, filename: str = '') -> Iterator[ExtractedFigure]:
    yield from _extract_office_figures(data, 'docx', filename, _extract_docx_direct)


def extract_figures_pptx(data: bytes, filename: str = '') -> Iterator[ExtractedFigure]:
    yield from _extract_office_figures(data, 'pptx', filename, _extract_pptx_direct)
