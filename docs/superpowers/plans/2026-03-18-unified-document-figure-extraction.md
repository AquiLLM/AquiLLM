# Unified Document Figure Extraction - Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract embedded figures from all supported document formats (PDF, DOCX, PPTX, XLSX, ODS, EPUB) using a unified `DocumentFigure` model, enabling multimodal RAG search and chat display.

**Architecture:** Replace `PDFDocumentFigure` with a generic `DocumentFigure` model using GenericForeignKey. Create extraction modules for each format family. Integrate with existing parsers and task processing.

**Tech Stack:** PyMuPDF, python-docx, python-pptx, openpyxl, odfpy, ebooklib, PIL/Pillow

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `aquillm/aquillm/models.py` | Modify | Replace `PDFDocumentFigure` with `DocumentFigure` |
| `aquillm/aquillm/ingestion/figure_extraction/__init__.py` | Create | Main entry point, format routing |
| `aquillm/aquillm/ingestion/figure_extraction/types.py` | Create | `ExtractedFigure` dataclass |
| `aquillm/aquillm/ingestion/figure_extraction/pdf.py` | Create | PDF extraction (refactor from `pdf_figures.py`) |
| `aquillm/aquillm/ingestion/figure_extraction/office.py` | Create | DOCX and PPTX extraction |
| `aquillm/aquillm/ingestion/figure_extraction/spreadsheet.py` | Create | XLSX and ODS extraction |
| `aquillm/aquillm/ingestion/figure_extraction/ebook.py` | Create | EPUB extraction |
| `aquillm/aquillm/ingestion/parsers.py` | Modify | Update extractors to return figure payloads |
| `aquillm/aquillm/tasks.py` | Modify | Create `DocumentFigure` for figure payloads |
| `aquillm/aquillm/ingestion/pdf_figures.py` | Delete | Replaced by `figure_extraction/pdf.py` |
| `aquillm/aquillm/tests/test_figure_extraction.py` | Create | Unit tests for all extractors |

---

## Chunk 1: Model Refactoring

### Task 1: Replace PDFDocumentFigure with DocumentFigure

**Files:**
- Modify: `aquillm/aquillm/models.py`

- [ ] **Step 1: Remove PDFDocumentFigure class**

Find and delete the `PDFDocumentFigure` class (around line 903-940) in `models.py`.

- [ ] **Step 2: Remove PDFDocumentFigure from DESCENDED_FROM_DOCUMENT**

Remove `PDFDocumentFigure` from the `DESCENDED_FROM_DOCUMENT` list.

- [ ] **Step 3: Remove PDFDocumentFigure from DocumentChild type alias**

Remove `PDFDocumentFigure` from both `DocumentChild` type alias definitions.

- [ ] **Step 4: Add ContentType import**

Add to imports at top of file:
```python
from django.contrib.contenttypes.models import ContentType
```

(Check if already imported - it likely is since GenericForeignKey is used elsewhere)

- [ ] **Step 5: Add DocumentFigure model**

Add after `ImageUploadDocument` class:

```python
class DocumentFigure(Document):
    """
    Figure/image extracted from any document format.
    Uses GenericForeignKey to link to any parent document type.
    """
    image_file = models.FileField(
        upload_to="document_figures/",
        validators=[FileExtensionValidator(IMAGE_UPLOAD_EXTENSIONS)],
    )
    
    parent_content_type = models.ForeignKey(
        ContentType,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
    )
    parent_object_id = models.UUIDField(null=True, blank=True)
    parent_document = GenericForeignKey('parent_content_type', 'parent_object_id')
    
    source_format = models.CharField(
        max_length=20,
        db_index=True,
        help_text="Source format: pdf, docx, pptx, xlsx, ods, epub"
    )
    figure_index = models.PositiveIntegerField(
        default=0,
        help_text="Index of this figure within the source document"
    )
    extracted_caption = models.TextField(
        blank=True,
        default="",
        help_text="Caption text extracted from nearby content"
    )
    location_metadata = models.JSONField(
        default=dict,
        help_text="Format-specific location info (page_number, slide_number, etc.)"
    )
    
    source_content_type = models.CharField(max_length=150, blank=True, default="")
    ocr_provider = models.CharField(max_length=64, blank=True, default="")
    ocr_model = models.CharField(max_length=200, blank=True, default="")

    class Meta:
        ordering = ['source_format', 'figure_index']
        indexes = [
            models.Index(fields=['parent_content_type', 'parent_object_id']),
            models.Index(fields=['source_format']),
        ]
```

- [ ] **Step 6: Add DocumentFigure to DESCENDED_FROM_DOCUMENT**

Add `DocumentFigure` to the `DESCENDED_FROM_DOCUMENT` list.

- [ ] **Step 7: Add DocumentFigure to DocumentChild type alias**

Add `DocumentFigure` to both `DocumentChild` type alias definitions.

- [ ] **Step 8: Commit**

```bash
git add aquillm/aquillm/models.py
git commit -m "refactor: replace PDFDocumentFigure with unified DocumentFigure model"
```

---

## Chunk 2: Extraction Module Infrastructure

### Task 2: Create Figure Extraction Types

**Files:**
- Create: `aquillm/aquillm/ingestion/figure_extraction/types.py`

- [ ] **Step 1: Create the types module**

Create `aquillm/aquillm/ingestion/figure_extraction/types.py`:

```python
"""Common types for document figure extraction."""

from dataclasses import dataclass, field


@dataclass
class ExtractedFigure:
    """Represents a figure extracted from any document format."""
    image_bytes: bytes
    image_format: str  # 'png', 'jpeg', 'webp'
    figure_index: int
    nearby_text: str
    width: int
    height: int
    location_metadata: dict = field(default_factory=dict)
```

- [ ] **Step 2: Commit**

```bash
git add aquillm/aquillm/ingestion/figure_extraction/types.py
git commit -m "feat: add ExtractedFigure dataclass for figure extraction"
```

---

### Task 3: Create PDF Figure Extractor

**Files:**
- Create: `aquillm/aquillm/ingestion/figure_extraction/pdf.py`
- Delete: `aquillm/aquillm/ingestion/pdf_figures.py`

- [ ] **Step 1: Create PDF extraction module**

Create `aquillm/aquillm/ingestion/figure_extraction/pdf.py`:

```python
"""PDF figure extraction using PyMuPDF (fitz)."""

import io
import logging
from typing import Iterator

from .types import ExtractedFigure

logger = logging.getLogger(__name__)

MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100
MIN_IMAGE_BYTES = 5_000
MAX_IMAGES_PER_DOCUMENT = 50


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


def extract_figures(data: bytes, filename: str = "") -> Iterator[ExtractedFigure]:
    """
    Extract figures from a PDF.
    
    Args:
        data: Raw PDF bytes
        filename: Optional filename for logging
        
    Yields:
        ExtractedFigure for each valid image found
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
    
    total_extracted = 0
    
    try:
        for page_num, page in enumerate(doc, start=1):
            if total_extracted >= MAX_IMAGES_PER_DOCUMENT:
                logger.info("Reached max image limit (%d) for PDF", MAX_IMAGES_PER_DOCUMENT)
                break
            
            figure_index_on_page = 0
            image_list = page.get_images(full=True)
            
            for img_info in image_list:
                if total_extracted >= MAX_IMAGES_PER_DOCUMENT:
                    break
                
                try:
                    xref = img_info[0]
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
                    
                    yield ExtractedFigure(
                        image_bytes=image_bytes,
                        image_format=img_ext,
                        figure_index=total_extracted,
                        nearby_text=nearby_text,
                        width=width,
                        height=height,
                        location_metadata={"page_number": page_num},
                    )
                    
                    figure_index_on_page += 1
                    total_extracted += 1
                    
                except Exception as exc:
                    logger.debug("Failed to extract image from page %d: %s", page_num, exc)
                    continue
    finally:
        doc.close()
    
    if total_extracted > 0:
        logger.info("Extracted %d figures from PDF %s", total_extracted, filename)
```

- [ ] **Step 2: Delete old pdf_figures.py**

Delete `aquillm/aquillm/ingestion/pdf_figures.py`.

- [ ] **Step 3: Commit**

```bash
git rm aquillm/aquillm/ingestion/pdf_figures.py
git add aquillm/aquillm/ingestion/figure_extraction/pdf.py
git commit -m "refactor: move PDF figure extraction to figure_extraction module"
```

---

### Task 4: Create Office Document Figure Extractor

**Files:**
- Create: `aquillm/aquillm/ingestion/figure_extraction/office.py`

- [ ] **Step 1: Create office extraction module**

Create `aquillm/aquillm/ingestion/figure_extraction/office.py`:

```python
"""DOCX and PPTX figure extraction."""

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


def extract_figures_docx(data: bytes, filename: str = "") -> Iterator[ExtractedFigure]:
    """
    Extract figures from a DOCX file.
    
    Args:
        data: Raw DOCX bytes
        filename: Optional filename for logging
        
    Yields:
        ExtractedFigure for each valid image found
    """
    try:
        from docx import Document
        from docx.opc.constants import RELATIONSHIP_TYPE as RT
    except ImportError:
        logger.warning("python-docx not installed; skipping DOCX figure extraction")
        return
    
    try:
        doc = Document(io.BytesIO(data))
    except Exception as exc:
        logger.warning("Failed to open DOCX for figure extraction: %s", exc)
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
                    nearby_text="",  # DOCX caption extraction is complex
                    width=width,
                    height=height,
                    location_metadata={"source": "docx_embedded"},
                )
                
                total_extracted += 1
                
            except Exception as exc:
                logger.debug("Failed to extract DOCX image: %s", exc)
                continue
                
    except Exception as exc:
        logger.warning("DOCX figure extraction failed: %s", exc)
    
    if total_extracted > 0:
        logger.info("Extracted %d figures from DOCX %s", total_extracted, filename)


def extract_figures_pptx(data: bytes, filename: str = "") -> Iterator[ExtractedFigure]:
    """
    Extract figures from a PPTX file.
    
    Args:
        data: Raw PPTX bytes
        filename: Optional filename for logging
        
    Yields:
        ExtractedFigure for each valid image found
    """
    try:
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE
    except ImportError:
        logger.warning("python-pptx not installed; skipping PPTX figure extraction")
        return
    
    try:
        prs = Presentation(io.BytesIO(data))
    except Exception as exc:
        logger.warning("Failed to open PPTX for figure extraction: %s", exc)
        return
    
    total_extracted = 0
    
    try:
        for slide_num, slide in enumerate(prs.slides, start=1):
            if total_extracted >= MAX_IMAGES_PER_DOCUMENT:
                break
            
            slide_title = ""
            for shape in slide.shapes:
                if shape.has_text_frame and shape.text.strip():
                    if hasattr(shape, 'is_placeholder') and shape.is_placeholder:
                        slide_title = shape.text.strip()[:200]
                        break
            
            for shape in slide.shapes:
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
                        },
                    )
                    
                    total_extracted += 1
                    
                except Exception as exc:
                    logger.debug("Failed to extract PPTX image from slide %d: %s", slide_num, exc)
                    continue
                    
    except Exception as exc:
        logger.warning("PPTX figure extraction failed: %s", exc)
    
    if total_extracted > 0:
        logger.info("Extracted %d figures from PPTX %s", total_extracted, filename)
```

- [ ] **Step 2: Commit**

```bash
git add aquillm/aquillm/ingestion/figure_extraction/office.py
git commit -m "feat: add DOCX and PPTX figure extraction"
```

---

### Task 5: Create Spreadsheet Figure Extractor

**Files:**
- Create: `aquillm/aquillm/ingestion/figure_extraction/spreadsheet.py`

- [ ] **Step 1: Create spreadsheet extraction module**

Create `aquillm/aquillm/ingestion/figure_extraction/spreadsheet.py`:

```python
"""XLSX and ODS figure extraction."""

import io
import logging
import zipfile
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


def extract_figures_xlsx(data: bytes, filename: str = "") -> Iterator[ExtractedFigure]:
    """
    Extract figures from an XLSX file.
    
    Args:
        data: Raw XLSX bytes
        filename: Optional filename for logging
        
    Yields:
        ExtractedFigure for each valid image found
    """
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
                        location_metadata={"sheet_name": sheet_name},
                    )
                    
                    total_extracted += 1
                    
                except Exception as exc:
                    logger.debug("Failed to extract XLSX image: %s", exc)
                    continue
                    
    except Exception as exc:
        logger.warning("XLSX figure extraction failed: %s", exc)
    
    if total_extracted > 0:
        logger.info("Extracted %d figures from XLSX %s", total_extracted, filename)


def extract_figures_ods(data: bytes, filename: str = "") -> Iterator[ExtractedFigure]:
    """
    Extract figures from an ODS file.
    
    ODS files are ZIP archives with images in the Pictures/ directory.
    
    Args:
        data: Raw ODS bytes
        filename: Optional filename for logging
        
    Yields:
        ExtractedFigure for each valid image found
    """
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
                        location_metadata={"source_path": zip_info.filename},
                    )
                    
                    total_extracted += 1
                    
                except Exception as exc:
                    logger.debug("Failed to extract ODS image %s: %s", zip_info.filename, exc)
                    continue
                    
    except Exception as exc:
        logger.warning("ODS figure extraction failed: %s", exc)
    
    if total_extracted > 0:
        logger.info("Extracted %d figures from ODS %s", total_extracted, filename)
```

- [ ] **Step 2: Commit**

```bash
git add aquillm/aquillm/ingestion/figure_extraction/spreadsheet.py
git commit -m "feat: add XLSX and ODS figure extraction"
```

---

### Task 6: Create EPUB Figure Extractor

**Files:**
- Create: `aquillm/aquillm/ingestion/figure_extraction/ebook.py`

- [ ] **Step 1: Create ebook extraction module**

Create `aquillm/aquillm/ingestion/figure_extraction/ebook.py`:

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add aquillm/aquillm/ingestion/figure_extraction/ebook.py
git commit -m "feat: add EPUB figure extraction"
```

---

### Task 7: Create Main Entry Point

**Files:**
- Create: `aquillm/aquillm/ingestion/figure_extraction/__init__.py`

- [ ] **Step 1: Create the __init__.py**

Create `aquillm/aquillm/ingestion/figure_extraction/__init__.py`:

```python
"""
Document figure extraction for all supported formats.

Usage:
    from aquillm.ingestion.figure_extraction import extract_figures_from_document
    
    for figure in extract_figures_from_document(data, "pdf", "document.pdf"):
        print(figure.width, figure.height, figure.location_metadata)
"""

import logging
from typing import Iterator

from .types import ExtractedFigure

logger = logging.getLogger(__name__)

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
        logger.debug("No figure extraction available for format: %s", source_format)


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
        logger.debug("OCR failed for figure: %s", exc)
        return "", "", ""
```

- [ ] **Step 2: Commit**

```bash
git add aquillm/aquillm/ingestion/figure_extraction/__init__.py
git commit -m "feat: add unified figure extraction entry point"
```

---

## Chunk 3: Parser and Task Integration

### Task 8: Update Parsers to Extract Figures

**Files:**
- Modify: `aquillm/aquillm/ingestion/parsers.py`

- [ ] **Step 1: Update _extract_pdf to use new module**

Replace the figure extraction imports in `_extract_pdf`:

```python
def _extract_pdf(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    """Extract text and figures from a PDF."""
    payloads: list[ExtractedTextPayload] = []
    
    reader = PdfReader(io.BytesIO(data))
    text_parts = []
    for page in reader.pages:
        text_parts.append((page.extract_text() or "").strip())
    full_text = "\n\n".join(part for part in text_parts if part)
    
    payloads.append(ExtractedTextPayload(
        title=_stem(filename),
        normalized_type="pdf",
        full_text=full_text,
    ))
    
    try:
        from aquillm.ingestion.figure_extraction import (
            extract_figures_from_document,
            generate_figure_caption,
            enhance_figure_with_ocr,
        )
        
        doc_title = _stem(filename)
        figure_count = 0
        for figure in extract_figures_from_document(data, "pdf", filename):
            caption = generate_figure_caption(figure, doc_title, "pdf")
            
            ocr_text, ocr_provider, ocr_model = "", "", ""
            try:
                ocr_text, ocr_provider, ocr_model = enhance_figure_with_ocr(figure)
            except Exception:
                pass
            
            combined_text = caption
            if ocr_text and ocr_text.strip():
                combined_text = f"{caption}\n\nText in figure: {ocr_text.strip()}"
            
            fig_filename = f"{doc_title}_fig{figure.figure_index}.{figure.image_format}"
            
            payloads.append(ExtractedTextPayload(
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
                    "source_format": "pdf",
                    "figure_index": figure.figure_index,
                    "extracted_caption": figure.nearby_text,
                    "location_metadata": figure.location_metadata,
                    "width": figure.width,
                    "height": figure.height,
                },
            ))
            figure_count += 1
            
        if figure_count > 0:
            logger.info("Extracted %d figures from PDF %s", figure_count, filename)
    except ImportError:
        logger.debug("Figure extraction not available for %s", filename)
    except Exception as exc:
        logger.warning("Figure extraction failed for %s: %s", filename, exc)
    
    return payloads
```

- [ ] **Step 2: Create helper function for figure extraction**

Add a helper function to reduce code duplication:

```python
def _extract_figures_for_format(
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
            
            payloads.append(ExtractedTextPayload(
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
                    "figure_index": figure.figure_index,
                    "extracted_caption": figure.nearby_text,
                    "location_metadata": figure.location_metadata,
                    "width": figure.width,
                    "height": figure.height,
                },
            ))
            figure_count += 1
        
        if figure_count > 0:
            logger.info("Extracted %d figures from %s %s", figure_count, source_format.upper(), filename)
            
    except ImportError:
        logger.debug("Figure extraction not available for %s", filename)
    except Exception as exc:
        logger.warning("Figure extraction failed for %s: %s", filename, exc)
```

- [ ] **Step 3: Update _extract_docx to return list and extract figures**

```python
def _extract_docx(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    try:
        from docx import Document  # type: ignore
    except Exception as exc:
        raise ExtractionError("python-docx is required for .docx extraction.") from exc
    doc = Document(io.BytesIO(data))
    text = "\n".join(paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip())
    
    payloads = [ExtractedTextPayload(title=_stem(filename), normalized_type="docx", full_text=text)]
    _extract_figures_for_format(filename, data, "docx", payloads)
    return payloads
```

- [ ] **Step 4: Update _extract_pptx to return list and extract figures**

```python
def _extract_pptx(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    try:
        from pptx import Presentation  # type: ignore
    except Exception as exc:
        raise ExtractionError("python-pptx is required for .pptx extraction.") from exc
    prs = Presentation(io.BytesIO(data))
    lines: list[str] = []
    for i, slide in enumerate(prs.slides, start=1):
        lines.append(f"# Slide {i}")
        for shape in slide.shapes:
            text = getattr(shape, "text", "")
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())
    
    payloads = [ExtractedTextPayload(title=_stem(filename), normalized_type="pptx", full_text="\n".join(lines))]
    _extract_figures_for_format(filename, data, "pptx", payloads)
    return payloads
```

- [ ] **Step 5: Update _extract_xlsx to return list and extract figures**

```python
def _extract_xlsx(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    try:
        from openpyxl import load_workbook  # type: ignore
    except Exception as exc:
        raise ExtractionError("openpyxl is required for .xlsx extraction.") from exc
    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    lines: list[str] = []
    for sheet in workbook.worksheets:
        lines.append(f"# Sheet: {sheet.title}")
        for row in sheet.iter_rows(values_only=True):
            values = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
            if values:
                lines.append(", ".join(values))
    
    payloads = [ExtractedTextPayload(title=_stem(filename), normalized_type="xlsx", full_text="\n".join(lines))]
    _extract_figures_for_format(filename, data, "xlsx", payloads)
    return payloads
```

- [ ] **Step 6: Update _extract_ods to return list and extract figures**

```python
def _extract_ods(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    try:
        from odf.opendocument import load  # type: ignore
        from odf.table import Table, TableCell, TableRow  # type: ignore
        from odf.text import P  # type: ignore
    except Exception as exc:
        raise ExtractionError("odfpy is required for .ods extraction.") from exc
    doc = load(io.BytesIO(data))
    lines: list[str] = []
    for table in doc.getElementsByType(Table):
        name = table.getAttribute("name") or "Sheet"
        lines.append(f"# Sheet: {name}")
        for row in table.getElementsByType(TableRow):
            row_values: list[str] = []
            for cell in row.getElementsByType(TableCell):
                text_nodes = cell.getElementsByType(P)
                value = " ".join(node.firstChild.data for node in text_nodes if getattr(node, "firstChild", None))
                value = value.strip()
                if value:
                    row_values.append(value)
            if row_values:
                lines.append(", ".join(row_values))
    
    payloads = [ExtractedTextPayload(title=_stem(filename), normalized_type="ods", full_text="\n".join(lines))]
    _extract_figures_for_format(filename, data, "ods", payloads)
    return payloads
```

- [ ] **Step 7: Update _extract_epub to return list and extract figures**

```python
def _extract_epub(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    try:
        from ebooklib import epub  # type: ignore
    except Exception as exc:
        raise ExtractionError("ebooklib is required for .epub extraction.") from exc
    book = epub.read_epub(io.BytesIO(data))
    text_parts: list[str] = []
    for item in book.get_items():
        content = getattr(item, "get_content", None)
        if not callable(content):
            continue
        raw = item.get_content()
        if not isinstance(raw, (bytes, bytearray)):
            continue
        text_parts.append(BeautifulSoup(raw.decode("utf-8", errors="ignore"), "html.parser").get_text("\n", strip=True))
    
    payloads = [ExtractedTextPayload(title=_stem(filename), normalized_type="epub", full_text="\n\n".join(part for part in text_parts if part))]
    _extract_figures_for_format(filename, data, "epub", payloads)
    return payloads
```

- [ ] **Step 8: Update extract_text_payloads call sites**

Update the call sites in `extract_text_payloads` to handle list returns:

```python
    if extension == ".docx":
        return _extract_docx(filename, data)
    # ... similar for other formats
    if extension == ".pptx":
        return _extract_pptx(filename, data)
    if extension == ".xlsx":
        return _extract_xlsx(filename, data)
    if extension == ".ods":
        return _extract_ods(filename, data)
    if extension == ".epub":
        return _extract_epub(filename, data)
```

- [ ] **Step 9: Commit**

```bash
git add aquillm/aquillm/ingestion/parsers.py
git commit -m "feat: integrate figure extraction into all document parsers"
```

---

### Task 9: Update Task Processing

**Files:**
- Modify: `aquillm/aquillm/tasks.py`

- [ ] **Step 1: Update import to use DocumentFigure**

Change the import:

```python
from .models import ImageUploadDocument, IngestionBatchItem, MediaUploadDocument, RawTextDocument, DocumentFigure
```

- [ ] **Step 2: Update image handling to use DocumentFigure for document_figure type**

```python
            if modality == "image":
                if payload.normalized_type == "document_figure":
                    doc = DocumentFigure(
                        title=safe_title,
                        full_text=full_text,
                        collection=item.batch.collection,
                        ingested_by=item.batch.user,
                        source_format=payload.metadata.get("source_format", "unknown"),
                        figure_index=payload.metadata.get("figure_index", 0),
                        extracted_caption=payload.metadata.get("extracted_caption", ""),
                        location_metadata=payload.metadata.get("location_metadata", {}),
                        source_content_type=source_content_type,
                        ocr_provider=provider,
                        ocr_model=model_name,
                    )
                else:
                    doc = ImageUploadDocument(
                        title=safe_title,
                        full_text=full_text,
                        collection=item.batch.collection,
                        ingested_by=item.batch.user,
                        source_content_type=source_content_type,
                        ocr_provider=provider,
                        ocr_model=model_name,
                    )
                media_bytes = payload.media_bytes
                media_filename = payload.media_filename or item.original_filename or f"image-{item.id}-{index}.bin"
                if media_bytes:
                    doc.image_file.save(media_filename, ContentFile(media_bytes), save=False)
                    raw_media_saved = True
```

- [ ] **Step 3: Commit**

```bash
git add aquillm/aquillm/tasks.py
git commit -m "feat: use DocumentFigure for extracted document figures"
```

---

## Chunk 4: Testing

### Task 10: Create Unit Tests

**Files:**
- Create: `aquillm/aquillm/tests/test_figure_extraction.py`
- Delete: `aquillm/aquillm/tests/test_pdf_figure_extraction.py`

- [ ] **Step 1: Create comprehensive test file**

Create `aquillm/aquillm/tests/test_figure_extraction.py`:

```python
"""Tests for unified document figure extraction."""

import pytest
from io import BytesIO

from aquillm.ingestion.figure_extraction.types import ExtractedFigure


class TestExtractedFigureDataclass:
    """Test the ExtractedFigure dataclass."""
    
    def test_extracted_figure_creation(self):
        figure = ExtractedFigure(
            image_bytes=b"fake-image-data",
            image_format="png",
            figure_index=0,
            nearby_text="Figure 1: Test caption",
            width=100,
            height=100,
            location_metadata={"page_number": 1},
        )
        
        assert figure.figure_index == 0
        assert figure.nearby_text == "Figure 1: Test caption"
        assert figure.location_metadata == {"page_number": 1}


class TestGenerateFigureCaption:
    """Test caption generation."""
    
    def test_caption_with_page_number(self):
        from aquillm.ingestion.figure_extraction import generate_figure_caption
        
        figure = ExtractedFigure(
            image_bytes=b"",
            image_format="png",
            figure_index=0,
            nearby_text="Figure 2: Architecture diagram",
            width=100,
            height=100,
            location_metadata={"page_number": 3},
        )
        
        caption = generate_figure_caption(figure, "Research Paper", "pdf")
        
        assert "Figure 2: Architecture diagram" in caption
        assert "Page 3" in caption
        assert "Research Paper" in caption
    
    def test_caption_with_slide_number(self):
        from aquillm.ingestion.figure_extraction import generate_figure_caption
        
        figure = ExtractedFigure(
            image_bytes=b"",
            image_format="png",
            figure_index=0,
            nearby_text="",
            width=100,
            height=100,
            location_metadata={"slide_number": 5, "slide_title": "Overview"},
        )
        
        caption = generate_figure_caption(figure, "Presentation", "pptx")
        
        assert "Slide 5" in caption
        assert "Overview" in caption


class TestPDFExtraction:
    """Test PDF figure extraction."""
    
    @pytest.fixture
    def fitz(self):
        return pytest.importorskip("fitz")
    
    def test_empty_pdf_returns_no_figures(self, fitz):
        from aquillm.ingestion.figure_extraction import extract_figures_from_document
        
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((100, 100), "Hello World")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        figures = list(extract_figures_from_document(pdf_bytes, "pdf"))
        assert len(figures) == 0
    
    def test_pdf_with_image_extracts_figure(self, fitz):
        from aquillm.ingestion.figure_extraction import extract_figures_from_document
        from PIL import Image
        
        img = Image.new('RGB', (200, 200), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        doc = fitz.open()
        page = doc.new_page()
        page.insert_image(fitz.Rect(50, 50, 250, 250), stream=img_bytes.getvalue())
        pdf_bytes = doc.tobytes()
        doc.close()
        
        figures = list(extract_figures_from_document(pdf_bytes, "pdf"))
        
        assert len(figures) == 1
        assert figures[0].location_metadata.get("page_number") == 1
        assert figures[0].width >= 100


class TestDOCXExtraction:
    """Test DOCX figure extraction."""
    
    def test_docx_without_images(self):
        pytest.importorskip("docx")
        from aquillm.ingestion.figure_extraction import extract_figures_from_document
        from docx import Document
        
        doc = Document()
        doc.add_paragraph("Hello World")
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        
        figures = list(extract_figures_from_document(buffer.getvalue(), "docx"))
        assert len(figures) == 0


class TestPPTXExtraction:
    """Test PPTX figure extraction."""
    
    def test_pptx_without_images(self):
        pytest.importorskip("pptx")
        from aquillm.ingestion.figure_extraction import extract_figures_from_document
        from pptx import Presentation
        
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        buffer = BytesIO()
        prs.save(buffer)
        buffer.seek(0)
        
        figures = list(extract_figures_from_document(buffer.getvalue(), "pptx"))
        assert len(figures) == 0


class TestUnknownFormat:
    """Test handling of unknown formats."""
    
    def test_unknown_format_returns_empty(self):
        from aquillm.ingestion.figure_extraction import extract_figures_from_document
        
        figures = list(extract_figures_from_document(b"fake data", "unknown_format"))
        assert len(figures) == 0
```

- [ ] **Step 2: Delete old test file**

Delete `aquillm/aquillm/tests/test_pdf_figure_extraction.py`.

- [ ] **Step 3: Run tests**

Run:
```bash
docker compose -f docker-compose-development.yml exec web pytest /app/aquillm/aquillm/tests/test_figure_extraction.py -v
```

- [ ] **Step 4: Commit**

```bash
git rm aquillm/aquillm/tests/test_pdf_figure_extraction.py
git add aquillm/aquillm/tests/test_figure_extraction.py
git commit -m "test: add comprehensive tests for unified figure extraction"
```

---

## Chunk 5: Final Steps

### Task 11: Create Migration and Verify

- [ ] **Step 1: Create migration**

Run:
```bash
docker compose -f docker-compose-development.yml exec web python /app/aquillm/manage.py makemigrations aquillm --name document_figure_model
```

- [ ] **Step 2: Apply migration**

Run:
```bash
docker compose -f docker-compose-development.yml exec web python /app/aquillm/manage.py migrate
```

- [ ] **Step 3: Commit migration**

```bash
git add aquillm/aquillm/migrations/
git commit -m "migration: add DocumentFigure model"
```

### Task 12: End-to-End Verification

- [ ] **Step 1: Restart services**

```bash
docker compose -f docker-compose-development.yml restart web worker
```

- [ ] **Step 2: Test PDF upload**

Upload a PDF with figures and verify:
- Text is extracted
- Figures appear as separate documents
- Figures are searchable

- [ ] **Step 3: Test DOCX upload**

Upload a DOCX with embedded images and verify extraction.

- [ ] **Step 4: Test PPTX upload**

Upload a PPTX with images and verify extraction with slide metadata.

- [ ] **Step 5: Test RAG retrieval**

Ask questions about figure content and verify figures are retrieved and displayed.

---

## Summary

This plan implements:
1. **Unified `DocumentFigure` model** with JSONField for format-specific metadata
2. **Extraction modules** for PDF, DOCX, PPTX, XLSX, ODS, and EPUB
3. **Parser integration** so all formats automatically extract figures
4. **Task processing** that creates `DocumentFigure` records
5. **Comprehensive tests** for all extractors

After implementation, users can:
- Upload any supported document and get figures automatically extracted
- Search for content within figures via OCR and captions
- See figures displayed inline in chat responses
- Have the LLM analyze and describe figures directly
