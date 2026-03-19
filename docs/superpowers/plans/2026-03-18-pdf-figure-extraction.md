# PDF Figure Extraction Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract embedded figures and diagrams from PDFs during ingestion, store them with associated captions, and make them searchable via multimodal RAG.

**Architecture:** Extend PDF extraction to use PyMuPDF (fitz) for image extraction. Each extracted figure becomes an ImageUploadDocument linked to the parent PDF. Figures are stored with nearby text as captions, OCR'd for additional context, and embedded multimodally. The existing RAG infrastructure will automatically retrieve and display figures in chat.

**Tech Stack:** PyMuPDF (fitz), PIL/Pillow, existing multimodal embedding infrastructure

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `aquillm/requirements.txt` | Modify | Add PyMuPDF dependency |
| `aquillm/aquillm/ingestion/pdf_figures.py` | Create | Figure extraction logic |
| `aquillm/aquillm/ingestion/parsers.py` | Modify | Integrate figure extraction into PDF pipeline |
| `aquillm/aquillm/models.py` | Modify | Add `PDFDocumentFigure` model to link figures to PDFs |
| `aquillm/aquillm/migrations/XXXX_pdf_figures.py` | Create | Database migration for new model |
| `aquillm/aquillm/tests/test_pdf_figure_extraction.py` | Create | Unit tests for extraction logic |

---

## Chunk 1: Infrastructure Setup

### Task 1: Add PyMuPDF Dependency

**Files:**
- Modify: `aquillm/requirements.txt`

- [ ] **Step 1: Add PyMuPDF to requirements**

Add to `aquillm/requirements.txt`:
```
PyMuPDF>=1.24.0
```

- [ ] **Step 2: Rebuild Docker container**

Run:
```bash
docker compose -f docker-compose-development.yml build web worker
```

- [ ] **Step 3: Verify installation**

Run:
```bash
docker compose -f docker-compose-development.yml exec web python -c "import fitz; print(fitz.version)"
```
Expected: Version string like `(1.24.x, ...)`

- [ ] **Step 4: Commit**

```bash
git add aquillm/requirements.txt
git commit -m "deps: add PyMuPDF for PDF figure extraction"
```

---

### Task 2: Create PDFDocumentFigure Model

**Files:**
- Modify: `aquillm/aquillm/models.py`
- Create: Migration file (auto-generated)

- [ ] **Step 1: Add PDFDocumentFigure model**

Add after `ImageUploadDocument` class in `aquillm/aquillm/models.py`:

```python
class PDFDocumentFigure(Document):
    """
    Represents a figure/diagram extracted from a PDF document.
    Links back to the parent PDF for provenance.
    """
    image_file = models.FileField(
        upload_to="pdf_figures/",
        validators=[FileExtensionValidator(IMAGE_UPLOAD_EXTENSIONS)],
    )
    parent_pdf = models.ForeignKey(
        'PDFDocument',
        on_delete=models.CASCADE,
        related_name='figures',
        null=True,
        blank=True,
    )
    page_number = models.PositiveIntegerField(
        help_text="1-indexed page number where this figure appears"
    )
    figure_index = models.PositiveIntegerField(
        default=0,
        help_text="Index of this figure on the page (0-indexed)"
    )
    extracted_caption = models.TextField(
        blank=True,
        default="",
        help_text="Caption text extracted from nearby PDF text"
    )
    source_content_type = models.CharField(max_length=150, blank=True, default="")
    ocr_provider = models.CharField(max_length=64, blank=True, default="")
    ocr_model = models.CharField(max_length=200, blank=True, default="")

    class Meta:
        ordering = ['parent_pdf', 'page_number', 'figure_index']
        indexes = [
            models.Index(fields=['parent_pdf', 'page_number']),
        ]
```

- [ ] **Step 2: Add to DESCENDED_FROM_DOCUMENT list**

Find `DESCENDED_FROM_DOCUMENT` list in `models.py` and add `PDFDocumentFigure`:

```python
DESCENDED_FROM_DOCUMENT = [
    PDFDocument,
    TeXDocument,
    RawTextDocument,
    ImageUploadDocument,
    MediaUploadDocument,
    HandwrittenNotesDocument,
    PDFDocumentFigure,  # Add this line
]
```

- [ ] **Step 3: Create migration**

Run:
```bash
docker compose -f docker-compose-development.yml exec web python /app/aquillm/manage.py makemigrations aquillm --name pdf_document_figures
```

- [ ] **Step 4: Apply migration**

Run:
```bash
docker compose -f docker-compose-development.yml exec web python /app/aquillm/manage.py migrate
```

- [ ] **Step 5: Commit**

```bash
git add aquillm/aquillm/models.py aquillm/aquillm/migrations/
git commit -m "feat: add PDFDocumentFigure model for extracted PDF figures"
```

---

## Chunk 2: Figure Extraction Logic

### Task 3: Create PDF Figure Extraction Module

**Files:**
- Create: `aquillm/aquillm/ingestion/pdf_figures.py`

- [ ] **Step 1: Create the extraction module**

Create `aquillm/aquillm/ingestion/pdf_figures.py`:

```python
"""
PDF figure extraction using PyMuPDF (fitz).

Extracts embedded images from PDFs along with nearby text that may serve as captions.
"""

import io
import logging
import re
from dataclasses import dataclass
from typing import Iterator

logger = logging.getLogger(__name__)

# Minimum dimensions for extracted images (skip tiny icons/bullets)
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100
# Minimum file size to consider (skip tiny images)
MIN_IMAGE_BYTES = 5_000
# Maximum images to extract per PDF (prevent runaway extraction)
MAX_IMAGES_PER_PDF = 50


@dataclass
class ExtractedFigure:
    """Represents a figure extracted from a PDF."""
    image_bytes: bytes
    image_format: str  # 'png', 'jpeg', etc.
    page_number: int   # 1-indexed
    figure_index: int  # 0-indexed within page
    bbox: tuple[float, float, float, float] | None  # x0, y0, x1, y1
    nearby_text: str   # Text near the figure (potential caption)
    width: int
    height: int


def _extract_nearby_text(page, bbox: tuple, margin: float = 50) -> str:
    """
    Extract text near a figure's bounding box that might be a caption.
    
    Looks for text below the figure first (most common caption location),
    then above, then to the sides.
    """
    if not bbox:
        return ""
    
    x0, y0, x1, y1 = bbox
    page_height = page.rect.height
    
    # Define search regions (prioritize below the figure)
    search_regions = [
        # Below the figure
        (x0 - margin, y1, x1 + margin, min(y1 + 100, page_height)),
        # Above the figure
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
    
    # Look for figure/table references in candidates
    for candidate in caption_candidates:
        lower = candidate.lower()
        if any(marker in lower for marker in ['figure', 'fig.', 'fig ', 'table', 'diagram', 'chart']):
            # Clean up the caption
            lines = candidate.split('\n')
            # Take first few lines that look like a caption
            caption_lines = []
            for line in lines[:5]:
                line = line.strip()
                if line:
                    caption_lines.append(line)
                    # Stop if we hit a period (end of caption)
                    if line.endswith('.'):
                        break
            return ' '.join(caption_lines)[:500]
    
    # Fallback: return first non-empty candidate truncated
    for candidate in caption_candidates:
        if len(candidate) > 20:
            return candidate[:300]
    
    return ""


def extract_figures_from_pdf(pdf_bytes: bytes) -> Iterator[ExtractedFigure]:
    """
    Extract figures/images from a PDF.
    
    Args:
        pdf_bytes: Raw PDF file content
        
    Yields:
        ExtractedFigure objects for each valid image found
    """
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF not installed; skipping PDF figure extraction")
        return
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        logger.warning("Failed to open PDF for figure extraction: %s", exc)
        return
    
    total_extracted = 0
    
    try:
        for page_num, page in enumerate(doc, start=1):
            if total_extracted >= MAX_IMAGES_PER_PDF:
                logger.info("Reached max image limit (%d) for PDF", MAX_IMAGES_PER_PDF)
                break
            
            figure_index = 0
            image_list = page.get_images(full=True)
            
            for img_info in image_list:
                if total_extracted >= MAX_IMAGES_PER_PDF:
                    break
                
                try:
                    xref = img_info[0]
                    
                    # Extract image
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
                    
                    # Get image format
                    img_ext = base_image.get("ext", "png")
                    if img_ext not in ("png", "jpeg", "jpg", "webp"):
                        # Convert to PNG for unsupported formats
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
                    
                    # Try to get bounding box for caption extraction
                    bbox = None
                    try:
                        for img_rect in page.get_image_rects(xref):
                            bbox = tuple(img_rect)
                            break
                    except Exception:
                        pass
                    
                    # Extract nearby text as potential caption
                    nearby_text = _extract_nearby_text(page, bbox) if bbox else ""
                    
                    yield ExtractedFigure(
                        image_bytes=image_bytes,
                        image_format=img_ext,
                        page_number=page_num,
                        figure_index=figure_index,
                        bbox=bbox,
                        nearby_text=nearby_text,
                        width=width,
                        height=height,
                    )
                    
                    figure_index += 1
                    total_extracted += 1
                    
                except Exception as exc:
                    logger.debug("Failed to extract image from page %d: %s", page_num, exc)
                    continue
    finally:
        doc.close()
    
    logger.info("Extracted %d figures from PDF", total_extracted)


def generate_figure_caption(figure: ExtractedFigure, pdf_title: str) -> str:
    """
    Generate a caption/description for a figure.
    
    Combines extracted nearby text with page context.
    """
    parts = []
    
    if figure.nearby_text:
        parts.append(figure.nearby_text)
    else:
        parts.append(f"Figure from page {figure.page_number}")
    
    parts.append(f"(Source: {pdf_title}, page {figure.page_number})")
    
    return " ".join(parts)
```

- [ ] **Step 2: Add import for fitz at module level check**

The module already handles the import inside the function, which is correct for optional dependencies.

- [ ] **Step 3: Commit**

```bash
git add aquillm/aquillm/ingestion/pdf_figures.py
git commit -m "feat: add PDF figure extraction module using PyMuPDF"
```

---

### Task 4: Write Unit Tests for Figure Extraction

**Files:**
- Create: `aquillm/aquillm/tests/test_pdf_figure_extraction.py`

- [ ] **Step 1: Create test file**

Create `aquillm/aquillm/tests/test_pdf_figure_extraction.py`:

```python
"""Tests for PDF figure extraction."""

import pytest
from io import BytesIO

# Skip all tests if PyMuPDF is not installed
fitz = pytest.importorskip("fitz")


class TestExtractedFigureDataclass:
    """Test the ExtractedFigure dataclass."""
    
    def test_extracted_figure_creation(self):
        from aquillm.ingestion.pdf_figures import ExtractedFigure
        
        figure = ExtractedFigure(
            image_bytes=b"fake-image-data",
            image_format="png",
            page_number=1,
            figure_index=0,
            bbox=(0, 0, 100, 100),
            nearby_text="Figure 1: Test caption",
            width=100,
            height=100,
        )
        
        assert figure.page_number == 1
        assert figure.figure_index == 0
        assert figure.nearby_text == "Figure 1: Test caption"


class TestGenerateFigureCaption:
    """Test caption generation."""
    
    def test_caption_with_nearby_text(self):
        from aquillm.ingestion.pdf_figures import ExtractedFigure, generate_figure_caption
        
        figure = ExtractedFigure(
            image_bytes=b"",
            image_format="png",
            page_number=3,
            figure_index=0,
            bbox=None,
            nearby_text="Figure 2: Architecture diagram",
            width=100,
            height=100,
        )
        
        caption = generate_figure_caption(figure, "Research Paper")
        
        assert "Figure 2: Architecture diagram" in caption
        assert "page 3" in caption
        assert "Research Paper" in caption
    
    def test_caption_without_nearby_text(self):
        from aquillm.ingestion.pdf_figures import ExtractedFigure, generate_figure_caption
        
        figure = ExtractedFigure(
            image_bytes=b"",
            image_format="png",
            page_number=5,
            figure_index=1,
            bbox=None,
            nearby_text="",
            width=100,
            height=100,
        )
        
        caption = generate_figure_caption(figure, "Test Document")
        
        assert "page 5" in caption
        assert "Test Document" in caption


class TestExtractFiguresFromPDF:
    """Test PDF figure extraction (requires a real PDF)."""
    
    def test_empty_pdf_returns_no_figures(self):
        from aquillm.ingestion.pdf_figures import extract_figures_from_pdf
        
        # Create a minimal PDF with no images
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((100, 100), "Hello World")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        figures = list(extract_figures_from_pdf(pdf_bytes))
        assert len(figures) == 0
    
    def test_pdf_with_image_extracts_figure(self):
        from aquillm.ingestion.pdf_figures import extract_figures_from_pdf
        from PIL import Image
        
        # Create a test image
        img = Image.new('RGB', (200, 200), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Create a PDF with the image
        doc = fitz.open()
        page = doc.new_page()
        page.insert_image(fitz.Rect(50, 50, 250, 250), stream=img_bytes.getvalue())
        page.insert_text((50, 280), "Figure 1: A red square")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        figures = list(extract_figures_from_pdf(pdf_bytes))
        
        assert len(figures) == 1
        assert figures[0].page_number == 1
        assert figures[0].width >= 100
        assert figures[0].height >= 100
```

- [ ] **Step 2: Run tests**

Run:
```bash
docker compose -f docker-compose-development.yml exec web pytest /app/aquillm/aquillm/tests/test_pdf_figure_extraction.py -v
```
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add aquillm/aquillm/tests/test_pdf_figure_extraction.py
git commit -m "test: add unit tests for PDF figure extraction"
```

---

## Chunk 3: Integration with Ingestion Pipeline

### Task 5: Integrate Figure Extraction into PDF Parsing

**Files:**
- Modify: `aquillm/aquillm/ingestion/parsers.py`
- Modify: `aquillm/aquillm/ingestion/__init__.py`

- [ ] **Step 1: Update _extract_pdf to also extract figures**

Modify `_extract_pdf` function in `aquillm/aquillm/ingestion/parsers.py`:

```python
def _extract_pdf(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    """Extract text and figures from a PDF."""
    from pypdf import PdfReader
    
    payloads = []
    
    # Extract text (existing logic)
    reader = PdfReader(io.BytesIO(data))
    text_parts = []
    for page in reader.pages:
        text_parts.append((page.extract_text() or "").strip())
    full_text = "\n\n".join(part for part in text_parts if part)
    
    payloads.append(ExtractedTextPayload(
        title=_stem(filename),
        normalized_type="pdf",
        full_text=full_text,
        media_bytes=data,  # Keep original PDF bytes for reference
        media_filename=filename,
        media_content_type="application/pdf",
    ))
    
    # Extract figures (new logic)
    try:
        from aquillm.ingestion.pdf_figures import extract_figures_from_pdf, generate_figure_caption
        
        pdf_title = _stem(filename)
        for figure in extract_figures_from_pdf(data):
            caption = generate_figure_caption(figure, pdf_title)
            
            # Generate filename for the figure
            fig_filename = f"{_stem(filename)}_page{figure.page_number}_fig{figure.figure_index}.{figure.image_format}"
            
            payloads.append(ExtractedTextPayload(
                title=f"{pdf_title} - Figure (Page {figure.page_number})",
                normalized_type="pdf_figure",
                full_text=caption,
                modality="image",
                media_bytes=figure.image_bytes,
                media_filename=fig_filename,
                media_content_type=f"image/{figure.image_format}",
                metadata={
                    "source_pdf": filename,
                    "page_number": figure.page_number,
                    "figure_index": figure.figure_index,
                    "extracted_caption": figure.nearby_text,
                    "width": figure.width,
                    "height": figure.height,
                },
            ))
            
        logger.info("Extracted %d figures from PDF %s", len(payloads) - 1, filename)
    except ImportError:
        logger.debug("PyMuPDF not available; skipping figure extraction for %s", filename)
    except Exception as exc:
        logger.warning("Figure extraction failed for %s: %s", filename, exc)
    
    return payloads
```

- [ ] **Step 2: Update the call site to handle list return**

The `_extract_pdf` now returns a list. Find where it's called in `extract_text_payloads` and ensure it handles the list:

In `extract_text_payloads` function, change:
```python
if extension == ".pdf":
    return [_extract_pdf(filename, data)]
```

To:
```python
if extension == ".pdf":
    return _extract_pdf(filename, data)  # Now returns list directly
```

- [ ] **Step 3: Update __init__.py exports if needed**

Check `aquillm/aquillm/ingestion/__init__.py` and add export if needed.

- [ ] **Step 4: Run existing tests to verify no regressions**

Run:
```bash
docker compose -f docker-compose-development.yml exec web pytest /app/aquillm/aquillm/tests/ -v -k "pdf or ingest"
```

- [ ] **Step 5: Commit**

```bash
git add aquillm/aquillm/ingestion/parsers.py aquillm/aquillm/ingestion/__init__.py
git commit -m "feat: integrate PDF figure extraction into ingestion pipeline"
```

---

### Task 6: Handle PDF Figures in Task Processing

**Files:**
- Modify: `aquillm/aquillm/tasks.py`

- [ ] **Step 1: Add handling for pdf_figure modality**

In `ingest_uploaded_file_task`, the existing logic should mostly work since pdf_figure payloads have `modality="image"`. However, we need to ensure the metadata is preserved.

Check that the existing image handling in `tasks.py` (around line 53-67) works for pdf_figures. The key is that:
- `modality == "image"` → Creates `ImageUploadDocument`
- `media_bytes` contains the figure image
- `payload.metadata` contains source PDF info

If needed, add special handling for `normalized_type == "pdf_figure"` to use `PDFDocumentFigure` model instead:

```python
if modality == "image":
    if payload.normalized_type == "pdf_figure":
        # Use PDFDocumentFigure for extracted PDF figures
        doc = PDFDocumentFigure(
            title=safe_title,
            full_text=full_text,
            collection=item.batch.collection,
            ingested_by=item.batch.user,
            source_content_type=source_content_type,
            ocr_provider=provider,
            ocr_model=model_name,
            page_number=payload.metadata.get("page_number", 1),
            figure_index=payload.metadata.get("figure_index", 0),
            extracted_caption=payload.metadata.get("extracted_caption", ""),
        )
    else:
        doc = ImageUploadDocument(
            # ... existing code
        )
```

- [ ] **Step 2: Import PDFDocumentFigure**

Add to imports at top of `tasks.py`:
```python
from .models import ImageUploadDocument, IngestionBatchItem, MediaUploadDocument, RawTextDocument, PDFDocumentFigure
```

- [ ] **Step 3: Run integration test**

Test by uploading a PDF with figures through the UI and checking:
- Text is extracted
- Figures appear as separate documents
- Figures have image chunks

- [ ] **Step 4: Commit**

```bash
git add aquillm/aquillm/tasks.py
git commit -m "feat: handle PDF figures in ingestion task"
```

---

## Chunk 4: OCR Enhancement for Figures

### Task 7: Add OCR for Extracted Figures

**Files:**
- Modify: `aquillm/aquillm/ingestion/pdf_figures.py`

- [ ] **Step 1: Add OCR option to figure extraction**

The existing OCR infrastructure (`extract_text_from_image` in `ocr_utils.py`) can be used. Figures should be OCR'd to extract any text within them (labels, annotations, etc.).

Add to `pdf_figures.py`:

```python
def enhance_figure_with_ocr(figure: ExtractedFigure) -> tuple[str, str, str]:
    """
    Run OCR on a figure to extract embedded text.
    
    Returns:
        Tuple of (ocr_text, provider, model)
    """
    try:
        from aquillm.ocr_utils import extract_text_from_image
        import io
        
        result = extract_text_from_image(io.BytesIO(figure.image_bytes))
        
        ocr_text = result.get("extracted_text", "")
        provider = result.get("provider", "")
        model = result.get("model", "")
        
        return ocr_text, provider, model
    except Exception as exc:
        logger.debug("OCR failed for figure: %s", exc)
        return "", "", ""
```

- [ ] **Step 2: Update parsers.py to use OCR**

In `_extract_pdf`, after extracting each figure, optionally run OCR:

```python
# After figure extraction, enhance with OCR
ocr_text, ocr_provider, ocr_model = "", "", ""
try:
    from aquillm.ingestion.pdf_figures import enhance_figure_with_ocr
    ocr_text, ocr_provider, ocr_model = enhance_figure_with_ocr(figure)
except Exception:
    pass

# Combine caption with OCR text
combined_text = caption
if ocr_text and ocr_text.strip():
    combined_text = f"{caption}\n\nText in figure: {ocr_text.strip()}"

payloads.append(ExtractedTextPayload(
    # ... existing fields
    full_text=combined_text,
    provider=ocr_provider,
    model=ocr_model,
    # ...
))
```

- [ ] **Step 3: Commit**

```bash
git add aquillm/aquillm/ingestion/pdf_figures.py aquillm/aquillm/ingestion/parsers.py
git commit -m "feat: add OCR enhancement for extracted PDF figures"
```

---

## Chunk 5: Testing and Documentation

### Task 8: End-to-End Testing

- [ ] **Step 1: Restart all services**

```bash
docker compose -f docker-compose-development.yml restart web worker
```

- [ ] **Step 2: Upload a PDF with figures**

Use the UI to upload a PDF that contains diagrams or figures.

- [ ] **Step 3: Verify figure extraction**

Check the collection view - you should see:
- The main PDF document
- Separate image documents for each extracted figure

- [ ] **Step 4: Test RAG retrieval**

In chat, ask about something in a figure. Verify:
- The figure is retrieved via search
- The LLM can see and describe the figure
- The figure is displayed in the chat UI

- [ ] **Step 5: Document the feature**

Update any relevant documentation about PDF ingestion to mention figure extraction.

---

## Summary

This plan implements:
1. **PyMuPDF integration** for reliable image extraction from PDFs
2. **Caption extraction** by analyzing text near figures
3. **OCR enhancement** to extract text within figures
4. **Seamless integration** with existing multimodal RAG infrastructure
5. **Proper model relationships** linking figures to their source PDFs

After implementation, users can:
- Upload PDFs and automatically get figures extracted
- Search for content within figures (via OCR + captions)
- See figures displayed inline in chat responses
- The LLM can analyze and describe figures directly
