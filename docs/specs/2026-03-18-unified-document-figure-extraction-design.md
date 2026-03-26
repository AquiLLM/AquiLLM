# Unified Document Figure Extraction - Design Spec

**Date:** 2026-03-18  
**Status:** Approved  
**Goal:** Extract embedded figures/images from all supported document formats, feed them through the RAG pipeline, and display them in chat responses.

---

## Overview

A single `DocumentFigure` model handles extracted images from all supported document formats (PDF, DOCX, PPTX, XLSX, ODS, EPUB). Each format has a dedicated extraction module following a common interface. Figures flow through the existing RAG pipeline for multimodal embedding and retrieval.

```
Document Upload → Parser → Figure Extractor → DocumentFigure → Chunking → Multimodal Embedding → RAG Search → Chat Display
```

---

## Data Model

### `DocumentFigure` Model

Replaces the format-specific `PDFDocumentFigure` with a unified model:

```python
class DocumentFigure(Document):
    """Figure/image extracted from any document format."""
    
    image_file = models.FileField(
        upload_to="document_figures/",
        validators=[FileExtensionValidator(IMAGE_UPLOAD_EXTENSIONS)],
    )
    
    # Generic link to parent document
    parent_content_type = models.ForeignKey(
        ContentType, 
        on_delete=models.CASCADE,
        null=True,
        blank=True,
    )
    parent_object_id = models.UUIDField(null=True, blank=True)
    parent_document = GenericForeignKey('parent_content_type', 'parent_object_id')
    
    # Common fields (always populated)
    source_format = models.CharField(max_length=20, db_index=True)
    figure_index = models.PositiveIntegerField(default=0)
    extracted_caption = models.TextField(blank=True, default="")
    
    # Format-specific location stored as JSON
    location_metadata = models.JSONField(default=dict)
    
    # OCR metadata
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

### Location Metadata by Format

| Format | Example `location_metadata` |
|--------|----------------------------|
| PDF | `{"page_number": 5}` |
| DOCX | `{"paragraph_index": 12}` |
| PPTX | `{"slide_number": 3, "slide_title": "Architecture Overview"}` |
| XLSX | `{"sheet_name": "Dashboard"}` |
| ODS | `{"sheet_name": "Charts"}` |
| EPUB | `{"chapter": "Introduction", "chapter_index": 1}` |

---

## Extraction Modules

### Common Interface

All extraction modules expose the same function signature:

```python
def extract_figures(data: bytes, filename: str = "") -> Iterator[ExtractedFigure]
```

### `ExtractedFigure` Dataclass

Unified return type from all extractors (in `ingestion/figure_extraction/types.py`):

```python
@dataclass
class ExtractedFigure:
    image_bytes: bytes
    image_format: str  # 'png', 'jpeg', 'webp'
    figure_index: int
    nearby_text: str   # Potential caption
    width: int
    height: int
    location_metadata: dict  # Format-specific
```

### Module Structure

```
aquillm/aquillm/ingestion/figure_extraction/
├── __init__.py          # Exports extract_figures_from_document()
├── types.py             # ExtractedFigure dataclass
├── pdf.py               # PDF extraction (PyMuPDF)
├── office.py            # DOCX, PPTX extraction
├── spreadsheet.py       # XLSX, ODS extraction
└── ebook.py             # EPUB extraction
```

### Format-Specific Details

#### PDF (`pdf.py`)
- **Library:** PyMuPDF (fitz)
- **Method:** `doc.extract_image(xref)` for each image on each page
- **Caption:** Text below/above figure bounding box
- **Filters:** Min 100x100 pixels, min 5KB

#### DOCX (`office.py`)
- **Library:** python-docx
- **Method:** Iterate `doc.part.related_parts` for image relationships
- **Caption:** Shape alt text, paragraph before/after inline image
- **Filters:** Same size thresholds

#### PPTX (`office.py`)
- **Library:** python-pptx
- **Method:** Check `shape.shape_type == MSO_SHAPE_TYPE.PICTURE` on each slide
- **Caption:** Shape alt text, slide title, nearby text boxes
- **Filters:** Same size thresholds

#### XLSX (`spreadsheet.py`)
- **Library:** openpyxl
- **Method:** Access `worksheet._images` collection
- **Caption:** Sheet name, nearby cell content
- **Filters:** Same size thresholds

#### ODS (`spreadsheet.py`)
- **Library:** odfpy + zipfile
- **Method:** Extract from `Pictures/` directory in ODF archive
- **Caption:** Sheet name context
- **Filters:** Same size thresholds

#### EPUB (`ebook.py`)
- **Library:** ebooklib
- **Method:** Iterate items with `ITEM_IMAGE` type
- **Caption:** Alt text from HTML, chapter title context
- **Filters:** Same size thresholds

---

## Integration

### Parser Integration

Each `_extract_*` function in `parsers.py` that handles a format with potential images will:

1. Extract text as before (returns primary payload)
2. Call format-specific figure extractor
3. Yield additional payloads for each figure with `normalized_type="document_figure"`

Example for `_extract_docx`:

```python
def _extract_docx(filename: str, data: bytes) -> list[ExtractedTextPayload]:
    payloads = []
    
    # Text extraction (existing)
    doc = Document(io.BytesIO(data))
    text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    payloads.append(ExtractedTextPayload(title=_stem(filename), normalized_type="docx", full_text=text))
    
    # Figure extraction (new)
    try:
        from aquillm.ingestion.figure_extraction import extract_figures_from_document
        for figure in extract_figures_from_document(data, "docx", filename):
            # ... create figure payload
    except Exception:
        pass  # Graceful degradation
    
    return payloads
```

### Task Processing

`tasks.py` creates `DocumentFigure` when `normalized_type == "document_figure"`:

```python
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
```

---

## Configuration

### Environment Variables

```
INGEST_FIGURE_EXTRACTION_ENABLED=true    # Master switch
INGEST_FIGURE_MIN_WIDTH=100              # Minimum image width
INGEST_FIGURE_MIN_HEIGHT=100             # Minimum image height
INGEST_FIGURE_MIN_BYTES=5000             # Minimum file size
INGEST_FIGURE_MAX_PER_DOCUMENT=50        # Cap per document
INGEST_FIGURE_OCR_ENABLED=true           # Run OCR on figures
```

---

## Migration Path

1. Remove `PDFDocumentFigure` model (not yet deployed)
2. Create `DocumentFigure` model
3. Update `pdf_figures.py` to use new types
4. Update `tasks.py` to create `DocumentFigure`
5. Add new extraction modules for other formats
6. Update parsers to call extractors

---

## Testing Strategy

- Unit tests for each extraction module with synthetic documents
- Integration tests verifying end-to-end flow
- Test that figures appear in RAG search results
- Test that figures display in chat responses

---

## What This Enables

- Upload any supported document → figures automatically extracted
- RAG searches retrieve relevant figures based on captions and OCR text
- Chat displays figures inline when relevant to queries
- LLM can analyze and describe figure content using multimodal capabilities
