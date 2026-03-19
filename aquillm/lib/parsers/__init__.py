"""
File parsing utilities.

Provides pure Python parsers for various document formats:
- Documents: PDF, HTML, DOCX, EPUB
- Spreadsheets: XLSX, XLS, ODS, CSV
- Presentations: PPTX, ODP
- Structured: JSON, JSONL, XML, YAML
- Media: SRT captions

For full ingestion pipeline with figure extraction and media transcription,
use aquillm.ingestion.parsers.
"""

from .config import (
    DOCUMENT_EXTENSIONS,
    TABULAR_EXTENSIONS,
    PRESENTATION_EXTENSIONS,
    STRUCTURED_EXTENSIONS,
    TRANSCRIPT_EXTENSIONS,
    IMAGE_EXTENSIONS,
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    clean_filename,
    get_stem,
    guess_content_type,
    detect_ingest_type,
)
from .text_utils import read_text_bytes
from .documents import (
    extract_pdf_text,
    extract_html_text,
    extract_docx_text,
    extract_epub_text,
)
from .spreadsheets import (
    extract_xlsx_text,
    extract_xls_text,
    extract_ods_text,
    extract_csv_text,
)
from .presentations import (
    extract_pptx_text,
    extract_odp_text,
)
from .structured import (
    extract_json_text,
    extract_jsonl_text,
    extract_xml_text,
    extract_yaml_text,
)
from .media import (
    iter_srt_text,
    extract_srt_text,
)

__all__ = [
    # Config
    'DOCUMENT_EXTENSIONS',
    'TABULAR_EXTENSIONS',
    'PRESENTATION_EXTENSIONS',
    'STRUCTURED_EXTENSIONS',
    'TRANSCRIPT_EXTENSIONS',
    'IMAGE_EXTENSIONS',
    'AUDIO_EXTENSIONS',
    'VIDEO_EXTENSIONS',
    'SUPPORTED_EXTENSIONS',
    'clean_filename',
    'get_stem',
    'guess_content_type',
    'detect_ingest_type',
    # Utils
    'read_text_bytes',
    # Documents
    'extract_pdf_text',
    'extract_html_text',
    'extract_docx_text',
    'extract_epub_text',
    # Spreadsheets
    'extract_xlsx_text',
    'extract_xls_text',
    'extract_ods_text',
    'extract_csv_text',
    # Presentations
    'extract_pptx_text',
    'extract_odp_text',
    # Structured
    'extract_json_text',
    'extract_jsonl_text',
    'extract_xml_text',
    'extract_yaml_text',
    # Media
    'iter_srt_text',
    'extract_srt_text',
]
