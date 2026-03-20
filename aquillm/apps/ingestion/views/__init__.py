"""Views for ingestion app."""
from .api import (
    insert_one_from_arxiv as api_insert_one_from_arxiv,
    ingest_arxiv as api_ingest_arxiv,
    ingest_pdf as api_ingest_pdf,
    ingest_vtt as api_ingest_vtt,
    ingest_uploads,
    ingest_uploads_status,
    ingestion_monitor as api_ingestion_monitor,
    ingest_webpage,
)
from .pages import (
    insert_one_from_arxiv,
    insert_arxiv,
    ingest_pdf,
    ingest_vtt,
    ingest_handwritten_notes,
    ingestion_monitor,
    ingestion_dashboard,
    pdf_ingestion_monitor,
)

__all__ = [
    # API views
    'api_insert_one_from_arxiv',
    'api_ingest_arxiv',
    'api_ingest_pdf',
    'api_ingest_vtt',
    'ingest_uploads',
    'ingest_uploads_status',
    'api_ingestion_monitor',
    'ingest_webpage',
    # Page views
    'insert_one_from_arxiv',
    'insert_arxiv',
    'ingest_pdf',
    'ingest_vtt',
    'ingest_handwritten_notes',
    'ingestion_monitor',
    'ingestion_dashboard',
    'pdf_ingestion_monitor',
]
