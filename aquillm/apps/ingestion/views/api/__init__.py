"""Ingestion HTTP API (split modules; import from ``apps.ingestion.views.api``)."""
from apps.ingestion.services.arxiv_ingest import insert_one_from_arxiv

from .arxiv import ingest_arxiv
from .uploads import (
    ingest_pdf,
    ingest_uploads,
    ingest_uploads_status,
    ingest_vtt,
    ingestion_monitor,
)
from .web import ingest_webpage

__all__ = [
    "insert_one_from_arxiv",
    "ingest_arxiv",
    "ingest_pdf",
    "ingest_vtt",
    "ingest_uploads",
    "ingest_uploads_status",
    "ingestion_monitor",
    "ingest_webpage",
]
