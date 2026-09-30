"""Celery tasks for the documents app."""

from apps.documents.tasks.chunking import create_chunks as create_chunks
from apps.documents.tasks.chunk_recovery import recover_chunk_publications

__all__ = ["create_chunks", "recover_chunk_publications"]
