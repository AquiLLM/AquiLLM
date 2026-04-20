"""Celery tasks for the documents app."""

from apps.documents.tasks.chunking import create_chunks as create_chunks

__all__ = ["create_chunks"]
