"""Celery tasks for collection-scoped workflows."""

from .schema_generation import generate_collection_schema_task

__all__ = ["generate_collection_schema_task"]
