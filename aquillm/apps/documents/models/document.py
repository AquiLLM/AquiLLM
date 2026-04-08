"""Base Document model - abstract base class for all document types."""
from __future__ import annotations

import functools
import hashlib
import structlog
import uuid
from typing import TYPE_CHECKING, List, Optional, Any

from django.contrib.auth.models import User
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.db import models

if TYPE_CHECKING:
    from .chunks import TextChunk

logger = structlog.stdlib.get_logger(__name__)


def _get_document_types():
    """Lazy getter for document types to avoid circular imports."""
    from .document_types import (
        PDFDocument, TeXDocument, RawTextDocument, VTTDocument,
        HandwrittenNotesDocument, ImageUploadDocument, MediaUploadDocument,
        DocumentFigure
    )
    return [
        PDFDocument,
        TeXDocument,
        RawTextDocument,
        VTTDocument,
        HandwrittenNotesDocument,
        ImageUploadDocument,
        MediaUploadDocument,
        DocumentFigure,
    ]


# Type alias for any document subclass
type DocumentChild = Any  # Will be properly typed when all document types are defined


class Document(models.Model):
    """Abstract base model for all document types."""
    pkid = models.BigAutoField(primary_key=True, editable=False)
    id = models.UUIDField(default=uuid.uuid4, editable=False, db_index=True)
    title = models.CharField(max_length=200)
    full_text = models.TextField()
    collection = models.ForeignKey(
        'apps_collections.Collection',
        on_delete=models.CASCADE,
        related_name='%(class)s_documents'
    )
    full_text_hash = models.CharField(max_length=64, db_index=True)
    ingested_by = models.ForeignKey(User, on_delete=models.RESTRICT)
    ingestion_date = models.DateTimeField(auto_now_add=True)
    ingestion_complete = models.BooleanField(default=True)

    class Meta:
        abstract = True
        constraints = [
            models.UniqueConstraint(
                fields=['collection', 'full_text_hash'],
                name='%(class)s_document_collection_unique'
            )
        ]
        ordering = ['-ingestion_date', 'title']

    @staticmethod
    def hash_fn(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    @property
    def chunks(self):
        from .chunks import TextChunk
        return TextChunk.objects.filter(doc_id=self.id)

    @staticmethod
    def filter(*args, **kwargs) -> List[DocumentChild]:
        doc_types = _get_document_types()
        return functools.reduce(lambda l, r: l + r, [list(x.objects.filter(*args, **kwargs)) for x in doc_types])

    @staticmethod
    def get_by_id(doc_id: uuid.UUID) -> Optional[DocumentChild]:
        from django.conf import settings

        from apps.documents.services import rag_cache

        if getattr(settings, "RAG_CACHE_ENABLED", False):
            ref = rag_cache.get_cached_document_ref(doc_id)
            if ref is not None:
                try:
                    from django.apps import apps

                    model = apps.get_model("apps_documents", str(ref["model"]))
                    hit = model.objects.filter(pkid=int(ref["pkid"])).first()
                    if hit is not None and hit.id == doc_id:
                        return hit
                except Exception:
                    pass

        doc_types = _get_document_types()
        for t in doc_types:
            doc = t.objects.filter(id=doc_id).first()
            if doc:
                if getattr(settings, "RAG_CACHE_ENABLED", False):
                    rag_cache.set_cached_document_ref(
                        doc_id,
                        {"model": doc.__class__.__name__, "pkid": int(doc.pkid)},
                    )
                return doc
        return None

    def save(self, *args, dont_rechunk=False, **kwargs):
        if dont_rechunk:
            super().save(*args, **kwargs)
            return
        
        self.full_text_hash = self.hash_fn(self.full_text)

        is_new = (not (d := Document.get_by_id(doc_id=self.id))) or (self.full_text_hash != d.full_text_hash)
        super().save(*args, **kwargs)
        
        if is_new:
            self.ingestion_complete = False
            self.save(dont_rechunk=True, update_fields=['ingestion_complete'])
            try:
                from apps.documents.tasks.chunking import create_chunks

                create_chunks.delay(str(self.id))
                return
            except Exception as e:
                logger.error("obs.documents.chunk_error", doc_id=self.id, error_type=type(e).__name__, error=str(e))
                self.ingestion_complete = False
                self.save(dont_rechunk=True, update_fields=['ingestion_complete'])

    def move_to(self, new_collection):
        """Move this document to a new collection"""
        if not new_collection.user_can_edit(self.ingested_by):
            raise ValidationError("User does not have permission to move documents to this collection")
        self.collection = new_collection
        self.save()

    def delete(self, *args, **kwargs):
        from .chunks import TextChunk
        TextChunk.objects.filter(doc_id=self.id).delete()
        return super().delete(*args, **kwargs)

    def __str__(self):
        return f'{ContentType.objects.get_for_model(self)} -- {self.title} in {self.collection.name}'

    @property
    def original_text(self):
        return self.full_text
