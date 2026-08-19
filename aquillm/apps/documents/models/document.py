"""Base Document model - abstract base class for all document types."""
from __future__ import annotations

import functools
import hashlib
import structlog
import uuid
from contextlib import nullcontext
from typing import TYPE_CHECKING, List, Optional, Any

from django.contrib.auth.models import User
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import PermissionDenied, ValidationError
from django.db import models, router, transaction

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
    # Canonical origin of the document (e.g. the arXiv abstract page or the
    # crawled web URL), surfaced as a "View source" link in the citation modal.
    source_url = models.URLField(max_length=2000, null=True, blank=True)
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
        document_model = type(self)
        database_alias = kwargs.get("using") or router.db_for_write(
            document_model,
            instance=self,
        )
        update_fields = kwargs.get("update_fields")
        requested_fields = None if update_fields is None else set(update_fields)
        writes_text = (
            self._state.adding
            or requested_fields is None
            or bool({"full_text", "full_text_hash"} & requested_fields)
        )
        writes_collection = (
            self._state.adding
            or requested_fields is None
            or bool({"collection", "collection_id"} & requested_fields)
        )
        canonical_hash = self.hash_fn(self.full_text) if writes_text else None

        from apps.knowledge_graph.graph.invalidation import (
            DocumentLifecycleRef,
            consume_document_save_lifecycle,
            document_lifecycle_row_is_locked,
            locked_document_lifecycle_row,
            schedule_document_content_invalidation,
            schedule_document_move_invalidation,
        )

        def publish_chunks() -> None:
            from apps.documents.tasks.chunking import create_chunks

            if canonical_hash is None:
                raise RuntimeError("chunk publication requires a persisted content hash")
            try:
                create_chunks.delay(
                    str(self.id),
                    canonical_hash,
                    document_model._meta.label_lower,
                    int(self.pkid),
                )
            except Exception as exc:
                logger.error(
                    "obs.documents.chunk_enqueue_failed",
                    document_id=str(self.id),
                    concrete_model=document_model._meta.label_lower,
                    document_pkid=int(self.pkid),
                    expected_source_hash=canonical_hash,
                    error_type=type(exc).__name__,
                )

        lifecycle_guard = nullcontext(None)
        if (
            not self._state.adding
            and self.pkid is not None
            and writes_collection
            and not document_lifecycle_row_is_locked(self, using=database_alias)
        ):
            observed = (
                document_model._base_manager.using(database_alias)
                .filter(pkid=self.pkid)
                .values("id", "full_text_hash", "collection_id")
                .first()
            )
            if observed is None:
                raise ValidationError("The exact persisted document row no longer exists.")
            if observed["id"] != self.id:
                raise ValidationError({"id": "A persisted document UUID is immutable."})
            lifecycle_guard = locked_document_lifecycle_row(
                DocumentLifecycleRef(
                    concrete_model_label=document_model._meta.label_lower,
                    document_pkid=int(self.pkid),
                    document_id=self.id,
                ),
                (observed["collection_id"], self.collection_id),
                using=database_alias,
                active_or_building_only=True,
            )

        with lifecycle_guard as locked_state:
            with transaction.atomic(using=database_alias):
                previous = None
                if not self._state.adding and self.pkid is not None:
                    if locked_state is not None:
                        locked_row, _locked_collections = locked_state
                        previous = {
                            "id": locked_row.id,
                            "full_text_hash": locked_row.full_text_hash,
                            "collection_id": locked_row.collection_id,
                        }
                    else:
                        previous = (
                            document_model._base_manager.using(database_alias)
                            .select_for_update()
                            .filter(pkid=self.pkid)
                            .values("id", "full_text_hash", "collection_id")
                            .first()
                        )
                    if previous is not None and previous["id"] != self.id:
                        raise ValidationError(
                            {"id": "A persisted document UUID is immutable."}
                        )
                    if (
                        previous is not None
                        and writes_collection
                        and previous["collection_id"] != self.collection_id
                    ):
                        ownership_source = (
                            locked_state[0] if locked_state is not None else self
                        )
                        child_figures = getattr(
                            ownership_source,
                            "child_figures",
                            None,
                        )
                        if child_figures is not None and child_figures.exists():
                            raise ValidationError(
                                "Documents with derived figures cannot be moved until "
                                "a figure move policy exists"
                            )

                if writes_text:
                    if canonical_hash is None:
                        raise RuntimeError("text writes require a canonical content hash")
                    self.full_text_hash = canonical_hash
                    content_changed = (
                        previous is None
                        or previous["full_text_hash"] != canonical_hash
                    )
                else:
                    content_changed = False
                    if previous is not None:
                        self.full_text_hash = previous["full_text_hash"]

                if content_changed and not dont_rechunk:
                    self.ingestion_complete = False
                    if requested_fields is not None:
                        requested_fields.update(
                            {"full_text", "full_text_hash", "ingestion_complete"}
                        )
                elif requested_fields is not None and writes_text:
                    requested_fields.update({"full_text", "full_text_hash"})

                save_kwargs = dict(kwargs)
                save_kwargs["using"] = database_alias
                if requested_fields is not None:
                    save_kwargs["update_fields"] = sorted(requested_fields)
                super().save(*args, **save_kwargs)

                pending = consume_document_save_lifecycle(self)
                if pending is not None:
                    lifecycle_kind, event, lifecycle_alias = pending
                    if lifecycle_kind == "content":
                        schedule_document_content_invalidation(
                            event,
                            using=lifecycle_alias,
                            after_cleanup=None if dont_rechunk else publish_chunks,
                        )
                    elif lifecycle_kind == "move":
                        schedule_document_move_invalidation(
                            event,
                            using=lifecycle_alias,
                        )
                    else:
                        raise RuntimeError("unknown document lifecycle event")
                elif previous is None and not dont_rechunk:
                    transaction.on_commit(
                        publish_chunks,
                        using=database_alias,
                        robust=True,
                    )

    def move_to(self, new_collection, *, actor):
        """Move this document when ``actor`` may edit both collection boundaries."""
        from apps.knowledge_graph.graph.invalidation import (
            DocumentLifecycleRef,
            locked_document_lifecycle_row,
        )

        document_model = type(self)
        database_alias = self._state.db or router.db_for_write(
            document_model,
            instance=self,
        )
        document_ref = DocumentLifecycleRef(
            concrete_model_label=document_model._meta.label_lower,
            document_pkid=self.pkid,
            document_id=self.id,
        )

        with locked_document_lifecycle_row(
            document_ref,
            (self.collection_id, new_collection.pk),
            using=database_alias,
            active_or_building_only=True,
        ) as (current, _affected_collection_ids):
            if not current.collection.user_can_edit(actor):
                raise PermissionDenied(
                    "You do not have permission to edit the source collection"
                )
            if not new_collection.user_can_edit(actor):
                raise PermissionDenied(
                    "You do not have permission to edit the destination collection"
                )

            child_figures = getattr(current, "child_figures", None)
            if child_figures is not None and child_figures.exists():
                raise ValidationError(
                    "Documents with derived figures cannot be moved until a figure move policy exists"
                )

            current.collection = new_collection
            current.save(
                dont_rechunk=True,
                update_fields=["collection"],
                using=database_alias,
            )

        self.collection = new_collection

    def __str__(self):
        return f'{ContentType.objects.get_for_model(self)} -- {self.title} in {self.collection.name}'

    @property
    def original_text(self):
        return self.full_text
