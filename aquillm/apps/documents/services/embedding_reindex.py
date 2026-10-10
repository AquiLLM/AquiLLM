"""Explicit operator selection and metadata-only vector rebuild planning."""
from dataclasses import dataclass
from uuid import UUID

from django.core.management.base import CommandError
from django.db.models import Count

from apps.documents.models import DESCENDED_FROM_DOCUMENT, TextChunk
from lib.embeddings.document_contracts import build_budget, require_digest
from .representation_contracts import DocumentRefV1, SourceScopeUnavailable
from .representation_store import (
    active_chunk_scope, freeze_active_representations, validate_representation_snapshot,
)


@dataclass(frozen=True)
class ReindexScope:
    model: object
    exact: bool
    filters: dict
    limit: int
    budget: object

    def documents(self):
        query = self.model._base_manager.using("default").only(
            "pkid", "id", "collection_id", "full_text_hash",
        ).filter(**self.filters).order_by("pkid")
        rows = list(query[:self.limit])
        if self.exact and not rows:
            raise CommandError("Exact document owner does not exist.")
        return rows


def reindex_scope(options):
    """Validate all caller-controlled selection values before any DB access."""
    if options["database"] != "default":
        raise CommandError("Only the default database is supported.")
    models = {model._meta.label_lower: model for model in DESCENDED_FROM_DOCUMENT}
    model = models.get(options["model_label"])
    if model is None:
        raise CommandError("An exact allowlisted concrete model label is required.")
    try:
        require_digest(options["space_key"])
        budget = build_budget(options["timeout_ms"])
    except ValueError as error:
        raise CommandError("A SHA-256 space key and timeout 1..120000ms are required.") from error
    exact = options["document_pkid"] is not None or options["document_id"] is not None
    if exact:
        if any(options[key] is not None for key in ("collection_id", "limit", "after_pkid")):
            raise CommandError("Exact owner and collection inventory options are mutually exclusive.")
        try:
            pkid = options["document_pkid"]
            identifier = options["document_id"]
            if type(pkid) is not int or pkid <= 0 or not isinstance(identifier, (str, UUID)):
                raise ValueError
            identifier = UUID(str(identifier))
        except (TypeError, ValueError, AttributeError) as error:
            raise CommandError("Exact owner requires positive PK and valid document UUID.") from error
        return ReindexScope(model, True, dict(pkid=pkid, id=identifier), 1, budget)
    collection = options["collection_id"]
    limit = options["limit"] if options["limit"] is not None else 25
    after = options["after_pkid"] if options["after_pkid"] is not None else 0
    if (type(collection) is not int or collection <= 0 or type(limit) is not int
        or not 1 <= limit <= 100 or type(after) is not int or after < 0):
        raise CommandError("Inventory requires a positive collection, limit 1..100 and after-pkid >=0.")
    return ReindexScope(model, False, dict(collection_id=collection, pkid__gt=after), limit, budget)


def plan_embedding_reindex(document, *, space):
    """Observe current metadata; never certify artifact bytes or vector coverage."""
    ref = DocumentRefV1(document._meta.label_lower, document.pkid, document.id,
                        document.collection_id, document.full_text_hash)
    report = dict(model_label=ref.model_label, document_pkid=ref.pkid,
                  document_id=str(ref.id), collection_id=ref.collection_id,
                  source_hash=ref.source_hash, status="source_unavailable")
    try:
        snapshot = freeze_active_representations((ref,), using="default")
        rows = active_chunk_scope(TextChunk.objects.using("default").all(), snapshot=snapshot)
        counts = dict(rows.order_by().values_list("modality").annotate(count=Count("pk")))
        validate_representation_snapshot(snapshot, using="default")
    except SourceScopeUnavailable:
        return None, report
    selection = snapshot.selections[0]
    report.update(representation_key=str(selection.generation_key) if selection.generation_key else None,
                  active_chunk_count=sum(counts.values()), modalities=counts)
    report["status"] = ("empty" if not counts else "unsupported_modality"
                        if set(counts) - set(space.permitted_modalities) else "planned")
    return selection, report
