"""Bounded figure fields; figure provenance is distinct from chunk citations."""

from types import SimpleNamespace

from django.contrib.contenttypes.models import ContentType
from django.db.models.functions import MD5, Length

from apps.collections.services.django_retrieval_authorization import (
    build_selected_scope_authorization_context,
)
from apps.collections.services.retrieval_authorization import (
    revalidate_retrieval_authorization_context,
)
from apps.documents.models import DocumentFigure, TextChunk
from apps.documents.services.source_loading import (
    bounded_source_database,
    source_query_rows,
)
from lib.retrieval.evidence import FigureProvenance, fingerprint_source

from .source_documents import required_source_runtime


def _authorized(user, meta, alias):
    return (
        build_selected_scope_authorization_context(
            principal=user,
            selected_collection_ids=(meta["collection_id"],),
            selected_documents=(
                SimpleNamespace(
                    id=meta["id"],
                    collection_id=meta["collection_id"],
                ),
            ),
            database_alias=alias,
        )
        is not None
    )


def _metadata(query):
    return query.annotate(
        caption_length=Length("extracted_caption"),
        caption_revision=MD5("extracted_caption"),
        text_length=Length("full_text"),
        text_revision=MD5("full_text"),
    )


def bounded_figure_payloads(doc, *, user, max_figures=3):
    runtime = required_source_runtime()
    alias = runtime.authorization.database_alias
    payloads = []
    provenance: list[FigureProvenance] = []
    omitted = False
    if isinstance(doc, DocumentFigure):
        return payloads, provenance, omitted
    with runtime.lock, bounded_source_database(runtime, alias):
        scope = revalidate_retrieval_authorization_context(
            context=runtime.authorization
        )
        if doc.id not in scope.document_ids:
            return [], [], True
        content_type = ContentType.objects.db_manager(alias).get_for_model(
            doc,
            for_concrete_model=False,
        )
        query = (
            DocumentFigure.objects.using(alias)
            .filter(
                parent_content_type=content_type,
                parent_object_id=doc.id,
            )
            .order_by("figure_index", "title")
        )
        metadata = tuple(
            _metadata(query).values(
                "id",
                "title",
                "collection_id",
                "figure_index",
                "image_file",
                "caption_length",
                "caption_revision",
                "text_length",
                "text_revision",
            )[: max_figures * 3 + 1]
        )
        for meta in metadata:
            if len(payloads) >= max_figures:
                omitted = True
                break
            if not meta["image_file"] or not _authorized(user, meta, alias):
                omitted = True
                continue
            field = "extracted_caption" if meta["caption_length"] else "full_text"
            prefix = "caption" if field == "extracted_caption" else "text"
            length, revision = meta[prefix + "_length"], meta[prefix + "_revision"]
            chunk_ids = ()
            # Existing admitted figure chunks are preferred to an extra field read.
            chunks = ()
            if meta["id"] in scope.document_ids:
                chunk_query = (
                    TextChunk.objects.using(alias)
                    .filter(doc_id=meta["id"])
                    .order_by("chunk_number")
                )
                expected = chunk_query.count()
                if expected:
                    chunks = source_query_rows(chunk_query)
                    if len(chunks) != expected:
                        omitted = True
                        continue
            if chunks:
                text = "\n\n".join(chunk.content for chunk in chunks)
                chunk_ids = tuple(chunk.pk for chunk in chunks)
            else:
                if length and not runtime.budget.reserve_text(
                    length, kind="materialized"
                ):
                    omitted = True
                    continue
                if not runtime.budget.can_publish():
                    omitted = True
                    break
                text = (
                    _metadata(query)
                    .filter(
                        id=meta["id"],
                        **{prefix + "_length": length, prefix + "_revision": revision},
                    )
                    .values_list(field, flat=True)
                    .first()
                )
                if text is None:
                    omitted = True
                    continue
            payloads.append(
                {
                    "type": "image",
                    "title": meta["title"],
                    "text": text,
                    "image_url": f"/aquillm/document_image/{meta['id']}/",
                    "figure_index": meta["figure_index"] + 1,
                }
            )
            provenance.append(
                {
                    "figure_id": str(meta["id"]),
                    "parent_id": str(doc.id),
                    "field": field,
                    "revision": revision,
                    "length": length,
                    "source_fingerprint": fingerprint_source(text),
                    "chunk_ids": list(chunk_ids),
                }
            )
    return payloads, provenance, omitted


def revalidate_figure_payloads(payloads, provenance, *, user):
    """Call again immediately before normal-loop provider handoff (Task5)."""
    runtime = required_source_runtime()
    alias = runtime.authorization.database_alias
    retained, identities = [], []
    with bounded_source_database(runtime, alias):
        scope = revalidate_retrieval_authorization_context(
            context=runtime.authorization
        )
        parents = {str(pk) for pk in scope.document_ids}
        for payload, record in zip(payloads, provenance):
            if record["parent_id"] not in parents:
                continue
            meta = (
                _metadata(
                    DocumentFigure.objects.using(alias).filter(
                        id=record["figure_id"],
                        parent_object_id=record["parent_id"],
                    )
                )
                .values(
                    "id",
                    "collection_id",
                    "caption_length",
                    "caption_revision",
                    "text_length",
                    "text_revision",
                )
                .first()
            )
            if meta is None or not _authorized(user, meta, alias):
                continue
            prefix = "caption" if record["field"] == "extracted_caption" else "text"
            if (
                meta[prefix + "_length"] != record["length"]
                or meta[prefix + "_revision"] != record["revision"]
            ):
                continue
            if record["chunk_ids"]:
                chunks = source_query_rows(
                    TextChunk.objects.using(alias)
                    .filter(
                        pk__in=record["chunk_ids"],
                        doc_id=record["figure_id"],
                    )
                    .order_by("chunk_number")
                )
                text = "\n\n".join(chunk.content for chunk in chunks)
                if fingerprint_source(text) != record["source_fingerprint"]:
                    continue
            if fingerprint_source(payload["text"]) != record["source_fingerprint"]:
                continue
            retained.append(payload)
            identities.append(record)
    return retained, identities
