"""Exact database ownership validation for synthetic fixtures."""

from __future__ import annotations

from django.db.models import Q

from .fixture_manifest import (
    ResolvedFixtureManifest,
    assemble_fixture_document,
    canonical_embedding_sha256,
)
from .fixture_seed_contract import (
    FIXTURE_ID,
    HIDDEN_USERNAME,
    VISIBLE_USERNAME,
    FixtureSeedError,
    LogicalFixture,
)
from .fixture_seed_graph_occurrences import validate_occurrences
from .fixture_seed_graph_requests import (
    expected_request_by_collection,
    graph_context,
    physical_labels,
    validate_requests,
)
from .fixture_seed_query import bounded_rows, require_exact_unique_rows


def _user_has_foreign_reference(user, allowed_models: set[type]) -> bool:
    if user.groups.exists() or user.user_permissions.exists():
        return True
    for relation in type(user)._meta.related_objects:
        model = relation.related_model
        if model in allowed_models:
            continue
        try:
            if model._base_manager.filter(**{relation.field.name: user}).exists():
                return True
        except (TypeError, ValueError):
            return True
    return False


def _validate_principals_and_permissions(resolved, collection_ids):
    from django.contrib.auth.models import User

    from apps.collections.models import CollectionPermission

    users = bounded_rows(
        User.objects.filter(username__in=(VISIBLE_USERNAME, HIDDEN_USERNAME)),
        2,
    )
    by_username = require_exact_unique_rows(
        users,
        {VISIBLE_USERNAME, HIDDEN_USERNAME},
        key=lambda user: user.username,
    )
    if any(
        user.is_active
        or user.is_staff
        or user.is_superuser
        or user.has_usable_password()
        for user in users
    ):
        raise FixtureSeedError("fixture database topology is not exact")
    visible_ids = set(expected_request_by_collection(resolved))
    hidden_id = resolved.collections["collection-security-private"].collection_id
    expected = {
        *((by_username[VISIBLE_USERNAME].pk, value, "MANAGE") for value in visible_ids),
        (by_username[HIDDEN_USERNAME].pk, hidden_id, "MANAGE"),
    }
    rows = bounded_rows(
        CollectionPermission.objects.filter(collection_id__in=collection_ids),
        len(expected),
    )
    observed = {(row.user_id, row.collection_id, row.permission) for row in rows}
    if observed != expected:
        raise FixtureSeedError("fixture database topology has a foreign permission")
    return users, by_username


def _validate_documents(resolved, logical, collection_ids, by_username):
    from apps.documents.models import (
        DESCENDED_FROM_DOCUMENT,
        DocumentFigure,
        RawTextDocument,
    )

    document_ids = {binding.document_id for binding in resolved.documents.values()}
    rows = bounded_rows(
        RawTextDocument.objects.filter(collection_id__in=collection_ids),
        len(document_ids),
        order_by=("id", "pkid"),
    )
    by_document_id = require_exact_unique_rows(
        rows,
        document_ids,
        key=lambda document: document.id,
    )
    if list(
        DocumentFigure.objects.filter(
            parent_raw_text_document_id__in=[document.pkid for document in rows]
        ).order_by("pk")[:1]
    ):
        raise FixtureSeedError("fixture database topology has a foreign document")
    for model in DESCENDED_FROM_DOCUMENT:
        if (
            model is not RawTextDocument
            and model.objects.filter(
                Q(collection_id__in=collection_ids) | Q(id__in=document_ids)
            ).exists()
        ):
            raise FixtureSeedError("fixture database topology has a foreign document")
    for symbol, binding in resolved.documents.items():
        logical_document = logical.documents[symbol]
        full_text, _spans = assemble_fixture_document(
            tuple(chunk.text for chunk in logical_document.chunks)
        )
        document = by_document_id[binding.document_id]
        expected_user = (
            by_username[HIDDEN_USERNAME]
            if logical_document.collection_symbol == "collection-security-private"
            else by_username[VISIBLE_USERNAME]
        )
        if (
            document.collection_id != binding.collection_id
            or document.title != logical_document.title
            or document.full_text != full_text
            or document.full_text_hash != binding.full_text_sha256
            or document.ingested_by_id != expected_user.pk
            or document.ingestion_complete is not True
            or document.source_url is not None
            or bool(document.rendered_pdf.name)
        ):
            raise FixtureSeedError("fixture database topology is not exact")
    return document_ids, {document.pkid for document in rows}


def _validate_chunks(resolved, logical, document_ids):
    from apps.documents.models import TextChunk

    chunk_ids = {binding.chunk_id for binding in resolved.chunks.values()}
    rows = bounded_rows(
        TextChunk.objects.filter(doc_id__in=document_ids),
        len(chunk_ids),
    )
    by_chunk_id = require_exact_unique_rows(rows, chunk_ids, key=lambda chunk: chunk.pk)
    for symbol, binding in resolved.chunks.items():
        chunk = by_chunk_id[binding.chunk_id]
        _document_symbol, text, _number = logical.chunks[symbol]
        try:
            checksum = canonical_embedding_sha256(
                tuple(float(v) for v in chunk.embedding)
            )
        except (TypeError, ValueError) as error:
            raise FixtureSeedError("fixture database topology is not exact") from error
        if (
            chunk.doc_id != resolved.documents[binding.document_symbol].document_id
            or chunk.content != text
            or chunk.chunk_number != binding.chunk_number
            or chunk.start_position != binding.start
            or chunk.end_position != binding.end
            or chunk.start_time is not None
            or chunk.modality != TextChunk.Modality.TEXT
            or chunk.metadata != {"chunk_symbol": symbol, "fixture_id": FIXTURE_ID}
            or checksum != binding.embedding_sha256
        ):
            raise FixtureSeedError("fixture database topology is not exact")


def _validate_user_references(users, document_pkids):
    from apps.collections.models import Collection, CollectionPermission
    from apps.documents.models import RawTextDocument

    if (
        RawTextDocument.objects.filter(ingested_by_id__in=[user.pk for user in users])
        .exclude(pkid__in=document_pkids)
        .exists()
    ):
        raise FixtureSeedError("fixture database topology has a foreign user reference")
    allowed = {Collection, CollectionPermission, RawTextDocument}
    if any(_user_has_foreign_reference(user, allowed) for user in users):
        raise FixtureSeedError("fixture database topology has a foreign user reference")


def _validate_absent(resolved, collection_ids):
    from django.contrib.auth.models import User

    from apps.documents.models import DESCENDED_FROM_DOCUMENT, TextChunk

    document_ids = {binding.document_id for binding in resolved.documents.values()}
    chunk_ids = {binding.chunk_id for binding in resolved.chunks.values()}
    if (
        User.objects.filter(username__in=(VISIBLE_USERNAME, HIDDEN_USERNAME)).exists()
        or any(
            model.objects.filter(id__in=document_ids).exists()
            for model in DESCENDED_FROM_DOCUMENT
        )
        or TextChunk.objects.filter(
            Q(doc_id__in=document_ids) | Q(pk__in=chunk_ids)
        ).exists()
    ):
        raise FixtureSeedError("fixture database topology is not exact")
    context = graph_context(resolved, rows_present=False)
    validate_occurrences(context, validate_requests(context), post_cleanup=True)
    return [], []


def validate_owned_topology(
    resolved: ResolvedFixtureManifest,
    logical: LogicalFixture,
    *,
    allow_absent: bool = False,
    locked_collections: list | None = None,
) -> tuple[list, list]:
    from apps.collections.models import Collection

    labels = physical_labels(resolved)
    collection_ids = set(labels)
    collections = locked_collections
    if collections is None:
        collections = bounded_rows(
            Collection.objects.filter(pk__in=collection_ids), len(collection_ids)
        )
    if not collections:
        if not allow_absent:
            raise FixtureSeedError("fixture database topology is not exact")
        return _validate_absent(resolved, collection_ids)
    by_id = require_exact_unique_rows(
        collections, collection_ids, key=lambda item: item.pk
    )
    if any(
        by_id[collection_id].name != f"{FIXTURE_ID}-{label}"
        or by_id[collection_id].parent_id is not None
        for collection_id, label in labels.items()
    ):
        raise FixtureSeedError("fixture database topology is not exact")
    if Collection.objects.filter(parent_id__in=collection_ids).exists():
        raise FixtureSeedError("fixture database topology has a foreign descendant")
    users, by_username = _validate_principals_and_permissions(resolved, collection_ids)
    document_ids, document_pkids = _validate_documents(
        resolved, logical, collection_ids, by_username
    )
    _validate_chunks(resolved, logical, document_ids)
    _validate_user_references(users, document_pkids)
    context = graph_context(resolved, rows_present=True)
    validate_occurrences(context, validate_requests(context), post_cleanup=False)
    return collections, users
