"""Read and restore persisted collection schema versions."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from django.db import transaction

from apps.collections.models import (
    Collection,
    CollectionSchemaDraft,
    CollectionSchemaVersion,
)

from . import schema as core
from .schema import (
    SchemaOperationError,
    _candidate_definitions,
    _current_version,
    _diff_summary,
    _locked_draft,
    _published_definitions,
    definitions_checksum,
)


def draft_diff(collection: Collection) -> dict[str, Any]:
    draft = CollectionSchemaDraft.objects.filter(collection=collection).first()
    if draft is None:
        raise SchemaOperationError("draft_not_found", status=404)
    candidate = _candidate_definitions(draft.definitions)
    checksum = definitions_checksum(candidate)
    return _diff_summary(
        draft.base_version,
        core._next_version(collection),
        checksum,
        candidate,
    )


def version_diff(collection: Collection, version: int) -> dict[str, Any]:
    candidate = CollectionSchemaVersion.objects.filter(
        collection=collection, version=version
    ).first()
    if candidate is None:
        raise SchemaOperationError("version_not_found", status=404)
    base = CollectionSchemaVersion.objects.filter(
        collection=collection, version__lt=version
    ).first()
    return _diff_summary(
        base, candidate.version, candidate.checksum, candidate.definitions
    )


def history_page(collection: Collection, cursor: str | None = None) -> dict[str, Any]:
    try:
        offset = max(int(cursor or 0), 0)
    except ValueError as exc:
        raise SchemaOperationError("invalid_cursor") from exc
    page_size = 50
    rows = list(
        CollectionSchemaVersion.objects.filter(collection=collection).order_by(
            "-version"
        )[offset : offset + page_size + 1]
    )
    has_more = len(rows) > page_size
    rows = rows[:page_size]
    return {
        "versions": [
            {
                "version": row.version,
                "checksum": row.checksum,
                "published_at": row.published_at.isoformat(),
                "summary": row.summary,
            }
            for row in rows
        ],
        "next_cursor": str(offset + page_size) if has_more else None,
        "has_more": has_more,
    }


def _restore_challenge(collection_id: int, version: int, draft) -> str:
    return sha256(
        f"restore:{collection_id}:{version}:{draft.pk}:{draft.revision}".encode("ascii")
    ).hexdigest()


def restore_version(collection: Collection, user, version: int):
    with transaction.atomic():
        locked_collection = Collection.objects.select_for_update().get(pk=collection.pk)
        source = CollectionSchemaVersion.objects.filter(
            collection=collection, version=version
        ).first()
        if source is None:
            raise SchemaOperationError("version_not_found", status=404)
        existing = (
            CollectionSchemaDraft.objects.select_for_update()
            .filter(collection=collection)
            .first()
        )
        if existing is not None:
            return {
                "challenge_token": _restore_challenge(collection.pk, version, existing),
                "existing_draft_revision": existing.revision,
                "existing_draft_id": str(existing.pk),
                "last_editor": existing.last_editor.get_username(),
            }
        CollectionSchemaDraft.objects.create(
            collection=collection,
            base_version=_current_version(locked_collection),
            definitions=_published_definitions(source.definitions),
            last_editor=user,
        )
        return None


def replace_with_version(
    collection: Collection,
    user,
    version: int,
    challenge_token: str,
    revision: int | None,
) -> None:
    with transaction.atomic():
        locked_collection = Collection.objects.select_for_update().get(pk=collection.pk)
        draft = _locked_draft(collection, revision)
        if challenge_token != _restore_challenge(collection.pk, version, draft):
            raise SchemaOperationError("invalid_challenge")
        source = CollectionSchemaVersion.objects.filter(
            collection=collection, version=version
        ).first()
        if source is None:
            raise SchemaOperationError("version_not_found", status=404)
        draft.delete()
        CollectionSchemaDraft.objects.create(
            collection=collection,
            base_version=_current_version(locked_collection),
            definitions=_published_definitions(source.definitions),
            last_editor=user,
        )
