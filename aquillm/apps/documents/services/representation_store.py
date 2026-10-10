"""Exact legacy-null representation scopes. Source hashes never grant access."""

from dataclasses import asdict
from uuid import UUID
import re

from django.db.models import Exists, F, Func, Q

from lib.embeddings.document_contracts import digest, require_digest
from .representation_contracts import (
    DocumentRefV1,
    RepresentationSelectionV1,
    RepresentationSnapshotV1,
    SourceScopeUnavailable,
)


class SourceSHA256(Func):
    template = "encode(sha256(convert_to(%(expressions)s, 'UTF8')), 'hex')"


def _unavailable():
    raise SourceScopeUnavailable("source_scope_unavailable")


def document_payload(ref):
    if (
        type(ref) is not DocumentRefV1
        or type(ref.model_label) is not str
        or re.fullmatch(r"[a-z0-9_]+\.[a-z0-9_]+", ref.model_label) is None
        or type(ref.id) is not UUID
        or type(ref.pkid) is not int
        or ref.pkid <= 0
        or type(ref.collection_id) is not int
        or ref.collection_id <= 0
    ):
        _unavailable()
    try:
        require_digest(ref.source_hash)
    except ValueError:
        _unavailable()
    return {**asdict(ref), "id": str(ref.id)}


def ordered_selections(snapshot):
    if (
        type(snapshot) is not RepresentationSnapshotV1
        or type(snapshot.selections) is not tuple
    ):
        _unavailable()
    seen = set()
    for s in snapshot.selections:
        if type(s) is not RepresentationSelectionV1:
            _unavailable()
        document_payload(s.document)
        if s.document.id in seen:
            _unavailable()
        seen.add(s.document.id)
        if s.generation_key is not None or s.source_revision_sha256 is not None:
            _unavailable()  # I5 installs the exact active representation owner.
        if s.canonical_sha256 != s.document.source_hash:
            _unavailable()
    return tuple(
        sorted(
            snapshot.selections,
            key=lambda s: (s.document.model_label, s.document.pkid, str(s.document.id)),
        )
    )


def representation_snapshot_digest(snapshot: RepresentationSnapshotV1) -> str:
    return digest(
        {
            "version": "representation-snapshot-v1",
            "selections": [
                {
                    "document": document_payload(s.document),
                    "generation_key": None,
                    "canonical_sha256": s.canonical_sha256,
                    "source_revision_sha256": None,
                }
                for s in ordered_selections(snapshot)
            ],
        }
    )


def validate_representation_snapshot(
    snapshot: RepresentationSnapshotV1, *, using: str
) -> RepresentationSnapshotV1:
    from apps.documents.models import DESCENDED_FROM_DOCUMENT

    for s in ordered_selections(snapshot):
        ref = s.document
        owners = []
        for model in DESCENDED_FROM_DOCUMENT:
            rows = (
                model._base_manager.using(using)
                .filter(id=ref.id)
                .annotate(_canonical=SourceSHA256(F("full_text")))
                .values("pkid", "id", "collection_id", "full_text_hash", "_canonical")[
                    :2
                ]
            )
            owners.extend((model._meta.label_lower, row) for row in rows)
        if len(owners) != 1:
            _unavailable()
        label, row = owners[0]
        if (
            label,
            row["pkid"],
            row["id"],
            row["collection_id"],
            row["full_text_hash"],
            row["_canonical"],
        ) != (
            ref.model_label,
            ref.pkid,
            ref.id,
            ref.collection_id,
            ref.source_hash,
            s.canonical_sha256,
        ):
            _unavailable()
    return snapshot


def freeze_active_representations(
    documents: tuple[DocumentRefV1, ...], *, using: str
) -> RepresentationSnapshotV1:
    if type(documents) is not tuple:
        _unavailable()
    snapshot = RepresentationSnapshotV1(
        tuple(
            RepresentationSelectionV1(ref, None, ref.source_hash, None)
            for ref in documents
        )
    )
    return validate_representation_snapshot(snapshot, using=using)


def active_chunk_scope(queryset, *, snapshot):
    """Lazy predicates also close a UUID collision after this function returns."""
    from apps.documents.models import DESCENDED_FROM_DOCUMENT

    validate_representation_snapshot(snapshot, using=queryset.db)
    condition = Q(pk__in=[])
    for s in ordered_selections(snapshot):
        ref = s.document
        clause = Q(doc_id=ref.id, representation_generation__isnull=True)
        for model in DESCENDED_FROM_DOCUMENT:
            owners = model._base_manager.using(queryset.db).filter(id=ref.id)
            if model._meta.label_lower == ref.model_label:
                exact = owners.annotate(_canonical=SourceSHA256(F("full_text"))).filter(
                    pkid=ref.pkid,
                    collection_id=ref.collection_id,
                    full_text_hash=ref.source_hash,
                    _canonical=s.canonical_sha256,
                )
                clause &= Q(Exists(exact)) & ~Q(Exists(owners.exclude(pkid=ref.pkid)))
            else:
                clause &= ~Q(Exists(owners))
        condition |= clause
    return queryset.filter(condition)
