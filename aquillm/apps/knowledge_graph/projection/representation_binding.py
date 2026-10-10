"""Exact, versioned representation payloads stored on graph authority records."""

from dataclasses import asdict
from uuid import UUID

from django.core.exceptions import ValidationError

from apps.documents.services.representation_contracts import (
    DocumentRefV1,
    RepresentationSelectionV1,
    RepresentationSnapshotV1,
    SourceScopeUnavailable,
)
from apps.documents.services.representation_store import (
    freeze_active_representations,
    ordered_selections,
    representation_snapshot_digest,
    validate_representation_snapshot,
)


def snapshot_payload(snapshot):
    # I owns digest semantics and non-null ownership; G never relaxes that guard.
    representation_snapshot_digest(snapshot)
    return {
        "version": "representation-snapshot-v1",
        "selections": [
            {
                **asdict(s),
                "document": {**asdict(s.document), "id": str(s.document.id)},
                "generation_key": None
                if s.generation_key is None
                else str(s.generation_key),
            }
            for s in ordered_selections(snapshot)
        ],
    }


def snapshot_from_payload(payload, digest):
    try:
        if type(payload) is not dict or set(payload) != {"version", "selections"}:
            raise ValueError("invalid representation payload")
        if (
            payload["version"] != "representation-snapshot-v1"
            or type(payload["selections"]) is not list
        ):
            raise ValueError("invalid representation payload version")
        selections = []
        for row in payload["selections"]:
            if type(row) is not dict or set(row) != {
                "document",
                "generation_key",
                "canonical_sha256",
                "source_revision_sha256",
            }:
                raise ValueError("invalid representation selection")
            document = dict(row["document"])
            document["id"] = UUID(document["id"])
            key = row["generation_key"]
            selections.append(
                RepresentationSelectionV1(
                    DocumentRefV1(**document),
                    None if key is None else UUID(key),
                    row["canonical_sha256"],
                    row["source_revision_sha256"],
                )
            )
        snapshot = RepresentationSnapshotV1(tuple(selections))
        if (
            snapshot_payload(snapshot) != payload
            or representation_snapshot_digest(snapshot) != digest
        ):
            raise ValueError("representation payload digest mismatch")
        return snapshot
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValidationError("Invalid graph representation binding.") from exc


def validate_record_binding(record):
    payload, digest = record.representation_snapshot, record.representation_digest
    if payload is None and digest is None:
        return None
    snapshot = snapshot_from_payload(payload, digest)
    if hasattr(record, "scope_type"):
        if record.scope_type == "document":
            if len(snapshot.selections) != 1 or str(
                snapshot.selections[0].document.id
            ) != str(record.scope_id):
                raise ValidationError(
                    "Representation binding escaped document artifact."
                )
            if snapshot.selections[0].document.source_hash != record.source_hash:
                raise ValidationError(
                    "Representation binding source differs from artifact."
                )
        elif any(
            s.document.collection_id != record.collection_scope_id
            for s in snapshot.selections
        ):
            raise ValidationError("Representation binding escaped collection artifact.")
    elif any(
        s.document.collection_id != record.collection_pk_snapshot
        for s in snapshot.selections
    ):
        raise ValidationError("Representation binding escaped projection collection.")
    return snapshot


def freeze_document_ids(document_ids, *, using):
    from apps.documents.models import DESCENDED_FROM_DOCUMENT

    wanted = set(document_ids)
    refs = []
    for model in DESCENDED_FROM_DOCUMENT:
        for row in (
            model._base_manager.using(using)
            .filter(id__in=wanted)
            .values(
                "pkid",
                "id",
                "collection_id",
                "full_text_hash",
            )
        ):
            refs.append(
                DocumentRefV1(
                    model._meta.label_lower,
                    row["pkid"],
                    row["id"],
                    row["collection_id"],
                    row["full_text_hash"],
                )
            )
    if {r.id for r in refs} != wanted or len(refs) != len(wanted):
        raise SourceScopeUnavailable("source_scope_unavailable")
    return freeze_active_representations(tuple(refs), using=using)


def validate_binding_current(record, *, using):
    snapshot = validate_record_binding(record)
    if snapshot is not None:
        validate_representation_snapshot(snapshot, using=using)
    return snapshot
