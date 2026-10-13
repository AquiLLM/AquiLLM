"""Pure optional generation contract for an existing durable document snapshot."""

from uuid import UUID

from django.core.exceptions import ValidationError

DOCUMENT_FIELDS = frozenset(
    {
        "document_id",
        "document_pkid",
        "model_label",
        "collection_id",
        "source_hash",
    }
)
GUARD_FIELD = "representation_binding"
GUARD_VERSION = "graph-document-representation-v1"


def requested_document_reference(row):
    from apps.documents.services.representation_contracts import DocumentRefV1
    from apps.documents.services.representation_store import document_payload

    try:
        reference = DocumentRefV1(
            row["model_label"],
            row["document_pkid"],
            UUID(row["document_id"]),
            row["collection_id"],
            row["source_hash"],
        )
        document_payload(reference)
        if str(reference.id) != row["document_id"]:
            raise ValueError("noncanonical document UUID")
        return reference
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        raise ValidationError("Invalid representation request owner.") from error


def requested_representation_generation(row):
    if type(row) is not dict or set(row) not in (
        DOCUMENT_FIELDS,
        DOCUMENT_FIELDS | {GUARD_FIELD},
    ):
        raise ValidationError("Invalid representation request snapshot shape.")
    if GUARD_FIELD not in row:
        return None
    guard = row[GUARD_FIELD]
    try:
        if (
            type(guard) is not dict
            or set(guard) != {"version", "generation_key"}
            or guard["version"] != GUARD_VERSION
        ):
            raise ValueError("invalid representation request version")
        key = UUID(guard["generation_key"])
        if str(key) != guard["generation_key"] or key.version is None:
            raise ValueError("noncanonical representation UUID")
        requested_document_reference(row)
        return key
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        raise ValidationError("Invalid requested representation guard.") from error


def request_has_representation_guard(request):
    keys = tuple(
        requested_representation_generation(row) for row in request.requested_documents
    )
    return any(key is not None for key in keys)
