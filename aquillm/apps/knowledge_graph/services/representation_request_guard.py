"""A durable retry may publish only the generation its original owner captured."""

from apps.documents.services.representation_store import (
    freeze_active_representations,
    validate_representation_snapshot,
)

from .failure_codes import StaleBuildError
from .requested_representation import (
    DOCUMENT_FIELDS,
    requested_document_reference,
    requested_representation_generation,
)


def validate_requested_representation_context(row, context):
    _validate_requested_snapshot(row, context.representation_snapshot)
    validate_representation_snapshot(context.representation_snapshot, using="default")


def _validate_requested_snapshot(row, snapshot):
    key = requested_representation_generation(row)
    if key is None or snapshot is None or len(snapshot.selections) != 1:
        raise StaleBuildError("requested representation is no longer current")
    selection = snapshot.selections[0]
    if (
        selection.document != requested_document_reference(row)
        or selection.generation_key != key
    ):
        raise StaleBuildError("requested representation is no longer current")


def validate_guarded_request_context(request, context):
    if request is None:
        return
    for row in request.requested_documents:
        key = requested_representation_generation(row)
        if key is not None and row["document_id"] == str(context.identity.document_id):
            validate_requested_representation_context(row, context)


def validate_request_completion_snapshot(*, expected, current):
    # Validate the complete guard before comparing the original owner fields.
    keys = tuple(requested_representation_generation(row) for row in expected)
    owners = tuple({field: row[field] for field in DOCUMENT_FIELDS} for row in expected)
    if current != owners:
        raise StaleBuildError("request snapshot changed")
    for row, key in zip(expected, keys, strict=True):
        if key is not None:
            snapshot = freeze_active_representations(
                (requested_document_reference(row),), using="default"
            )
            _validate_requested_snapshot(row, snapshot)
