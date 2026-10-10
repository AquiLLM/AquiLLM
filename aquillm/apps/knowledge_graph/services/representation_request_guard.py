"""A durable retry may publish only the generation its original owner captured."""

from apps.documents.services.representation_store import (
    validate_representation_snapshot,
)

from .failure_codes import StaleBuildError
from .requested_representation import (
    requested_document_reference,
    requested_representation_generation,
)


def validate_requested_representation_context(row, context):
    key = requested_representation_generation(row)
    snapshot = context.representation_snapshot
    if key is None or snapshot is None or len(snapshot.selections) != 1:
        raise StaleBuildError("requested representation is no longer current")
    selection = snapshot.selections[0]
    if (
        selection.document != requested_document_reference(row)
        or selection.generation_key != key
    ):
        raise StaleBuildError("requested representation is no longer current")
    validate_representation_snapshot(snapshot, using="default")


def validate_guarded_request_context(request, context):
    if request is None:
        return
    for row in request.requested_documents:
        key = requested_representation_generation(row)
        if key is not None and row["document_id"] == str(context.identity.document_id):
            validate_requested_representation_context(row, context)
