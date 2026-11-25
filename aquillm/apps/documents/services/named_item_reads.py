"""Bounded, authority-checked reads for published PDF named-item metadata."""

from django.db import connections
from django.db.models import F
from django.db.models.functions import Length, MD5

from .named_item_projection import (
    MAX_CANONICAL,
    MAX_MEMBERS,
    MAX_PAYLOAD,
    owner_binding,
)
from .representation_contracts import RepresentationSnapshotV1, SourceScopeUnavailable
from .representation_store import (
    SourceSHA256,
    _active_chunk_predicates,
    validate_representation_snapshot,
)

PDF_MODEL = "apps_documents.pdfdocument"


def authorized_selections(scope, owner_ids):
    from .chunk_search_scope import FrozenRetrievalScope, current_scope
    from .graph_source_runtime import graph_source_runtime
    from .hybrid_graph_authorization import is_exact_authorization_context

    if (
        type(scope) is not FrozenRetrievalScope
        or scope is not current_scope()
        or scope.runtime is None
        or scope.runtime is not graph_source_runtime()
        or not is_exact_authorization_context(scope.authorization)
        or scope.runtime.authorization is not scope.authorization
        or scope.using != scope.authorization.database_alias
        or not set(owner_ids) <= scope.requested_document_ids
    ):
        raise SourceScopeUnavailable("named_item_scope_unavailable")
    selected = {
        s.document.id: s
        for s in scope.representation.selections
        if s.document.id in owner_ids
    }
    if len(selected) != len(owner_ids):
        raise SourceScopeUnavailable("named_item_scope_unavailable")
    return selected


def validate_sources(scope, selected):
    """Reauthorize the original selection and check only the requested owners' source bytes."""
    from .chunk_search_scope import validate_requested_authorization

    if not scope.runtime.budget.can_publish():
        raise SourceScopeUnavailable("named_item_scope_unavailable")
    if scope.proposal_binding is not None:
        from apps.chat.services.retrieval_query_runtime import (
            validate_bound_before_current_scope,
        )

        validate_bound_before_current_scope(scope.proposal_binding, scope)
    validate_requested_authorization(scope.runtime, scope.authorization, ())
    validate_representation_snapshot(
        RepresentationSnapshotV1(tuple(selected.values())), using=scope.using
    )
    if not scope.runtime.budget.can_publish():
        raise SourceScopeUnavailable("named_item_scope_unavailable")


def _legacy_membership(rows):
    """A single bounded digest of ordered integers. No body or body hash is read."""
    columns = rows.order_by("pk").values_list(
        "pk", "chunk_number", "start_position", "end_position"
    )[: MAX_MEMBERS + 1]
    sql, params = columns.query.get_compiler(using=rows.db).as_sql()
    with connections[rows.db].cursor() as cursor:
        cursor.execute(
            "SELECT encode(sha256(convert_to(string_agg(concat_ws(':', id, n, lo, hi), ',' ORDER BY id),"
            " 'UTF8')), 'hex') FROM (" + sql + ") AS members(id,n,lo,hi)",
            params,
        )
        return cursor.fetchone()[0]


def ready_owner(scope, selected):
    from apps.documents.models import PDFNamedItemIndex, TextChunk

    if selected.document.model_label != PDF_MODEL:
        return None
    header = (
        PDFNamedItemIndex.objects.using(scope.using)
        .filter(
            document_id=selected.document.pkid,
            generation_id=selected.generation_key,
        )
        .annotate(size=Length("payload"), payload_sha=SourceSHA256(F("payload")))
        .filter(size__lte=MAX_PAYLOAD)
    )
    # Header strings and the optional legacy scalar have fixed maximum lengths.
    if not scope.runtime.budget.reserve_text(256, kind="materialized"):
        return None
    value = header.values_list(
        "binding", "member_digest", "payload_sha", "size"
    ).first()
    if value is None:
        return None
    binding, member_sha, payload_sha, size = value
    if owner_binding(selected, member_sha, payload_sha) != binding:
        return None
    rows = _active_chunk_predicates(
        TextChunk.objects.using(scope.using).filter(
            doc_id=selected.document.id, modality=TextChunk.Modality.TEXT
        ),
        snapshot=RepresentationSnapshotV1((selected,)),
    )
    if selected.generation_key is None and _legacy_membership(rows) != member_sha:
        return None
    return (
        header.filter(
            binding=binding, member_digest=member_sha, payload_sha=payload_sha
        ),
        rows,
        size,
        binding,
    )


def current_revisions(rows, ids):
    # Body hashes are computed only for the at most three shortlisted chunks.
    data = (
        rows.filter(pk__in=ids)
        .annotate(
            body_sha=SourceSHA256(F("content")),
            native_md5=MD5("content"),
            body_size=Length("content"),
        )
        .filter(
            body_size__lte=MAX_CANONICAL,
            body_size=F("end_position") - F("start_position"),
        )
        .values_list(
            "pk",
            "chunk_number",
            "start_position",
            "end_position",
            "body_sha",
            "native_md5",
            "body_size",
        )
    )
    return {
        pk: (number, start, end, sha, md5, size)
        for pk, number, start, end, sha, md5, size in data
    }
